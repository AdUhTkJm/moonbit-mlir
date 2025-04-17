#include "CGModule.h"
#include "lib/parse/ASTNode.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "lib/dialect/MoonOps.h"
#include "mlir/IR/Builders.h"

using namespace mbt;
using namespace mlir;

CGModule::SemanticScope::SemanticScope(CGModule &cgm):
  oldTable(cgm.symbolTable), cgm(cgm) {}

CGModule::SemanticScope::~SemanticScope() {
  cgm.symbolTable = oldTable;
}

CGModule::CGModule(MLIRContext &ctx):
  ctx(ctx), builder(&ctx) {
  mlir::DialectRegistry registry;
  registry.insert<func::FuncDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<mir::MoonDialect>();

  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();

  // Initialize type cache.
  unitType = mir::UnitType::get(&ctx);
  boolType = mlir::IntegerType::get(&ctx, 1);
}

StringAttr CGModule::getStringAttr(llvm::StringRef str) {
  return mlir::StringAttr::get(&ctx, str);
}

mlir::Location CGModule::getLoc(Location loc) {
  return mlir::FileLineColLoc::get(getStringAttr(loc.filename), loc.line, loc.col);
}

mlir::Location CGModule::getLoc(ASTNode *node) {
  return mlir::FusedLoc::get(&ctx, { getLoc(node->begin), getLoc(node->end) });
}

mir::PointerType CGModule::pointerTo(mlir::Type ty) {
  return mir::PointerType::get(ty.getContext(), ty);
}

mlir::Value CGModule::createAlloca(mlir::Location loc, mlir::Type ty) {
  return builder.create<mir::AllocaOp>(loc, pointerTo(ty), ty);
}

mlir::Type CGModule::getTy(mbt::Type *ty) {
  if (auto intTy = dyn_cast<mbt::IntType>(ty))
    return mlir::IntegerType::get(&ctx, intTy->width);

  if (auto fnTy = dyn_cast<mbt::FunctionType>(ty)) {
    llvm::SmallVector<mlir::Type> paramTy;
    for (auto x : fnTy->paramTy)
      paramTy.push_back(getTy(x));
    
    return mlir::FunctionType::get(&ctx, paramTy, getTy(fnTy->retTy));
  }

  if (isa<mbt::UnitType>(ty))
    return unitType;

  if (isa<mbt::BoolType>(ty))
    return boolType;

  llvm::errs() << std::format("unhandled type: {}\n", ty->toString());
  assert(false && "NYI");
}

mlir::Value CGModule::emitIfExpr(IfNode *ifexpr) {
  auto loc = getLoc(ifexpr);

  mlir::Value cond = emitExpr(ifexpr->cond);
  bool withElse = ifexpr->ifnot;
  
  // This `if` shouldn't return anything.
  auto op = builder.create<scf::IfOp>(loc, getTy(ifexpr->type), cond, withElse);

  {
    // Generate if-branch.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(op.getBody());
    mlir::Value val = emitStmt(ifexpr->ifso);
    builder.create<scf::YieldOp>(loc, val);
  }
  if (withElse) {
    // Generate else-branch.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&op.getElseRegion().front());
    mlir::Value val = emitStmt(ifexpr->ifnot);
    builder.create<scf::YieldOp>(loc, val);
  }
  return op.getResult(0);
}

mlir::Value CGModule::emitBinaryExpr(BinaryNode *binary) {
  auto loc = getLoc(binary);

  mlir::Value l = emitExpr(binary->l);
  mlir::Value r = emitExpr(binary->r);
  switch (binary->op) {
  case BinaryNode::Add:
    return builder.create<arith::AddIOp>(loc, getTy(binary->type), l, r);
  case BinaryNode::Sub:
    return builder.create<arith::SubIOp>(loc, getTy(binary->type), l, r);
  case BinaryNode::Mul:
    return builder.create<arith::MulIOp>(loc, getTy(binary->type), l, r);
  case BinaryNode::Div:
    return builder.create<arith::DivSIOp>(loc, getTy(binary->type), l, r);
  case BinaryNode::Mod:
    return builder.create<arith::RemSIOp>(loc, getTy(binary->type), l, r);
  case BinaryNode::Lt:
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, l, r);
  case BinaryNode::Gt:
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, l, r);
  case BinaryNode::Eq:
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, l, r);
  case BinaryNode::Ne:
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, l, r);
  default: ;
  }
  assert(false && "NYI");
}

mlir::Value CGModule::emitCallExpr(FnCallNode *node) {
  auto loc = getLoc(node);
  mlir::Value fnCall = emitExpr(node->func);

  llvm::SmallVector<mlir::Value> argValues;
  for (auto x : node->args)
    argValues.push_back(emitExpr(x));
  
  auto callOp = builder.create<func::CallIndirectOp>(loc, fnCall, argValues);
  return callOp.getResult(0);
}

mlir::Value CGModule::getVariable(mlir::Location loc, const std::string &name) {
  if (symbolTable.contains(name)) {
    auto [value, mut] = symbolTable[name];
    if (mut) {
      auto ptrType = cast<mir::PointerType>(value.getType());
      return builder.create<mir::LoadOp>(loc, ptrType.getPointee(), value);
    }
    return value;
  }

  // This must have been a global.
  auto global = theModule.lookupSymbol(name);
  auto symref = SymbolRefAttr::get(global);

  if (auto fn = dyn_cast<func::FuncOp>(global))
    // We're referring to a function. We must emit a function pointer for this.
    return builder.create<mir::ClosureOp>(loc, fn.getFunctionType(), symref);

  // A normal variable.
  auto var = dyn_cast<mir::GlobalOp>(global);
  return builder.create<mir::FetchGlobalOp>(loc, var.getType(), symref);
}

mlir::Value CGModule::emitAssign(AssignNode *node) {
  auto loc = getLoc(node);
  auto value = emitExpr(node->rhs);
  auto variable = getVariable(loc, node->lhs->name.mangle());

  // If getVariable() finds a global, then it must be:
  //   moon.fetch_global @a
  // Now we have to transform it to
  //   %1 = moon.addressof @a
  //   moon.store %value, %1
  if (auto fetchGlobal = dyn_cast<mir::FetchGlobalOp>(variable.getDefiningOp())) {
    auto global = fetchGlobal.getGlobalName();
    OpBuilder builder(fetchGlobal);
    auto globalPtrTy = pointerTo(fetchGlobal.getResult().getType());
    auto globalAddr = builder.create<mir::AddressofOp>(loc, globalPtrTy, global);
    builder.create<mir::StoreOp>(loc, value, /*base=*/globalAddr);
    auto unit = builder.create<mir::GetUnitOp>(loc);
    fetchGlobal.erase();
    return unit;
  }

  // If getVariable() finds a local, then it emits:
  //   moon.load %base
  // We must transform this to
  //   moon.store %value, %base
  auto load = cast<mir::LoadOp>(variable.getDefiningOp());
  auto base = load.getBase();
  OpBuilder builder(load);
  builder.create<mir::StoreOp>(loc, value, base);
  auto unit = builder.create<mir::GetUnitOp>(loc);
  load.erase();
  return unit;
}

mlir::Value CGModule::emitExpr(ASTNode *node) {
  assert(node);

  if (auto binary = dyn_cast<BinaryNode>(node))
    return emitBinaryExpr(binary);

  if (auto assign = dyn_cast<AssignNode>(node))
    return emitAssign(assign);

  if (auto ifexpr = dyn_cast<IfNode>(node))
    return emitIfExpr(ifexpr);

  if (auto intLiteral = dyn_cast<IntLiteralNode>(node)) {
    auto loc = getLoc(node);
    int value = intLiteral->value;
    return builder.create<arith::ConstantIntOp>(loc, value, getTy(intLiteral->type));
  }

  if (auto var = dyn_cast<VarNode>(node))
    return getVariable(getLoc(node), var->name.mangle());

  if (auto call = dyn_cast<FnCallNode>(node))
    return emitCallExpr(call);

  assert(false && "NYI");
}

mlir::Value CGModule::emitStmt(ASTNode *node) {
  assert(node);

  if (auto block = dyn_cast<BlockNode>(node)) {
    mlir::Value result;
    for (auto x : block->body)
      result = emitStmt(x);
    
    if (!block->body.size())
      result = builder.create<mir::GetUnitOp>(getLoc(node));
    return result;
  }

  if (auto varDecl = dyn_cast<VarDeclNode>(node)) {
    auto loc = getLoc(node);
    auto value = emitExpr(varDecl->init);
    auto mangledName = varDecl->name.mangle();
    if (varDecl->mut) {
      auto address = createAlloca(loc, value.getType());
      symbolTable[mangledName] = { .value = address, .mut = true };
      return builder.create<mir::GetUnitOp>(loc);
    }

    symbolTable[mangledName] = { .value = value, .mut = false };
    return builder.create<mir::GetUnitOp>(loc);
  }

  return emitExpr(node);
}

void CGModule::emitFunctionPrologue(func::FuncOp funcOp, FnDeclNode *fn) {
  for (auto [i, argDecl] : llvm::enumerate(fn->params))
    // Function parameters are never mutable.
    symbolTable[argDecl->name.mangle()] = {
      .value = funcOp.getArgument(i),
      .mut = false
    };
}

void CGModule::emitGlobalFn(FnDeclNode *globalFn) {
  // Resets builder's insertion point when destroyed.
  OpBuilder::InsertionGuard guard(builder);

  builder.setInsertionPointToEnd(theModule.getBody());
  auto type = cast<mlir::FunctionType>(getTy(globalFn->type));
  auto loc = getLoc(globalFn);
  auto funcOp = builder.create<func::FuncOp>(loc, globalFn->name.mangle(), type);
  builder.setInsertionPointToStart(funcOp.addEntryBlock());

  // Resets symbol table when destroyed.
  SemanticScope functionScope(*this);

  if (auto intrin = dyn_cast<IntrinsicNode>(globalFn->body)) {
    auto retTy = funcOp.getResultTypes()[0];
    SmallVector<mlir::Value> values;

    // Note we can't directly assign `getArguments()` to `values`.
    values.reserve(funcOp.getNumArguments());
    for (auto x : funcOp.getArguments())
      values.push_back(x);
    
    auto intrinOp = builder.create<mir::IntrinsicOp>(getLoc(intrin), retTy, intrin->intrinsic, values);
    builder.create<func::ReturnOp>(loc, intrinOp.getResult());
    return;
  }

  emitFunctionPrologue(funcOp, globalFn);

  mlir::Value value = emitStmt(globalFn->body);
  builder.create<func::ReturnOp>(loc, value);
}

ModuleOp CGModule::emitModule(ASTNode *n) {
  theModule = ModuleOp::create(getLoc(n));
  auto toplevels = cast<BlockNode>(n);
  for (auto one : toplevels->body) {
    if (auto fndecl = dyn_cast<FnDeclNode>(one)) {
      emitGlobalFn(fndecl);
    }
  }
  return theModule;
}

void CGModule::dump() {
  theModule.dump();
}
