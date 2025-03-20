#include "CGModule.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "lib/dialect/MoonOps.h"

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

  llvm::errs() << ty->toString() << "\n";
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

mlir::Value CGModule::emitExpr(ASTNode *node) {
  assert(node);

  if (auto binary = dyn_cast<BinaryNode>(node))
    return emitBinaryExpr(binary);

  if (auto ifexpr = dyn_cast<IfNode>(node))
    return emitIfExpr(ifexpr);

  if (auto intLiteral = dyn_cast<IntLiteralNode>(node)) {
    auto loc = getLoc(node);
    int value = intLiteral->value;
    auto op = builder.create<arith::ConstantIntOp>(loc, value, getTy(intLiteral->type));
    return op;
  }

  if (auto var = dyn_cast<VarNode>(node)) {
    assert(symbolTable.contains(var->name));
    return symbolTable[var->name];
  }

  assert(false && "NYI");
}

mlir::Value CGModule::emitStmt(ASTNode *node) {
  assert(node);

  if (auto block = dyn_cast<BlockNode>(node)) {
    mlir::Value result;
    for (auto x : block->body)
      result = emitStmt(x);
    return result;
  }

  if (auto varDecl = dyn_cast<VarDeclNode>(node)) {
    auto loc = getLoc(node);
    auto value = emitExpr(varDecl->init);
    symbolTable[varDecl->name] = value;
    return builder.create<mir::GetUnitOp>(loc);
  }

  return emitExpr(node);
}

void CGModule::emitFunctionPrologue(func::FuncOp funcOp, FnDeclNode *fn) {
  for (auto [i, argDecl] : llvm::enumerate(fn->params))
    symbolTable[argDecl->name] = funcOp.getArgument(i);
}

void CGModule::emitGlobalFn(FnDeclNode *globalFn) {
  // Resets builder's insertion point when destroyed.
  OpBuilder::InsertionGuard guard(builder);
  // Resets symbol table when destroyed.
  SemanticScope functionScope(*this);

  builder.setInsertionPointToEnd(theModule.getBody());
  auto type = cast<mlir::FunctionType>(getTy(globalFn->type));
  auto loc = getLoc(globalFn);
  auto funcOp = builder.create<func::FuncOp>(loc, globalFn->name, type);
  builder.setInsertionPointToStart(funcOp.addEntryBlock());

  emitFunctionPrologue(funcOp, globalFn);

  mlir::Value value = emitStmt(globalFn->body);
  builder.create<func::ReturnOp>(loc, value);
}

void CGModule::emitModule(ASTNode *n) {
  theModule = ModuleOp::create(getLoc(n));
  auto toplevels = cast<BlockNode>(n);
  for (auto one : toplevels->body) {
    if (auto fndecl = dyn_cast<FnDeclNode>(one)) {
      emitGlobalFn(fndecl);
    }
  }
}

void CGModule::dump() {
  theModule.dump();
}
