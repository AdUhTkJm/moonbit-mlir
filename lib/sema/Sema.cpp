#include "Sema.h"
#include "lib/utils/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"

using namespace mbt;

TypeInferrer::SemanticScope::SemanticScope(TypeInferrer &inferrer):
  oldMap(inferrer.typeMap), inferrer(inferrer) { }

TypeInferrer::SemanticScope::~SemanticScope() {
  inferrer.typeMap = oldMap;
}

Type *TypeInferrer::fresh() {
  return new WeakType(id++);
}

// Checks if `t1` occurs in `t2`.
bool occursCheck(Type *t1, Type *t2) {
  if (t1 == t2)
    return true;

  if (auto fnType = dyn_cast<FunctionType>(t2)) {
    for (auto param : fnType->paramTy)
      if (occursCheck(t1, param))
        return true;
    return occursCheck(t1, fnType->retTy);
  }

  return false;
}

bool TypeInferrer::unify(Type *&t1, Type *&t2) {
  t1 = repr(t1);
  t2 = repr(t2);
  if (t1 == t2)
    return true;

  if (auto weak = dyn_cast<WeakType>(t1)) {
    if (occursCheck(weak, t2))
      return false;
    
    weak->real = t2;
    return true;
  }
  
  if (isa<WeakType>(t2))
    return unify(t2, t1);
  
  if (auto fun1 = dyn_cast<FunctionType>(t1)) {
    if (auto fun2 = dyn_cast<FunctionType>(t2)) {
      if (fun1->paramTy.size() != fun2->paramTy.size())
        return false;

      for (unsigned i = 0; i < fun1->paramTy.size(); i++)
        if (!unify(fun1->paramTy[i], fun2->paramTy[i]))
          return false;
      
      return unify(fun1->retTy, fun2->retTy);
    }
    return false;
  }

  if (t1->getKind() == t2->getKind())
    return true;

  return false;
}

Type *TypeInferrer::repr(Type *ty) {
  // We don't want to return nullptr for weak types not yet inferred.
  if (auto weak = dyn_cast<WeakType>(ty)) {
    if (weak->real)
      return weak->real = repr(weak->real);

    return weak;
  }

  if (auto fn = dyn_cast<FunctionType>(ty)) {
    std::vector<Type*> paramTy;
    for (auto x : fn->paramTy)
      paramTy.push_back(x = repr(x));
    fn->paramTy = paramTy;
    fn->retTy = repr(fn->retTy);
    return fn;
  }
  
  return ty;
}

Type *TypeInferrer::inferFn(FnDeclNode *fn) {
  std::vector<Type*> paramsTy;
  Type *retTy = nullptr;

  {
    // We must record the function itself to typeMap, so we limit the scope
    // of this guard.
    SemanticScope scope(*this);
    for (auto param : fn->params)
      paramsTy.push_back(infer(param));

    retTy = infer(fn->body);
  }
  
  if (fn->type) {
    typeMap[fn->name.mangle()] = fn->type;

    auto fnTy = cast<FunctionType>(fn->type);
    if (fnTy->retTy && !unify(fnTy->retTy, retTy)) {
      Diagnostics::error(fn->begin, fn->end,
        std::format("return type {} does not match expression type {}",
          fnTy->retTy->toString(), retTy->toString()));
      
      return new UnitType();
    }
  }

  fn->type = new FunctionType(paramsTy, retTy);
  typeMap[fn->name.mangle()] = fn->type;
  return new UnitType();
}

Type *TypeInferrer::inferIf(IfNode *ifexpr) {
  if (!isa<BoolType>(infer(ifexpr->cond)))
    Diagnostics::error(ifexpr->cond->begin, ifexpr->cond->end,
      "expected if-condition to be of type bool");

  if (!ifexpr->ifnot) {
    if (!isa<UnitType>(infer(ifexpr->ifso))) {
      Diagnostics::error(ifexpr->ifso->begin, ifexpr->ifso->end,
        "if-statement without an else branch must return unit");
    }
    return ifexpr->type = new UnitType();
  }

  Type *ifsoTy = infer(ifexpr->ifso);
  Type *ifnotTy = infer(ifexpr->ifnot);
  if (!unify(ifsoTy, ifnotTy)) {
    Diagnostics::error(ifexpr->begin, ifexpr->end,
      "if-branch and else-branch return different types");
  }
  return ifexpr->type = ifsoTy;
}

Type *TypeInferrer::inferBinary(BinaryNode *binary) {
  Type *lty = infer(binary->l);
  Type *rty = infer(binary->r);
  if (!unify(lty, rty)) {
    Diagnostics::error(binary->begin, binary->end,
      std::format("{} and {} don't work together", lty->toString(), rty->toString()));
  }

  switch (binary->op) {
  case BinaryNode::Lt:
  case BinaryNode::Gt:
  case BinaryNode::Eq:
  case BinaryNode::Ne:
    // Comparisons.
    return binary->type = new BoolType();
  default:
    // Arithmetic operation.
    return binary->type = lty;
  }
}

Type *TypeInferrer::inferVarDecl(VarDeclNode *decl) {
  // This is a function parameter.
  if (!decl->init) {
    if (decl->type)
      return typeMap[decl->name.mangle()] = decl->type;

    return typeMap[decl->name.mangle()] = decl->type = fresh();
  }

  Type *ty = infer(decl->init);
  typeMap[decl->name.mangle()] = ty;
  if (decl->type && !unify(decl->type, ty)) {
    Diagnostics::error(decl->begin, decl->end,
      std::format("variable {} with type {} cannot accept this initializer (type {})",
        decl->name.mangle(), decl->type->toString(), ty->toString()));

    return new UnitType();
  }
  decl->type = ty;
  // Var declaration itself shouldn't have type.
  return new UnitType();
}

Type *TypeInferrer::inferCall(FnCallNode *call) {
  auto type = infer(call->func);
  auto fnTy = dyn_cast<FunctionType>(type);
  
  if (!fnTy) {
    assert(isa<WeakType>(type));

    std::vector<Type*> paramTy;
    paramTy.reserve(call->args.size());
    for (auto arg : call->args)
      paramTy.push_back(infer(arg));
    
    Type *retTy = fresh();
    Type *expected = new FunctionType(paramTy, retTy);
    
    bool success = unify(type, expected);
    assert(success);
    return call->type = retTy;
  }

  for (auto [i, arg] : llvm::enumerate(call->args)) {
    auto argTy = infer(arg);
    if (!unify(argTy, fnTy->paramTy[i]))
      Diagnostics::error(arg->begin, arg->end,
        std::format("argument type {} does not match parameter type {}",
          argTy->toString(), fnTy->paramTy[i]->toString()));
  }
  return call->type = fnTy->retTy;
}

Type *TypeInferrer::infer(ASTNode *node) {
  assert(node);
  
  if (auto var = dyn_cast<VarNode>(node)) {
    if (typeMap.contains(var->name.mangle()))
      return var->type = typeMap[var->name.mangle()];

    return var->type = fresh();
  }

  if (auto decl = dyn_cast<VarDeclNode>(node))
    return inferVarDecl(decl);

  if (auto binary = dyn_cast<BinaryNode>(node))
    return inferBinary(binary);

  if (auto intLit = dyn_cast<IntLiteralNode>(node))
    return intLit->type = new IntType();

  if (auto fn = dyn_cast<FnDeclNode>(node))
    return inferFn(fn);

  if (auto call = dyn_cast<FnCallNode>(node))
    return inferCall(call);

  if (auto ifexpr = dyn_cast<IfNode>(node))
    return inferIf(ifexpr);

  // We might be able to infer type based on intrinsic name in future.
  if (isa<IntrinsicNode>(node))
    return node->type = fresh();

  if (auto block = dyn_cast<BlockNode>(node)) {
    SemanticScope scope(*this);

    Type *lastType = nullptr;
    for (auto x : block->body)
      lastType = infer(x);
    
    // For empty block.
    if (!lastType)
      return node->type = new UnitType();

    return node->type = lastType;
  }

  assert(false);
}

void TypeInferrer::tidy(ASTNode *node) {
  node->walk([&](ASTNode *subnode) {
    subnode->type = repr(subnode->type);

    if (subnode->type->isWeak())
      Diagnostics::error(subnode->begin, subnode->end,
        "unable to deduce this type");

    return true;
  });
}
