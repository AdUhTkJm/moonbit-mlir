#include "Types.h"
#include "llvm/ADT/STLExtras.h"
#include <sstream>

using namespace mbt;

std::string interleave(const std::vector<Type*> types) {
  std::stringstream ss;
  for (auto x : types)
    ss << x->toString() << ", ";
  auto str = ss.str();
  // Remove the extra ", " at the end
  if (str.size() > 2) {
    str.pop_back();
    str.pop_back();
  }
  return str;
}

std::string FunctionType::toString() const {
  return std::format("({}) -> {}", interleave(paramTy), retTy->toString());
}

std::string WeakType::toString() const {
  return std::format("'{}", id);
}

std::string StructType::toString() const {
  // Double braces are escaping them.
  return std::format("{{{}}}[{}]", interleave(fields), interleave(typeArgs));
}

void IntType::walk(TypeWalker walker) {
  walker(this);
}

void BoolType::walk(TypeWalker walker) {
  walker(this);
}

void UnitType::walk(TypeWalker walker) {
  walker(this);
}

void StringType::walk(TypeWalker walker) {
  walker(this);
}

void UnresolvedType::walk(TypeWalker walker) {
  walker(this);
}

void WeakType::walk(TypeWalker walker) {
  walker(this);
  if (real)
    real->walk(walker);
}

void StructType::walk(TypeWalker walker) {
  walker(this);
  for (auto type : typeArgs)
    type->walk(walker);
}

void FunctionType::walk(TypeWalker walker) {
  walker(this);
  for (auto param : paramTy)
    param->walk(walker);
  retTy->walk(walker);
}

bool Type::isWeak() {
  bool weak = false;
  walk([&](Type *ty) {
    weak |= isa<WeakType>(ty);
  });
  return weak;
}

bool mbt::operator==(const Type &a, const Type &b) {
  if (a.kind != b.kind)
    return false;

  if (auto weakA = dyn_cast<WeakType>(&a)) {
    auto weakB = dyn_cast<WeakType>(&b);
    return weakA->id == weakB->id;
  }

  if (auto fnA = dyn_cast<FunctionType>(&a)) {
    auto fnB = dyn_cast<FunctionType>(&b);
    if (*fnA->retTy != *fnB->retTy)
      return false;

    if (fnA->paramTy.size() != fnB->paramTy.size())
      return false;

    for (auto [i, x] : llvm::enumerate(fnA->paramTy))
      if (*fnA->paramTy[i] != *fnB->paramTy[i])
        return false;

    return true;
  }

  if (auto sA = dyn_cast<StructType>(&a)) {
    auto sB = dyn_cast<StructType>(&b);
    return sA->name == sB->name;
  }

  return true;
}
