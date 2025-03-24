#include "Types.h"
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
