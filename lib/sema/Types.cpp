#include "Types.h"
#include <sstream>

using namespace mbt;

std::string FunctionType::toString() const {
  std::stringstream ss;
  for (auto x : paramTy)
    ss << x->toString() << ", ";
  auto str = ss.str();
  // Remove the extra ", " at the end
  if (str.size() > 2) {
    str.pop_back();
    str.pop_back();
  }
  return std::format("({}) -> {}", str, retTy->toString());
}

std::string WeakType::toString() const {
  return std::format("'{}", id);
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

void WeakType::walk(TypeWalker walker) {
  walker(this);
  if (real)
    real->walk(walker);
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
