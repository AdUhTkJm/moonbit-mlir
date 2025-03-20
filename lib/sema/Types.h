#ifndef TYPES_H
#define TYPES_H

#include "lib/utils/Common.h"

namespace mbt {

class Type {
public:
  int kind;
  constexpr static int typeKind = 0;
  virtual std::string toString() = 0;
  virtual ~Type() {}
};

class IntType : public Type {
public:
  int width;
  constexpr static int typeKind = 1;

  IntType(int width = 32): width(width) {
    kind = typeKind;
  }

  std::string toString() override { return "int"; }
};

class FunctionType : public Type {
public:
  std::vector<Type*> paramTy;
  Type *retTy;
  constexpr static int typeKind = 2;
  FunctionType(const std::vector<Type*> &paramTy, Type *retTy):
    paramTy(paramTy), retTy(retTy) {
    kind = typeKind;
  }

  std::string toString() override;
};

class WeakType : public Type {
public:
  int id;
  Type *real;
  constexpr static int typeKind = 3;
  WeakType(int id): id(id), real(nullptr) {
    kind = typeKind;
  }

  std::string toString() override;
};

class UnitType : public Type {
public:
  constexpr static int typeKind = 4;
  UnitType() {
    kind = typeKind;
  }

  std::string toString() override { return "unit"; }
};

class BoolType : public Type {
public:
  constexpr static int typeKind = 5;
  BoolType() {
    kind = typeKind;
  }

  std::string toString() override { return "bool"; }
};

// This is a user-defined type that we don't yet know about.
// Parser will collect all information of known types,
// and Sema will resolve them.
class UnresolvedType : public Type {
public:
  constexpr static int typeKind = 6;

  std::string name;
  std::vector<std::string> typeArgs;
  UnresolvedType() {
    kind = typeKind;
  }

  std::string toString() override { return "<unresolved>"; }
};

template<class T>
bool isa(Type *t) {
  assert(t);
  return t->kind == T::typeKind;
}

template<class T>
T *cast(Type *t) {
  assert(isa<T>(t));
  return (T*) t;
}

template<class T>
T *dyn_cast(Type *t) {
  if (!isa<T>(t))
    return nullptr;
  return (T*) t;
}

} // namespace mbt

#endif
