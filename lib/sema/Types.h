#ifndef TYPES_H
#define TYPES_H

#include "lib/utils/Common.h"
#include <functional>

namespace mbt {

class Type;
using TypeWalker = std::function<void (Type*)>;

class Type {
  int kind;
public:
  int getKind() const { return kind; }

  Type(int kind): kind(kind) {}
  virtual ~Type() {}

  // Unlike AST nodes, this returns the full type.
  virtual std::string toString() const = 0;
  virtual void walk(TypeWalker) = 0;

  // Returns true if any part of this type is weak.
  bool isWeak();
};

template<class T, int TypeKind>
class TypeImpl : public Type {
public:
  template<class Fn>
  void walk(Fn &&walker) {
    static_cast<T*>(this)->walkImpl(walker);
  }

  static bool classof(const Type *type) {
    return type->getKind() == TypeKind;
  }

  TypeImpl(): Type(TypeKind) {}

  std::string toString() const override = 0;
  virtual void walk(TypeWalker) override = 0;
};

class IntType : public TypeImpl<IntType, 1> {
public:
  int width;

  IntType(int width = 32): width(width) { }

  std::string toString() const override { return "int"; }
  virtual void walk(TypeWalker) override;
};

class FunctionType : public TypeImpl<FunctionType, 2> {
public:
  std::vector<Type*> paramTy;
  Type *retTy;
  
  FunctionType(const std::vector<Type*> &paramTy, Type *retTy):
    paramTy(paramTy), retTy(retTy) { }

  std::string toString() const override;
  virtual void walk(TypeWalker) override;
};

class WeakType : public TypeImpl<WeakType, 3> {
public:
  int id;
  Type *real;
  
  WeakType(int id): id(id), real(nullptr) { }

  std::string toString() const override;
  virtual void walk(TypeWalker) override;
};

class UnitType : public TypeImpl<UnitType, 4> {
public:
  UnitType() { }

  std::string toString() const override { return "unit"; }
  virtual void walk(TypeWalker) override;
};

class BoolType : public TypeImpl<BoolType, 5> {
public:
  BoolType() { }

  std::string toString() const override { return "bool"; }
  virtual void walk(TypeWalker) override;
};

// This is a user-defined type that we don't yet know about.
// Parser will collect all information of known types,
// and Sema will resolve them.
class UnresolvedType : public TypeImpl<UnresolvedType, 6> {
public:
  std::string name;
  std::vector<std::string> typeArgs;
  UnresolvedType() { }

  std::string toString() const override { return "<unresolved>"; }
  virtual void walk(TypeWalker) override;
};

template<class T>
bool isa(Type *t) {
  assert(t);
  return T::classof(t);
}

template<class T>
T *cast(Type *t) {
  assert(isa<T>(t));
  return (T*) t;
}

template<class T>
T *dyn_cast(Type *t) {
  return isa<T>(t) ? cast<T>(t) : nullptr;
}

} // namespace mbt

#endif
