#ifndef TYPES_H
#define TYPES_H

#include "lib/parse/Identifier.h"
#include <functional>

namespace mbt {

class Type;
using TypeWalker = std::function<void (Type*)>;

class Type {
  int kind;
public:
  friend bool operator==(const Type &a, const Type &b);;

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

class IntType : public TypeImpl<IntType, __LINE__> {
public:
  int width;

  IntType(int width = 32): width(width) { }

  std::string toString() const override { return "int"; }
  void walk(TypeWalker) override;
};

class FunctionType : public TypeImpl<FunctionType, __LINE__> {
public:
  std::vector<Type*> paramTy;
  Type *retTy;
  
  FunctionType(const std::vector<Type*> &paramTy, Type *retTy):
    paramTy(paramTy), retTy(retTy) { }

  std::string toString() const override;
  void walk(TypeWalker) override;
};

class WeakType : public TypeImpl<WeakType, __LINE__> {
public:
  int id;
  Type *real;
  
  WeakType(int id): id(id), real(nullptr) { }

  std::string toString() const override;
  void walk(TypeWalker) override;
};

class UnitType : public TypeImpl<UnitType, __LINE__> {
public:
  UnitType() { }

  std::string toString() const override { return "unit"; }
  void walk(TypeWalker) override;
};

class BoolType : public TypeImpl<BoolType, __LINE__> {
public:
  BoolType() { }

  std::string toString() const override { return "bool"; }
  void walk(TypeWalker) override;
};

// This is a user-defined type that we don't yet know about.
// Parser will collect all information of known types,
// and Sema will resolve them.
class UnresolvedType : public TypeImpl<UnresolvedType, __LINE__> {
public:
  Identifier name;
  std::vector<std::string> typeArgs;
  UnresolvedType() { }
  UnresolvedType(Identifier name, const std::vector<std::string> typeArgs = {}):
    name(name), typeArgs(typeArgs) {}

  std::string toString() const override { return "<unresolved>"; }
  void walk(TypeWalker) override;
};

class StringType : public TypeImpl<StringType, __LINE__> {
public:
  StringType() { }

  std::string toString() const override { return "string"; }
  void walk(TypeWalker) override;
};

class StructType : public TypeImpl<StructType, __LINE__> {
public:
  std::string name;
  std::vector<Type*> fields;
  // Generics.
  std::vector<Type*> typeArgs;

  StructType(std::string name, const std::vector<Type*> &fields, const std::vector<Type*> &typeArgs = {}):
    name(name), fields(fields), typeArgs(typeArgs) {}

  std::string toString() const override;
  void walk(TypeWalker) override;
};
  
template<class T>
bool isa(const Type *t) {
  assert(t);
  return T::classof(t);
}

template<class T>
T *cast(Type *t) {
  assert(isa<T>(t));
  return (T*) t;
}

template<class T>
const T *cast(const Type *t) {
  assert(isa<T>(t));
  return (const T*) t;
}

template<class T>
T *dyn_cast(Type *t) {
  return isa<T>(t) ? cast<T>(t) : nullptr;
}

template<class T>
const T* dyn_cast(const Type *t) {
  return isa<T>(t) ? cast<T>(t) : nullptr;
}

} // namespace mbt

#endif
