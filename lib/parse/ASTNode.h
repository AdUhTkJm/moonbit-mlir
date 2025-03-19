#ifndef AST_NODES_H
#define AST_NODES_H

#include "lib/utils/Common.h"
#include "lib/sema/Types.h"
#include "llvm/ADT/StringRef.h"
#include <functional>

namespace mbt {

class ASTNode {
public:
  int kind;
  Type *type = nullptr;
  Location begin, end;
  constexpr static int nodeType = 0;

  ASTNode(int kind, Location begin, Location end):
    kind(kind), begin(begin), end(end) {}
};

class BinaryNode : public ASTNode {
public:
  constexpr static int nodeType = 1;
  enum Type {
    Add, Sub, Mul, Div, Mod, And, Or, Xor,
    Lt, Gt, Eq, Ne
  } op;
  ASTNode *l, *r;
  BinaryNode(Type op, ASTNode *l, ASTNode *r, Location begin, Location end):
    ASTNode(nodeType, begin, end), op(op), l(l), r(r) {}
};

class UnaryNode : public ASTNode {
public:
  constexpr static int nodeType = 2;
  enum Type {
    Plus, Minus
  } op;
  ASTNode *child;
  UnaryNode(Type op, ASTNode *child, Location begin, Location end):
    ASTNode(nodeType, begin, end), op(op), child(child) {}
};

class VarDeclNode : public ASTNode {
public:
  constexpr static int nodeType = 3;
  std::string name;
  ASTNode *init;
  VarDeclNode(llvm::StringRef name, ASTNode *init, Location begin, Location end):
    ASTNode(nodeType, begin, end), name(name), init(init)  {}
};

class FnDeclNode : public ASTNode {
public:
  constexpr static int nodeType = 4;
  std::string name;
  ASTNode *body;
  std::vector<VarDeclNode*> params;
  FnDeclNode(llvm::StringRef name, const std::vector<VarDeclNode*> &params,
             ASTNode *body, Location begin, Location end):
    ASTNode(nodeType, begin, end), name(name), body(body), params(params)  {}
};

class IntLiteralNode : public ASTNode {
public:
  constexpr static int nodeType = 5;
  int value;
  IntLiteralNode(int value, Location begin, Location end):
    ASTNode(nodeType, begin, end), value(value) {}
};

class BlockNode : public ASTNode {
public:
  constexpr static int nodeType = 6;

  std::vector<ASTNode*> body;
  BlockNode(Location begin, Location end): ASTNode(nodeType, begin, end) {}
};

class VarNode : public ASTNode {
public:
  constexpr static int nodeType = 7;

  std::string name;
  VarNode(std::string name, Location begin, Location end):
    ASTNode(nodeType, begin, end), name(name) {}
};

class IfNode : public ASTNode {
public:
  constexpr static int nodeType = 8;

  ASTNode *cond;
  ASTNode *ifso;
  ASTNode *ifnot;

  IfNode(ASTNode *cond, ASTNode *ifso, Location begin, Location end):
    ASTNode(nodeType, begin, end), cond(cond), ifso(ifso), ifnot(nullptr) {}

  IfNode(ASTNode *cond, ASTNode *ifso, ASTNode *ifnot, Location begin, Location end):
    ASTNode(nodeType, begin, end), cond(cond), ifso(ifso), ifnot(ifnot) {}
};

template<class T>
bool isa(ASTNode *node) {
  assert(node);
  return node->kind == T::nodeType;
}

template<class T>
T* cast(ASTNode *node) {
  assert(isa<T>(node));
  return (T*) node;
}

template<class T>
T* dyn_cast(ASTNode *node) {
  if (isa<T>(node))
    return (T*) node;
  return nullptr;
}

};

#endif
