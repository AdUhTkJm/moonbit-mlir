#ifndef AST_NODES_H
#define AST_NODES_H

#include "lib/utils/Common.h"
#include "lib/sema/Types.h"
#include "llvm/ADT/StringRef.h"
#include <functional>

namespace mbt {

class ASTNode {
  int kind;
public:
  Type *type = nullptr;
  Location begin, end;

  int getKind() const { return kind; }

  ASTNode(int kind, Location begin, Location end):
    kind(kind), begin(begin), end(end) {}
  virtual ~ASTNode() {}

  // This only outputs the current layer;
  // further layers (for example, nodes inside a BlockNode)
  // shouldn't be outputted.
  // Use `dump()` for that.
  virtual std::string toString() const = 0;
};

template<class T, int NodeType>
class ASTNodeImpl : public ASTNode {
public:
  template<class Fn>
  void walk(Fn &&walker) {
    static_cast<T*>(this)->walkImpl(walker);
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == NodeType;
  }

  ASTNodeImpl(Location begin, Location end):
    ASTNode(NodeType, begin, end) {}

  std::string toString() const override = 0;
};

class BinaryNode : public ASTNodeImpl<BinaryNode, 1> {
public:
  enum Type {
    Add, Sub, Mul, Div, Mod, And, Or, Xor,
    Lt, Gt, Eq, Ne
  } op;
  ASTNode *l, *r;
  BinaryNode(Type op, ASTNode *l, ASTNode *r, Location begin, Location end):
    ASTNodeImpl(begin, end), op(op), l(l), r(r) {}

  template<class Fn>
  void walkImpl(Fn &&walker) {
    walker(this);
    walker(l);
    walker(r);
  }

  std::string toString() const override;
};

class UnaryNode : public ASTNodeImpl<UnaryNode, 2> {
public:
  enum Type {
    Plus, Minus
  } op;
  ASTNode *child;
  
  UnaryNode(Type op, ASTNode *child, Location begin, Location end):
    ASTNodeImpl(begin, end), op(op), child(child) {}
    
  std::string toString() const override;
};

class VarDeclNode : public ASTNodeImpl<VarDeclNode, 3> {
public:
  std::string name;
  ASTNode *init;
  VarDeclNode(llvm::StringRef name, ASTNode *init, Location begin, Location end):
    ASTNodeImpl(begin, end), name(name), init(init) {}

  std::string toString() const override;
};

class FnDeclNode : public ASTNodeImpl<FnDeclNode, 4> {
public:
  constexpr static int nodeType = 4;
  std::string name;
  ASTNode *body;
  std::vector<VarDeclNode*> params;
  FnDeclNode(llvm::StringRef name, const std::vector<VarDeclNode*> &params,
             ASTNode *body, Location begin, Location end):
    ASTNodeImpl(begin, end), name(name), body(body), params(params)  {}

  std::string toString() const override;
};

class IntLiteralNode : public ASTNodeImpl<IntLiteralNode, 5> {
public:
  constexpr static int nodeType = 5;
  int value;
  IntLiteralNode(int value, Location begin, Location end):
    ASTNodeImpl(begin, end), value(value) {}

  std::string toString() const override;
};

class BlockNode : public ASTNodeImpl<BlockNode, 6> {
public:
  std::vector<ASTNode*> body;
  BlockNode(Location begin, Location end): ASTNodeImpl(begin, end) {}

  std::string toString() const override;
};

class VarNode : public ASTNodeImpl<VarNode, 7> {
public:
  std::string name;
  VarNode(std::string name, Location begin, Location end):
    ASTNodeImpl(begin, end), name(name) {}

  std::string toString() const override;
};

class IfNode : public ASTNodeImpl<IfNode, 8> {
public:
  ASTNode *cond;
  ASTNode *ifso;
  ASTNode *ifnot;

  IfNode(ASTNode *cond, ASTNode *ifso, Location begin, Location end):
    ASTNodeImpl(begin, end), cond(cond), ifso(ifso), ifnot(nullptr) {}

  IfNode(ASTNode *cond, ASTNode *ifso, ASTNode *ifnot, Location begin, Location end):
    ASTNodeImpl(begin, end), cond(cond), ifso(ifso), ifnot(ifnot) {}

  std::string toString() const override;
};

template<class T>
bool isa(ASTNode *node) {
  assert(node);
  return T::classof(node);
}

template<class T>
T* cast(ASTNode *node) {
  assert(isa<T>(node));
  return (T*) node;
}

template<class T>
T* dyn_cast(ASTNode *node) {
  return isa<T>(node) ? cast<T>(node) : nullptr;
}

};

#endif
