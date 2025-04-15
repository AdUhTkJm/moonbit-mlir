#ifndef AST_NODES_H
#define AST_NODES_H

#include "Identifier.h"
#include "lib/utils/Common.h"
#include "lib/sema/Types.h"
#include "llvm/ADT/StringRef.h"
#include <functional>

namespace mbt {

class ASTNode;

// Return false to stop walking.
using ASTWalker = std::function<bool (ASTNode*)>;
using ASTWalkerWithDepth = std::function<bool (ASTNode*, int)>;

class ASTNode {
  int kind;

protected:
  std::string typeString() const;
  
public:
  Type *type = nullptr;
  Location begin, end;

  int getKind() const { return kind; }

  ASTNode(int kind, Location begin, Location end):
    kind(kind), begin(begin), end(end) {}
  virtual ~ASTNode() {}

  // This only outputs the current layer;
  // further layers (eg. nodes inside a BlockNode) won't be output.
  // Use `dump()` for that.
  virtual std::string toString() const = 0;

  virtual bool walk(ASTWalkerWithDepth, int = 0) = 0;
  bool walk(ASTWalker);
  void dump();
};

template<class T, int NodeType>
class ASTNodeImpl : public ASTNode {
public:
  static bool classof(const ASTNode *node) {
    return node->getKind() == NodeType;
  }

  ASTNodeImpl(Location begin, Location end):
    ASTNode(NodeType, begin, end) {}

  std::string toString() const override = 0;
  virtual bool walk(ASTWalkerWithDepth, int) override = 0;
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

  std::string toString() const override;
  bool walk(ASTWalkerWithDepth, int) override;
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
  bool walk(ASTWalkerWithDepth, int) override;
};

class VarDeclNode : public ASTNodeImpl<VarDeclNode, 3> {
public:
  Identifier name;
  ASTNode *init;

  VarDeclNode(llvm::StringRef name, ASTNode *init, Location begin, Location end):
    ASTNodeImpl(begin, end), name(name), init(init) {}

  std::string toString() const override;
  bool walk(ASTWalkerWithDepth, int) override;
};

class FnDeclNode : public ASTNodeImpl<FnDeclNode, 4> {
public:
  constexpr static int nodeType = 4;
  Identifier name;
  // For a method, this is the name of the struct to which this `fn` belongs.
  std::optional<std::string> belongsTo;
  ASTNode *body;
  std::vector<VarDeclNode*> params;

  FnDeclNode(llvm::StringRef name, const std::vector<VarDeclNode*> &params,
             ASTNode *body, Location begin, Location end):
    ASTNodeImpl(begin, end), name(name), belongsTo(std::nullopt), body(body), params(params)  {}

  FnDeclNode(llvm::StringRef belongsTo, llvm::StringRef name, const std::vector<VarDeclNode*> &params,
             ASTNode *body, Location begin, Location end):
    ASTNodeImpl(begin, end), name(name), belongsTo(belongsTo), body(body), params(params)  {}

  std::string toString() const override;
  bool walk(ASTWalkerWithDepth, int) override;
};

class IntLiteralNode : public ASTNodeImpl<IntLiteralNode, 5> {
public:
  constexpr static int nodeType = 5;
  int value;

  IntLiteralNode(int value, Location begin, Location end):
    ASTNodeImpl(begin, end), value(value) {}

  std::string toString() const override;
  bool walk(ASTWalkerWithDepth, int) override;
};

class BlockNode : public ASTNodeImpl<BlockNode, 6> {
public:
  std::vector<ASTNode*> body;

  BlockNode(Location begin, Location end): ASTNodeImpl(begin, end) {}

  std::string toString() const override;
  bool walk(ASTWalkerWithDepth, int) override;
};

class VarNode : public ASTNodeImpl<VarNode, 7> {
public:
  Identifier name;

  VarNode(Identifier name, Location begin, Location end):
    ASTNodeImpl(begin, end), name(name) {}

  std::string toString() const override;
  bool walk(ASTWalkerWithDepth, int) override;
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
  bool walk(ASTWalkerWithDepth, int) override;
};

class IntrinsicNode : public ASTNodeImpl<IntrinsicNode, 9> {
public:
  std::string intrinsic;
  
  IntrinsicNode(Identifier name, Location begin, Location end):
    ASTNodeImpl(begin, end), intrinsic(name) {}

  std::string toString() const override;
  bool walk(ASTWalkerWithDepth, int) override;
};

class FnCallNode : public ASTNodeImpl<FnCallNode, 10> {
public:
  ASTNode *func;
  std::vector<ASTNode*> args;
  
  FnCallNode(ASTNode *func, std::vector<ASTNode*> args, Location begin, Location end):
    ASTNodeImpl(begin, end), func(func), args(args) {}

  std::string toString() const override;
  bool walk(ASTWalkerWithDepth, int) override;
};

class StructDeclNode : public ASTNodeImpl<StructDeclNode, 11> {
public:
  Identifier name;
  std::vector<std::pair<std::string, Type*>> fields;
  
  StructDeclNode(Identifier name, const decltype(fields) &fields, Location begin, Location end):
    ASTNodeImpl(begin, end), name(name), fields(fields) {}

  std::string toString() const override;
  bool walk(ASTWalkerWithDepth, int) override;
};

class MemAccessNode : public ASTNodeImpl<MemAccessNode, 12> {
public:
  ASTNode *record;
  Identifier memName;
  
  MemAccessNode(ASTNode *record, Identifier memName, Location begin, Location end):
    ASTNodeImpl(begin, end), record(record), memName(memName) {}

  std::string toString() const override;
  bool walk(ASTWalkerWithDepth, int) override;
};

class InitializerNode : public ASTNodeImpl<InitializerNode, 12> {
public:
  std::vector<std::pair<Identifier, ASTNode*>> inits;
  
  InitializerNode(const decltype(inits) &inits, Location begin, Location end):
    ASTNodeImpl(begin, end), inits(inits) {}

  std::string toString() const override;
  bool walk(ASTWalkerWithDepth, int) override;
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
