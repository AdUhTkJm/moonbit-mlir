#include "ASTNode.h"

using namespace mbt;

std::string ASTNode::typeString() const {
  return type ? type->toString() : "<unknown>";
}

std::string IntLiteralNode::toString() const {
  return std::format("IntLiteral  {} : {}", value, typeString());
}

bool IntLiteralNode::walk(ASTWalkerWithDepth walker, int depth) {
  return walker(this, depth);
}

std::string VarDeclNode::toString() const {
  return std::format("VarDecl  {} : {}", name, typeString());
}

bool VarDeclNode::walk(ASTWalkerWithDepth walker, int depth) {
  return walker(this, depth) && (!init || init->walk(walker, depth + 1));
}

std::string FnDeclNode::toString() const {
  return std::format("FnDecl  {} : {}", name, typeString());
}

bool FnDeclNode::walk(ASTWalkerWithDepth walker, int depth) {
  if (!walker(this, depth))
    return false;
  
  for (auto param : params)
    if (!param->walk(walker, depth + 1))
      return false;
  
  return body->walk(walker, depth + 1);
}

std::string BlockNode::toString() const {
  return "Block";
}

bool BlockNode::walk(ASTWalkerWithDepth walker, int depth) {
  if (!walker(this, depth))
    return false;
  
  for (auto stmt : body)
    if (!stmt->walk(walker, depth + 1))
      return false;

  return true;
}

std::string BinaryNode::toString() const {
  static std::map<BinaryNode::Type, const char*> literals = {
    { BinaryNode::Add, "+" },
    { BinaryNode::Sub, "-" },
    { BinaryNode::Mul, "*" },
    { BinaryNode::Div, "/" },
    { BinaryNode::Mod, "%" },
    { BinaryNode::And, "&" },
    { BinaryNode::Or , "|" },
    { BinaryNode::Xor, "^" },
    { BinaryNode::Lt , "<" },
    { BinaryNode::Gt , ">" },
    { BinaryNode::Eq , "==" },
    { BinaryNode::Ne , "!=" },
  };

  return std::format("BinaryOp  {} : {}", literals[op], typeString());
}

bool BinaryNode::walk(ASTWalkerWithDepth walker, int depth) {
  return walker(this, depth) && l->walk(walker, depth + 1) && r->walk(walker, depth + 1);
}

std::string UnaryNode::toString() const {
  static std::map<UnaryNode::Type, const char*> literals = {
    { UnaryNode::Plus, "+" },
    { UnaryNode::Minus, "-" },
  };

  return std::format("UnaryOp  {}", literals[op]);
}

bool UnaryNode::walk(ASTWalkerWithDepth walker, int depth) {
  return walker(this, depth) && child->walk(walker, depth + 1);
}

std::string IfNode::toString() const {
  return "If";
}

bool IfNode::walk(ASTWalkerWithDepth walker, int depth) {
  return walker(this, depth)
      && cond->walk(walker, depth + 1)
      && ifso->walk(walker, depth + 1)
      && ifnot->walk(walker, depth + 1);
}

std::string VarNode::toString() const {
  return std::format("Var  {} : {}", name, typeString());
}

bool VarNode::walk(ASTWalkerWithDepth walker, int depth) {
  return walker(this, depth);
}

std::string IntrinsicNode::toString() const {
  return std::format("Intrinsic  {} : {}", intrinsic, typeString());
}

bool IntrinsicNode::walk(ASTWalkerWithDepth walker, int depth) {
  return walker(this, depth);
}

std::string FnCallNode::toString() const {
  return std::format("Call : {}", typeString());
}

bool FnCallNode::walk(ASTWalkerWithDepth walker, int depth) {
  if (!(walker(this, depth) && func->walk(walker, depth + 1)))
    return false;
  
  for (auto arg : args)
    if (!arg->walk(walker, depth + 1))
      return false;

  return true;
}

std::string StructDeclNode::toString() const {
  return std::format("Struct  {}", name);
}

bool StructDeclNode::walk(ASTWalkerWithDepth walker, int depth) {
  return walker(this, depth);
}

std::string MemAccessNode::toString() const {
  return std::format("AccessMember  {} : {}", memName, typeString());
}

bool MemAccessNode::walk(ASTWalkerWithDepth walker, int depth) {
  return walker(this, depth) && record->walk(walker, depth + 1);
}

std::string InitializerNode::toString() const {
  return std::format("Initializer : {}", typeString());
}

bool InitializerNode::walk(ASTWalkerWithDepth walker, int depth) {
  walker(this, depth);
  for (auto [_, node] : inits)
    if (!node->walk(walker, depth + 1))
      return false;

  return true;
}

std::string AssignNode::toString() const {
  return "Assign";
}

bool AssignNode::walk(ASTWalkerWithDepth walker, int depth) {
  return walker(this, depth) && lhs->walk(walker, depth + 1) && rhs->walk(walker, depth + 1);
}

void ASTNode::dump() {
  walk([&](ASTNode *node, int depth) {
    std::cerr << std::string(depth * 2, ' ') << node->toString() << "\n";
    return true;
  });
}

bool ASTNode::walk(ASTWalker walker) {
  return walk([&](ASTNode *node, int) {
    return walker(node);
  });
}
