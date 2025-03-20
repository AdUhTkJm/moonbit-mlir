#include "ASTNode.h"

using namespace mbt;

std::string IntLiteralNode::toString() const {
  return std::format("int-literal  {}", value);
}

std::string VarDeclNode::toString() const {
  return std::format("var-decl  {}: {}", name, type ? type->toString() : "<unknown>");
}

std::string FnDeclNode::toString() const {
  return std::format("fn-decl  {}", name);
}

std::string BlockNode::toString() const {
  return "block";
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

  return std::format("binary-op  {}", literals[op]);
}

std::string IfNode::toString() const {
  return "if";
}

std::string VarNode::toString() const {
  return std::format("var  {}: {}", name, type ? type->toString() : "<unknown>");
}
