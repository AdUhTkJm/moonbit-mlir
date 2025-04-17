#ifndef PARSER_H
#define PARSER_H

#include "lib/utils/Common.h"
#include "Lexer.h"
#include "ASTNode.h"

namespace mbt {

class Parser {
  const std::vector<Token> tokens;
  int place = 0;

  bool checkEOF();

  Token consume();
  Token expect(Token::Type ty);
  Token peek();
  Token last();
  bool peek(Token::Type ty);
  bool test(Token::Type ty);

  ASTNode *primary();
  ASTNode *memAccessExpr();
  ASTNode *callExpr();
  ASTNode *ifExpr();
  ASTNode *compareExpr();
  ASTNode *mulExpr();
  ASTNode *addExpr();
  ASTNode *expr();
  ASTNode *blockStmt();
  ASTNode *stmt();
  ASTNode *toplevel();
  ASTNode *topFn();
  ASTNode *topStruct();

  ASTNode *assignStmt(ASTNode *lhs);

  mbt::Type *parseType();

public:
  Parser(const std::vector<Token> &tokens): tokens(tokens) {}
  ASTNode *parse();
};

} // namespace mbt

#endif