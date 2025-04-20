#ifndef PARSER_H
#define PARSER_H

#include "Identifier.h"
#include "Lexer.h"
#include "ASTNode.h"
#include <optional>

namespace mbt {

class Parser {
  const std::vector<Token> tokens;
  int place = 0;

  bool checkEOF();

  Token consume();
  Token expect(Token::Type ty);
  Token peek();
  Token last();

  // Returns true if the current token is one of the `tys`.
  template<class... T>
  requires (... && std::is_same_v<T, Token::Type>)
  bool peek(T... tys) {
    auto peeked = peek().ty;
    return ((peeked == tys) || ...);
  }

  // Returns true if the current token is one of the `tys`, and consumes the token.
  template<class... T>
  requires (... && std::is_same_v<T, Token::Type>)
  bool test(T... tys) {
    if (peek(tys...)) {
      consume();
      return true;
    }
    return false;
  }

  // Lookahead from current position (inclusive).
  template<class... T>
  requires (... && std::is_same_v<T, Token::Type>)
  bool lookahead(T... tys) {
    constexpr size_t count = sizeof...(T);
    if (place + count > tokens.size()) return false;

    Token::Type expected[] = { tys... };
    for (size_t i = 0; i < count; ++i) {
      if (tokens[place + i].ty != expected[i])
        return false;
    }
    return true;
  }

  // Parse the next identifier.
  std::optional<Identifier> getIdentifier();
  Identifier expectIdentifier();
  Identifier expectUnqualifiedIdentifier();

  ASTNode *primary();
  ASTNode *memAccessExpr();
  ASTNode *callExpr();
  ASTNode *ifExpr();
  ASTNode *compareExpr();
  ASTNode *mulExpr();
  ASTNode *addExpr();
  ASTNode *blockExpr(); // Note that every block can return some value.
  ASTNode *expr();
  ASTNode *stmt();
  ASTNode *toplevel();
  ASTNode *topFn();
  ASTNode *topStruct();

  ASTNode *structLiteralExpr();
  ASTNode *assignStmt(ASTNode *lhs);

  mbt::Type *parseType();

public:
  Parser(const std::vector<Token> &tokens): tokens(tokens) {}
  ASTNode *parse();
};

} // namespace mbt

#endif
