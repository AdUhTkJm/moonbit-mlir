#ifndef LEXER_H
#define LEXER_H

#include "lib/utils/Common.h"
#include "lib/utils/Diagnostics.h"

namespace mbt {

class Token {
public:

#define TOKEN_TYPES(X) \
  X(Ident) X(IntLit) X(StrLit) \
  X(Eq) X(Ne) X(Le) X(Ge) X(Lt) X(Gt) \
  X(Plus) X(Minus) X(Mul) X(Div) X(Mod) \
  X(PlusEq) X(MinusEq) X(MulEq) X(DivEq) \
  X(BitAnd) X(BitOr) X(Xor) \
  X(BitAndEq) X(BitOrEq) X(XorEq) \
  X(And) X(Or) \
  X(Assign) X(Exclaim) \
  X(Semicolon) X(Colon) X(Arrow) \
  X(Comma) X(LPar) X(RPar) X(LBrak) X(RBrak) X(LBrace) X(RBrace) \
  X(If) X(Else) X(Let) X(While) X(For) X(Fn) \
  X(Int) X(Bool) X(FixedArray) \
  X(Return) \
  X(End)

#define X(name) name, 
  enum Type {
    TOKEN_TYPES(X)
  } ty;
#undef X

#define X(name) #name, 
  static constexpr const char* type_names[] = {
    TOKEN_TYPES(X)
  };
#undef X

  int vi;
  std::string vs;
  Location begin, end;

  Token(Type t, Location begin, int len);
  Token(Type t, std::string name, Location begin);
  Token(Type t, int vi, Location begin, int len);
};

const char *stringifyToken(Token t);

class Tokenizer {
private:
  std::string input, filename;
  // Index of `input`
  size_t loc;
  // Index of the last '\n' at the previous line
  size_t last_line;
  size_t line;
  
public:
  Tokenizer(const std::string &filename, const std::string& input);

  Token nextToken();
  bool hasMore() const;
};

} // namespace mbt

#endif
