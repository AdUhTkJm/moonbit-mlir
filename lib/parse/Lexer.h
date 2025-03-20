#ifndef LEXER_H
#define LEXER_H

#include "lib/utils/Common.h"
#include "lib/utils/Diagnostics.h"

namespace mbt {

class Token {
public: 
  enum Type {
    // literals
    Ident, IntLit, StrLit,

    // operators
    Eq, Ne, Le, Ge, Lt, Gt,
    Plus, Minus, Mul, Div, Mod,
    PlusEq, MinusEq, MulEq, DivEq,
    BitAnd, BitOr, Xor,
    BitAndEq, BitOrEq, XorEq,
    And, Or,
    Assign, Exclaim, Semicolon, Colon, Arrow,
    Comma, LPar, RPar, LBrak, RBrak, LBrace, RBrace,

    // keywords
    If, Else, Let, While, For, Fn, Return,

    // types
    Int, Bool, FixedArray, Unit,

    // EOF
    End
  } ty;

  int vi;
  std::string vs;
  Location begin, end;

  Token(Type t, Location begin, int len);
  Token(Type t, std::string name, Location begin);
  Token(Type t, int vi, Location begin, int len);
};

const char *stringifyToken(Token t);
const char *stringifyToken(Token::Type t);

class Tokenizer {
private:
  std::string input, filename;
  // Index of `input`
  size_t loc;
  // Index of the last '\n' at the previous line
  size_t last_line;
  size_t line;

  // Get the escaped character.
  // For example, if esc == 'n', then it returns '\n' (0x0A).
  char escaped(char esc);
  
public:
  Tokenizer(const std::string &filename, const std::string& input);

  Token nextToken();
  bool hasMore() const;
};

} // namespace mbt

#endif
