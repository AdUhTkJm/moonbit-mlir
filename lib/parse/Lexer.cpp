#include "Lexer.h"
#include <format>
#include <map>
#include <cmath>

using namespace mbt;

std::map<std::string, Token::Type> keywords = {
  { "if", Token::If },
  { "else", Token::Else },
  { "let", Token::Let },
  { "while", Token::While },
  { "for", Token::For },
  { "fn", Token::Fn },
  { "return", Token::Return },
  { "struct", Token::Struct },
  
  // Internal types.
  { "Int", Token::Int },
  { "Bool", Token::Bool },
  { "FixedArray", Token::FixedArray },
  { "Unit", Token::Unit },
  { "String", Token::String },
};

Tokenizer::Tokenizer(const std::string &filename, const std::string& input):
  input(input), filename(filename), loc(0), last_line(0), line(1) {}

Token::Token(Type t, Location begin, int len):
  ty(t), begin(begin), end(begin) {
  end.col = begin.col + len;
}

Token::Token(Type t, std::string name, Location begin):
  ty(t), vs(name), begin(begin), end(begin) {
  end.col = begin.col + name.length();
}

Token::Token(Type t, int vi, Location begin, int len):
  ty(t), vi(vi), begin(begin), end(begin) {
  end.col = begin.col + len;
}

char Tokenizer::escaped(char esc) {
  switch (esc) {
  case 'n': return '\n';
  case 't': return '\t';
  case '"': return '"';
  case '\\': return '\\';
  case 'r': return '\r';
  case 'b': return '\b';
  case 'f': return '\f';
  case 'v': return '\v';
  default:
    Diagnostics::error(
      Location { filename, line, loc - 1 - last_line },
      Location { filename, line, loc - last_line + 1 },
      std::format("invalid escape sequence '\\{}'", esc)
    );
    return esc;
  }
}

Token Tokenizer::nextToken() {
  assert(loc < input.size());

  // Skip whitespace
  while (loc < input.size() && std::isspace(input[loc])) {
    if (input[loc] == '\n')
      last_line = loc, line++;
    loc++;
  }

  // Hit end of input because of skipping whitespace
  if (loc >= input.size()) {
    return Token(Token::End, Location { filename, line, input.size() - last_line }, 0);
  }

  char c = input[loc];
  Location curr_loc = Location { filename, line, loc - last_line };

  // Identifiers and keywords
  if (std::isalpha(c) || c == '_') {
    std::string name;
    while (loc < input.size() && (std::isalnum(input[loc]) || input[loc] == '_')) {
      name += input[loc++];
    }
    if (keywords.contains(name))
      return Token(keywords[name], curr_loc, name.length());
    return Token(Token::Ident, name, curr_loc);
  }

  // Integer literals
  if (std::isdigit(c)) {
    int value = 0;
    int last_loc = loc;
    while (loc < input.size() && std::isdigit(input[loc])) {
      // Overflow.
      if (value > 214748364 || (value == 214748364 && input[loc] >= '8')) {
        auto from_loc = Location { filename, line, last_loc - last_line };
        Diagnostics::error(from_loc, curr_loc, std::format("overflowing literal"));
      }
      value = value * 10 + (input[loc++] - '0');
    }
    return Token(Token::IntLit, value, curr_loc, loc - last_loc);
  }

  // String literal
  if (c == '"') {
    size_t start_loc = loc++;
    Location str_start = curr_loc;
    std::string value;
    bool closed = false;

    while (loc < input.size()) {
      if (input[loc] == '\\') {
        if (++loc >= input.size()) {
          Diagnostics::error(
            Location { filename, line, start_loc - last_line },
            Location { filename, line, loc - last_line },
            "escape sequence not terminated"
          );
          value += '\\';
          break;
        }
        value.push_back(escaped(input[loc++]));
        continue;
      }
      
      if (input[loc] == '"') {
        closed = true;
        loc++;
        break;
      }

      value.push_back(input[loc++]);
    }

    if (!closed)
      Diagnostics::error(
        str_start,
        Location { filename, line, loc - last_line },
        "string literal not closed"
      );

    return Token(Token::StrLit, value, str_start);
  }

  // Check for multi-character operators like >=, <=, ==, !=, +=, etc.
  if (loc + 1 < input.size()) {
    switch (c) {
    case '=': 
      if (input[loc + 1] == '=') { loc += 2; return Token(Token::Eq, curr_loc, 2); }
      break;
    case '>':
      if (input[loc + 1] == '=') { loc += 2; return Token(Token::Ge, curr_loc, 2); }
      break;
    case '<': 
      if (input[loc + 1] == '=') { loc += 2; return Token(Token::Le, curr_loc, 2); }
      break;
    case '!': 
      if (input[loc + 1] == '=') { loc += 2; return Token(Token::Ne, curr_loc, 2); }
      break;
    case '+': 
      if (input[loc + 1] == '=') { loc += 2; return Token(Token::PlusEq, curr_loc, 2); }
      break;
    case '-': 
      if (input[loc + 1] == '=') { loc += 2; return Token(Token::MinusEq, curr_loc, 2); }
      if (input[loc + 1] == '>') { loc += 2; return Token(Token::Arrow, curr_loc, 2); }
      break;
    case '*': 
      if (input[loc + 1] == '=') { loc += 2; return Token(Token::MulEq, curr_loc, 2); }
      break;
    case '/': 
      if (input[loc + 1] == '=') { loc += 2; return Token(Token::DivEq, curr_loc, 2); }
      if (input[loc + 1] == '/') { 
        // Loop till we find a line break, then retries to find the next Token
        // (we can't continue working in the same function frame)
        for (; loc < input.size(); loc++) {
          if (input[loc] == '\n')
            return nextToken();
        }
      }
      break;
    case '&': 
      if (input[loc + 1] == '=') { loc += 2; return Token(Token::BitAndEq, curr_loc, 2); }
      if (input[loc + 1] == '&') { loc += 2; return Token(Token::And, curr_loc, 2); }
      break;
    case '|': 
      if (input[loc + 1] == '=') { loc += 2; return Token(Token::BitOrEq, curr_loc, 2); }
      if (input[loc + 1] == '|') { loc += 2; return Token(Token::Or, curr_loc, 2); }
      break;
    case '^': 
      if (input[loc + 1] == '=') { loc += 2; return Token(Token::XorEq, curr_loc, 2); }
      break;
    case ':': 
      if (input[loc + 1] == ':') { loc += 2; return Token(Token::ColonColon, curr_loc, 2); }
      break;
    default: break;
    }
  }

  // Single-character operators and symbols
  switch (c) {
  case '+': loc++; return Token(Token::Plus, curr_loc, 1);
  case '-': loc++; return Token(Token::Minus, curr_loc, 1);
  case '*': loc++; return Token(Token::Mul, curr_loc, 1);
  case '/': loc++; return Token(Token::Div, curr_loc, 1);
  case '%': loc++; return Token(Token::Div, curr_loc, 1);
  case '&': loc++; return Token(Token::BitAnd, curr_loc, 1);
  case '|': loc++; return Token(Token::BitOr, curr_loc, 1);
  case '^': loc++; return Token(Token::Xor, curr_loc, 1);
  case ';': loc++; return Token(Token::Semicolon, curr_loc, 1);
  case ':': loc++; return Token(Token::Colon, curr_loc, 1);
  case '=': loc++; return Token(Token::Assign, curr_loc, 1);
  case '!': loc++; return Token(Token::Exclaim, curr_loc, 1);
  case '(': loc++; return Token(Token::LPar, curr_loc, 1);
  case ')': loc++; return Token(Token::RPar, curr_loc, 1);
  case '[': loc++; return Token(Token::LBrak, curr_loc, 1);
  case ']': loc++; return Token(Token::RBrak, curr_loc, 1);
  case '<': loc++; return Token(Token::Lt, curr_loc, 1);
  case '>': loc++; return Token(Token::Gt, curr_loc, 1);
  case ',': loc++; return Token(Token::Comma, curr_loc, 1);
  case '{': loc++; return Token(Token::LBrace, curr_loc, 1);
  case '}': loc++; return Token(Token::RBrace, curr_loc, 1);
  case '.': loc++; return Token(Token::Dot, curr_loc, 1);
  default:
    Diagnostics::error(curr_loc, curr_loc, std::format("unexpected character: {}", c));
    loc++; return nextToken();
  }
}

bool Tokenizer::hasMore() const {
  return loc < input.size();
}

const char *mbt::stringifyToken(Token t) {
  return stringifyToken(t.ty);
}

const char *mbt::stringifyToken(Token::Type t) {
  static std::map<Token::Type, const char*> literals = {
    { Token::And, "'&&'" },
    { Token::Arrow, "'->'" },
    { Token::Assign, "'='" },
    { Token::BitAnd, "'&'" },
    { Token::BitAndEq, "'&='" },
    { Token::BitOr, "'|'" },
    { Token::BitOrEq, "'|='" },
    { Token::Bool, "builtin-type 'Bool'" },
    { Token::Colon, "':'" },
    { Token::ColonColon, "'::'" },
    { Token::Comma, "','" },
    { Token::Div, "'/'" },
    { Token::DivEq, "'/='" },
    { Token::Dot, "'.'" },
    { Token::Else, "keyword 'else'" },
    { Token::End, "EOF" },
    { Token::Eq, "'=='" },
    { Token::Exclaim, "'!'" },
    { Token::FixedArray, "builtin-type 'FixedArray'"},
    { Token::Fn, "keyword 'fn'" },
    { Token::For, "keyword 'for'" },
    { Token::Ge, "'>='" },
    { Token::Gt, "'>'" },
    { Token::Ident, "identifier" },
    { Token::If, "keyword 'if'" },
    { Token::Int, "builtin-type 'Int'" },
    { Token::IntLit, "int literal" },
    { Token::LBrace, "'{'" },
    { Token::LBrak, "'['" },
    { Token::Le, "'<='" },
    { Token::Let, "keyword 'let'" },
    { Token::LPar, "'('" },
    { Token::Lt, "'<'" },
    { Token::Minus, "'-'" },
    { Token::MinusEq, "'-='" },
    { Token::Mod, "'%'" },
    { Token::Mul, "'*'" },
    { Token::MulEq, "'*='" },
    { Token::Ne, "'!='" },
    { Token::Or, "'||'" },
    { Token::Plus, "'+'" },
    { Token::PlusEq, "'+='" },
    { Token::RBrace, "'}'" },
    { Token::RBrak, "']'" },
    { Token::Return, "keyword 'return'" },
    { Token::RPar, "')'" },
    { Token::Semicolon, "';'" },
    { Token::String, "builtin-type 'string'" },
    { Token::StrLit, "string literal" },
    { Token::Struct, "keyword 'struct'" },
    { Token::Unit, "builtin-type 'unit'" },
    { Token::While, "keyword 'while'" },
    { Token::Xor, "'^'" },
    { Token::XorEq, "'^='" },
  };
  assert(literals.contains(t));
  return literals[t];
}
