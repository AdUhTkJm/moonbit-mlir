#include "Parser.h"
#include "ASTNode.h"
#include "lib/utils/Diagnostics.h"

using namespace mbt;

bool Parser::checkEOF() {
  return (unsigned) place >= tokens.size();
}

Token Parser::consume() {
  if (checkEOF())
    return tokens.back();
  return tokens[place++];
}

Token Parser::expect(Token::Type ty) {
  if (checkEOF())
    return tokens.back();
  auto tok = tokens[place];
  if (tok.ty != ty) {
    Diagnostics::error(tok.begin, tok.end,
      std::format("expected {}, but got {}\n", stringifyToken(ty), stringifyToken(tok)));
  }
  place++;
  return tok;
}

Token Parser::peek() {
  if (checkEOF())
    return tokens.back();
  return tokens[place];
}

Token Parser::last() {
  if (checkEOF())
    return tokens.back();
  return tokens[place - 1];
}

bool Parser::peek(Token::Type ty) {
  return peek().ty == ty;
}

bool Parser::test(Token::Type ty) {
  if (peek(ty))
    return ++place;
  return false;
}

mbt::Type *Parser::parseType() {
  if (test(Token::Int))
    return new IntType();

  if (test(Token::Bool))
    return new BoolType();

  if (test(Token::Unit))
    return new UnitType();

  if (test(Token::String))
    return new StringType();

  if (test(Token::Ident))
    return new UnresolvedType(last().vs);
  
  consume();
  Diagnostics::error(peek().begin, peek().end,
    std::format("expected type, but got {}", stringifyToken(peek())));
  return nullptr;
}

ASTNode *Parser::primary() {
  if (peek(Token::IntLit)) {
    auto tok = consume();
    return new IntLiteralNode(tok.vi, tok.begin, tok.end);
  }
  
  if (test(Token::Ident)) {
    Location begin = last().begin;
    auto ident = last().vs;
    return new VarNode(ident, begin, last().end);
  }

  if (test(Token::LPar)) {
    // TODO: tuples, unit literal
    auto x = expr();
    expect(Token::RPar);
    return x;
  }

  consume();
  Diagnostics::error(peek().begin, peek().end,
    std::format("unexpected token: {}", stringifyToken(peek())));
  return nullptr;
}

ASTNode *Parser::memAccessExpr() {
  auto begin = peek().begin;
  auto x = primary();

  while (test(Token::Dot)) {
    auto ident = expect(Token::Ident).vs;
    x = new MemAccessNode(x, ident, begin, last().end);
  }

  return x;
}

ASTNode *Parser::callExpr() {
  auto begin = peek().begin;
  auto x = memAccessExpr();
  
  // A function call.
  while (test(Token::LPar)) {
    std::vector<ASTNode*> args;
    while (!test(Token::RPar)) {
      args.push_back(expr());
      if (test(Token::RPar))
        break;
      
      if (!test(Token::Comma) && !peek(Token::RPar))
        Diagnostics::error(peek().begin, peek().end, "expected ','");
    }
    x = new FnCallNode(x, args, begin, last().end);
  }

  return x;
}

ASTNode *Parser::ifExpr() {
  if (test(Token::If)) {
    Location begin = last().begin;
    ASTNode *cond = expr();
    ASTNode *ifso = blockStmt();
    if (test(Token::Else)) {
      ASTNode *ifnot = blockStmt();
      return new IfNode(cond, ifso, ifnot, begin, last().end);
    }
    return new IfNode(cond, ifso, begin, last().end);
  }

  return callExpr();
}

ASTNode *Parser::compareExpr() {
  Location begin = peek().begin;
  auto x = ifExpr();
  if (peek(Token::Le) || peek(Token::Ge) || peek(Token::Gt)
   || peek(Token::Lt) || peek(Token::Ne) || peek(Token::Eq)) {
    auto ty = consume().ty;
    auto next = ifExpr();
    switch (ty) {
    case Token::Lt:
      x = new BinaryNode(BinaryNode::Lt, x, next, begin, last().end); break;
    case Token::Gt:
      x = new BinaryNode(BinaryNode::Gt, x, next, begin, last().end); break;
    case Token::Eq:
      x = new BinaryNode(BinaryNode::Eq, x, next, begin, last().end); break;
    case Token::Ne:
      x = new BinaryNode(BinaryNode::Ne, x, next, begin, last().end); break;
    // We always assume (x >= y) <=> (y < x), even for user-defined operators.
    case Token::Le:
      x = new BinaryNode(BinaryNode::Gt, next, x, begin, last().end); break;
    case Token::Ge:
      x = new BinaryNode(BinaryNode::Lt, next, x, begin, last().end); break;
    default:
      assert(false);
    }
  }
  return x;
}

ASTNode *Parser::mulExpr() {
  Location begin = peek().begin;
  auto x = compareExpr();
  while (peek(Token::Mul) || peek(Token::Div) || peek(Token::Mod)) {
    auto ty = consume().ty;
    BinaryNode::Type op;
    switch (ty) {
    case Token::Mul:
      op = BinaryNode::Mul; break;
    case Token::Div:
      op = BinaryNode::Div; break;
    case Token::Mod:
      op = BinaryNode::Mod; break;
    default:
      assert(false);
    }
    auto next = compareExpr();
    x = new BinaryNode(op, x, next, begin, last().end);
  }
  return x;
}

ASTNode *Parser::addExpr() {
  Location begin = peek().begin;
  auto x = mulExpr();
  while (peek(Token::Plus) || peek(Token::Minus)) {
    auto ty = consume().ty;
    auto op = ty == Token::Plus ? BinaryNode::Add : BinaryNode::Sub;
    auto nextmul = mulExpr();
    x = new BinaryNode(op, x, nextmul, begin, last().end);
  }
  return x;
}

ASTNode *Parser::expr() {
  return addExpr();
}

ASTNode *Parser::blockStmt() {
  expect(Token::LBrace);
  auto block = new BlockNode(last().begin, {});
  while (!test(Token::RBrace) && !test(Token::End))
    block->body.push_back(stmt());
  
  block->end = last().end;
  return block;
}

ASTNode *Parser::assignStmt(ASTNode *lhs) {
  static std::map<Token::Type, BinaryNode::Type> mapping = {
    { Token::PlusEq, BinaryNode::Add },
    { Token::MinusEq, BinaryNode::Sub },
    { Token::MulEq, BinaryNode::Mul },
    { Token::DivEq, BinaryNode::Div },
    { Token::ModEq, BinaryNode::Mod },
  };

  auto op = last();
  if (auto var = dyn_cast<VarNode>(lhs)) {
    auto n = expr();
    ASTNode *rhs = op.ty == Token::Assign
      ? n
      : new BinaryNode(mapping[op.ty], var, n, lhs->begin, last().end);
    auto node = new AssignNode(var, rhs, lhs->begin, last().end);

    // Optional semicolon.
    test(Token::Semicolon);
    return node;
  }

  Diagnostics::error(lhs->begin, lhs->end,
    "left-hand side of '=' must be a variable");
  return nullptr;
}

ASTNode *Parser::stmt() {
  if (test(Token::Let)) {
    Location begin = last().begin;
    // let (a, b, c) (: Type)? = Expr;
    if (test(Token::LPar))
      assert(false && "NYI");

    bool mut = test(Token::Mut);
    
    // let (mut)? x (: Type)? = Expr;
    auto name = expect(Token::Ident).vs;

    // if there's no name, `expect` would have reported an error
    if (name.length() != 0 && isupper(name[0]))
      Diagnostics::error(last().begin, last().end,
        "variable must start with a lower-case letter");

    Type *ty = nullptr;
    if (test(Token::Colon))
      ty = parseType();
    expect(Token::Assign);
    auto init = expr();
    
    // Optional semicolon.
    test(Token::Semicolon);
    
    auto decl = new VarDeclNode(name, init, mut, begin, last().end);
    decl->type = ty;
    return decl;
  }

  if (peek(Token::LBrace))
    return blockStmt();

  auto x = expr();
  if (test(Token::Assign)
   || test(Token::PlusEq) || test(Token::MinusEq) || test(Token::MulEq) || test(Token::DivEq)
   || test(Token::ModEq))
    return assignStmt(x);

  // Optional semicolon.
  test(Token::Semicolon);
  return x;
}

ASTNode *Parser::topFn() {
  Location begin = last().begin; // 'fn'
  auto ident = expect(Token::Ident).vs;
  std::string name;
  std::optional<std::string> belongsTo;

  // `ident` is actually the name of a struct.
  if (test(Token::ColonColon)) {
    belongsTo = ident;
    name = expect(Token::Ident).vs;
  } else {
    name = ident;
  }

  std::vector<VarDeclNode*> params;
  std::vector<Type*> paramTy;
  Type *fnTy = nullptr;
  
  // A normal function.
  if (name != "main" && name != "init") {
    expect(Token::LPar);
    while (!test(Token::RPar)) {
      auto begin = peek().begin;
      auto paramName = expect(Token::Ident).vs;
      // Type annotation is required.
      expect(Token::Colon);
      Type *ty = parseType();
      paramTy.push_back(ty);
      
      auto decl = new VarDeclNode(paramName, nullptr, begin, last().end);
      decl->type = ty;
      params.push_back(decl);
      
      if (!test(Token::Comma) && !peek(Token::RPar))
        Diagnostics::error(peek().begin, peek().end, "expected ','");
    }

    // Return value is also required.
    expect(Token::Arrow);
    Type *retTy = parseType();
    fnTy = new FunctionType(paramTy, retTy);
  } else {
    // Types of "main" and "init" are already known.
    fnTy = new FunctionType({}, new UnitType());
  }

  mbt::ASTNode *body = nullptr;

  // This is a builtin function:
  //    fn println_mono(x : String) -> Unit = "%println"
  if (test(Token::Assign)) {
    auto builtin = expect(Token::StrLit).vs;
    body = new IntrinsicNode(builtin, last().begin, last().end);
  } else {
    body = blockStmt();
  }

  // Optional ';'
  test(Token::Semicolon);

  auto fn = belongsTo
    ? new FnDeclNode(*belongsTo, name, params, body, begin, body->end)
    : new FnDeclNode(name, params, body, begin, body->end);
  fn->type = fnTy;
  return fn;
}

ASTNode *Parser::topStruct() {
  Location begin = last().begin; // 'struct'
  auto name = expect(Token::Ident).vs;
  expect(Token::LBrace);

  std::vector<std::pair<std::string, Type*>> fields;
  while (!test(Token::RBrace)) {
    auto field = expect(Token::Ident).vs;
    expect(Token::Colon);
    Type *ty = parseType();
    fields.push_back(std::make_pair(field, ty));
  }

  // Optional ';'
  test(Token::Semicolon);
  return new StructDeclNode(name, fields, begin, last().end);
}

ASTNode *Parser::toplevel() {
  if (test(Token::Fn))
    return topFn();

  if (test(Token::Struct))
    return topStruct();

  consume();
  Diagnostics::error(last().begin, last().end,
    std::format("unknown top-level character: {}", stringifyToken(last()))
  );
  return nullptr;
}

ASTNode *Parser::parse() {
  // The range is from the first token to the last non-EOF token.
  auto theModule = new BlockNode(peek().begin, (tokens.end() - 2)->end);
  while (!test(Token::End))
    theModule->body.push_back(toplevel());
  
  return theModule;
}
