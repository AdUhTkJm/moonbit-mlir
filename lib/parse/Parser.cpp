#include "Parser.h"

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
  
  if (peek(Token::Ident)) {
    auto tok = consume();
    return new VarNode(tok.vs, tok.begin, tok.end);
  }

  consume();
  Diagnostics::error(peek().begin, peek().end,
    std::format("unexpected token: {}", stringifyToken(peek())));
  return nullptr;
}

ASTNode *Parser::callExpr() {
  auto begin = peek().begin;
  auto x = primary();
  
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
  auto x = ifExpr();
  if (peek(Token::Le) || peek(Token::Ge) || peek(Token::Gt)
   || peek(Token::Lt) || peek(Token::Ne) || peek(Token::Eq)) {
    Location begin = peek().begin;
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
  auto x = compareExpr();
  while (peek(Token::Mul) || peek(Token::Div) || peek(Token::Mod)) {
    Location begin = peek().begin;
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
  auto x = mulExpr();
  while (peek(Token::Plus) || peek(Token::Minus)) {
    Location begin = peek().begin;
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

ASTNode *Parser::stmt() {
  if (test(Token::Let)) {
    Location begin = last().begin;
    // let (a, b, c) (: Type)? = Expr;
    if (test(Token::LPar))
      assert(false && "NYI");
    
    // let x (: Type)? = Expr;
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
    
    auto decl = new VarDeclNode(name, init, begin, last().end);
    decl->type = ty;
    return decl;
  }

  if (peek(Token::LBrace))
    return blockStmt();

  auto x = expr();
  // Optional semicolon.
  test(Token::Semicolon);
  return x;
}

ASTNode *Parser::topFn() {
  Location begin = last().begin; // 'fn'
  auto name = expect(Token::Ident).vs;
  
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

  auto fn = new FnDeclNode(name, params, body, begin, body->end);
  fn->type = fnTy;
  return fn;
}

ASTNode *Parser::toplevel() {
  if (test(Token::Fn))
    return topFn();

  return nullptr;
}

ASTNode *Parser::parse() {
  // The range is from the first token to the last non-EOF token.
  auto theModule = new BlockNode(peek().begin, (tokens.end() - 2)->end);
  while (!test(Token::End))
    theModule->body.push_back(toplevel());
  
  return theModule;
}
