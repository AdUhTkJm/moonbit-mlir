#ifndef CGMODULE_H
#define CGMODULE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "lib/parse/ASTNode.h"
#include <map>

namespace mbt {

// Code Generation Module.
class CGModule {
  mlir::MLIRContext &ctx;
  mlir::OpBuilder builder;

  mlir::ModuleOp theModule;

  // Obtain a fused location from ASTNode
  mlir::Location getLoc(ASTNode *node);

  std::map<std::string, mlir::Value> symbolTable;

  // Type cache:
  mlir::Type unitType;
  mlir::Type boolType;

  // Symbol table manager with RAII.
  struct SemanticScope {
    decltype(symbolTable) oldTable;
    CGModule &cgm;
  public:
    SemanticScope(CGModule &cgm);
    ~SemanticScope();
  };
public:
  CGModule(mlir::MLIRContext &ctx);

  void emitModule(ASTNode *node);

  void emitGlobalFn(FnDeclNode *globalFn);
  mlir::Value emitStmt(ASTNode *node);
  mlir::Value emitExpr(ASTNode *node);
  mlir::Value emitIfExpr(IfNode *ifexpr);
  mlir::Value emitBinaryExpr(BinaryNode *binary);

  void emitFunctionPrologue(mlir::func::FuncOp op, FnDeclNode *fn);

  void dump();

  mlir::Location getLoc(Location loc);
  mlir::StringAttr getStringAttr(llvm::StringRef str);
  mlir::Type getTy(mbt::Type *ty);
};

};

#endif
