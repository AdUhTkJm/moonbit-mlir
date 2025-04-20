#ifndef SEMA_H
#define SEMA_H

#include "lib/parse/ASTNode.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "TypeContext.h"

namespace mbt {

class TypeInferrer {
  TypeContext ctx;

  llvm::StringMap<Type*> typeMap;
  llvm::StringSet<> mutables;
  llvm::StringMap<Type*> structs;
  std::map<std::vector<std::string>, Type*> structFields;

  int id = 0;

  // Create a brand-new WeakType instance.
  Type *fresh();

  // Find the real type of this WeakType.
  Type *repr(Type *weakType);

  // Returns true for success.
  bool unify(Type *&t1, Type *&t2);

  Type *inferFn(FnDeclNode *fn);
  Type *inferIf(IfNode *ifexpr);
  Type *inferVarDecl(VarDeclNode *decl);
  Type *inferBinary(BinaryNode *binary);
  Type *inferCall(FnCallNode *call);
  Type *inferAssign(AssignNode *assign);
  Type *inferWhile(WhileNode *whileLoop);

  struct SemanticScope {
    decltype(typeMap) oldMap;
    TypeInferrer &inferrer;
  public:
    SemanticScope(TypeInferrer &inferrer);
    ~SemanticScope();
  };

  // This also updates the `type` field in ASTNode.
  Type *infer(ASTNode *node);

public:
  void process(ASTNode *node);

  // Tidies up all weak types, making them referring to the concrete type.
  void tidy(ASTNode *node);
};

} // namespace mbt

#endif
