#ifndef SEMA_H
#define SEMA_H

#include "lib/parse/ASTNode.h"

namespace mbt {

class TypeInferrer {
  std::map<std::string, Type*> typeMap;
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

  struct SemanticScope {
    decltype(typeMap) oldMap;
    TypeInferrer &inferrer;
  public:
    SemanticScope(TypeInferrer &inferrer);
    ~SemanticScope();
  };
public:
  
  // This also updates the `type` field in ASTNode.
  Type *infer(ASTNode *node);
};

} // namespace mbt

#endif
