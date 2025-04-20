#ifndef TYPE_CONTEXT_H
#define TYPE_CONTEXT_H

#include <functional>
#include <unordered_set>
#include <utility>
#include <iostream>
#include "Types.h"

namespace mbt {

class TypeContext {
  struct Hash {
    size_t operator()(const Type* t) const {
      int hash = 0xDEADBEEF;
      const_cast<Type*>(t)->walk([&](const Type *t) {
        hash ^= t->getKind();
        hash *= (t->getKind() >> 1) * 2 + 1;

        if (auto weak = dyn_cast<WeakType>(t)) {
          hash ^= weak->id;
          return;
        }
        
        if (auto s = dyn_cast<StructType>(t)) {
          hash ^= std::hash<std::string>()(s->name);
          return;
        }
      });
      return hash;
    }
  };

  struct Eq {
    bool operator()(const Type *a, const Type *b) const {
      return *a == *b;
    }
  };

  std::unordered_set<Type*, Hash, Eq> types;
public:
  template<class T, class... Args>
  T *create(Args... args) {
    auto x = new T(std::forward<Args>(args)...);
    if (auto [it, absent] = types.insert(x); !absent) {
      delete x;
      return cast<T>(*it);
    }
    return x;
  }
};

}

#endif
