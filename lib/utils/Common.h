#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <map>
#include <vector>
#include <tuple>
#include <cassert>
#include <iostream>
#include <format>
#include "llvm/ADT/StringRef.h"

namespace mbt {

struct Location {
  llvm::StringRef filename;
  size_t line;
  size_t col;
};

};

#endif
