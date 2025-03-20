#include "Types.h"
#include <sstream>

using namespace mbt;

std::string FunctionType::toString() const {
  std::stringstream ss;
  for (auto x : paramTy)
    ss << x->toString() << ", ";
  auto str = ss.str();
  // Remove the extra ", " at the end
  str.pop_back();
  str.pop_back();
  return std::format("{} -> {}", str, retTy->toString());
}

std::string WeakType::toString() const {
  return std::format("'{}", id);
}
