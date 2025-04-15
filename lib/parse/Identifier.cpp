#include "Identifier.h"

mbt::Identifier::Identifier(llvm::StringRef str):
  package("builtin"), record(""), name(str) { }

mbt::Identifier::Identifier(std::string str):
  mbt::Identifier((llvm::StringRef) str) { }

mbt::Identifier::operator std::string() const {
  return std::format("@{}.{}::{}", package, record == "" ? "<null>" : record, name);
}
