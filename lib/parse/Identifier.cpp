#include "Identifier.h"

mbt::Identifier::Identifier(llvm::StringRef str):
  name(str), mangled(mangleImpl()) { }

mbt::Identifier::Identifier(std::string str):
  mbt::Identifier((llvm::StringRef) str) { }

std::string mbt::Identifier::mangleImpl() const {
  // Do not mangle main().
  if (name == "main")
    return "main";

  std::string packageStr = package.size() ? std::format("{}{}", package.size(), package) : "";
  std::string recordStr = record.size() ? std::format("{}{}N", record.size(), record) : "";

  // Note: arguments are not included.
  // Support overloading in future.
  return std::format("_Z{}{}{}{}",
    packageStr, recordStr, name.size(), name);
}
