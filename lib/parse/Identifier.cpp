#include "Identifier.h"

mbt::Identifier::Identifier(llvm::StringRef name):
  name(name), mangled(mangleImpl()) { }

mbt::Identifier::Identifier(std::string name):
  mbt::Identifier((llvm::StringRef) name) { }

mbt::Identifier::Identifier(llvm::StringRef package, llvm::StringRef record, llvm::StringRef name):
  package(package), record(record), name(name), mangled(mangleImpl()) { }


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
