#ifndef IDENTIFIER_H
#define IDENTIFIER_H

#include "llvm/ADT/StringRef.h"
#include <format>

namespace mbt {

// As an example, the identifier `@builtin.Array::new` obeys:
//   - package = "builtin";
//   - record = "Array";
//   - name = "new".
//
// For non-static variables and functions, and other identifiers, `record` is "".
class Identifier {
  std::string package;
  std::string record;
  std::string name;

  // A cache for the name.
  std::string mangled;
  std::string mangleImpl() const;
public:
  Identifier() {}
  Identifier(llvm::StringRef name);
  Identifier(std::string name);
  Identifier(llvm::StringRef package, llvm::StringRef record, llvm::StringRef name);
  
  llvm::StringRef getName() const { return name; }
  llvm::StringRef getRecordName() const { return record; }
  llvm::StringRef getPackageName() const { return package; }

  std::string mangle() const { return mangled; }
};

} // namespace mbt

template <>
class std::formatter<mbt::Identifier> {
public:
  constexpr auto parse(std::format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const mbt::Identifier& ident, FormatContext& ctx) const {
    return std::format_to(ctx.out(), "{}", ident.mangle());
  }
};

#endif
