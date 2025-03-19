#include "MoonDialect.h"
#include "MoonOps.h"
#include "MoonTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lib/dialect/MoonDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/dialect/MoonTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/dialect/MoonOps.cpp.inc"

namespace mir {

void MoonDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/dialect/MoonTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/dialect/MoonOps.cpp.inc"
      >();
}

} // namespace mir
