#ifndef MOON_DIALECT_TD
#define MOON_DIALECT_TD

include "mlir/IR/OpBase.td"

def MoonDialect : Dialect {
  let name = "moon";
  let summary = "Moonbit-specific MLIR dialect";
  let cppNamespace = "::mir";
}

#endif
