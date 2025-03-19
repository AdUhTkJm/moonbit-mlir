#ifndef MOON_DIALECT_TD
#define MOON_DIALECT_TD

include "mlir/IR/OpBase.td"

def MoonDialect : Dialect {
  let name = "moon";
  let summary = "Moonbit-specific MLIR dialect";
  let cppNamespace = "::mir";

  let useDefaultTypePrinterParser = 1;
  
  // Enable when we have attributes.
  // let useDefaultAttributePrinterParser = 1;
}

#endif
