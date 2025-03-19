#ifndef MOON_TYPES_TD
#define MOON_TYPES_TD

include "MoonDialect.td"
include "mlir/IR/AttrTypeBase.td"

class MoonType<string name, string typeMnemonic> : TypeDef<MoonDialect, name> {
  let mnemonic = typeMnemonic;
}

def UnitType : MoonType<"Unit", "unit"> {
  let summary = "unit type in Moonbit";

  let description = [{
    A zero-size type that contains no information.
  }];
}

#endif
