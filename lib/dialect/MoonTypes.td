#ifndef MOON_TYPES_TD
#define MOON_TYPES_TD

include "MoonDialect.td"
include "mlir/IR/AttrTypeBase.td"

class MoonType<string name, string typeMnemonic> : TypeDef<MoonDialect, name> {
  let mnemonic = typeMnemonic;
}

def UnitType : MoonType<"Unit", "unit"> {
  let summary = "The `unit` type in Moonbit.";

  let description = [{
    A zero-size type that contains no information.
  }];
}

def PointerType : MoonType<"Pointer", "ptr"> {
  let summary = "A transparent pointer, as opposed to opaque ones in LLVM IR.";

  let parameters = (ins "mlir::Type":$pointee);

  let assemblyFormat = [{
    `<` $pointee `>`
  }];
}

#endif
