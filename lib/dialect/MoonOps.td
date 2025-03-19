#ifndef MOON_OPS_TD
#define MOON_OPS_TD

include "MoonDialect.td"
include "MoonTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class MoonOp<string mnemonic, list<Trait> traits = []> :
  Op<MoonDialect, mnemonic, traits>;

def GetUnitOp : MoonOp<"get_unit"> {
  let summary = "Creates a unit-type literal.";

  let arguments = (ins);
  let results = (outs UnitType:$unit);
}

#endif
