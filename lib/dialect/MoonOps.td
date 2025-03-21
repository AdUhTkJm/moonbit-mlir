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

  let assemblyFormat = "attr-dict";
}

def GlobalOp : MoonOp<"global"> {
  let summary = "Defines a global variable.";

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<AnyAttr>:$value,
                       UnitAttr:$constant);
  let results = (outs);
  
  let assemblyFormat = "$sym_name `:` $type (`=` $value^)? attr-dict";
}

def AddressofOp : MoonOp<"addressof"> {
  let summary = "Get address of a global variable.";

  let arguments = (ins SymbolRefAttr:$global_name);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$global_name `:` type($result) attr-dict";
}

def FetchGlobalOp : MoonOp<"fetch_global"> {
  let summary = "Loads a value from a global variable.";
  
  let arguments = (ins SymbolRefAttr:$global_name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$global_name `:` type($result) attr-dict";
}

def FPtrOp : MoonOp<"fptr"> {
  let summary = "Fetches a function pointer based on the global.";
  
  let arguments = (ins SymbolRefAttr:$function);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = "$function `:` type($result) attr-dict";
}

def IntrinsicOp : MoonOp<"intrinsic"> {
  let summary = "A Moonbit intrinsic.";
  
  let arguments = (ins StrAttr:$intrinsic_name,
                       Variadic<AnyType>:$args);
  let results = (outs Optional<AnyType>:$result);
  
  let assemblyFormat = [{
    $intrinsic_name $args `:` functional-type($args, $result) attr-dict
  }];
}

#endif
