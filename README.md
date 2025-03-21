## Moonbit Compiler

This is a compiler for Moonbit language.

[The official compiler](https://github.com/moonbitlang/moonbit-compiler) uses OCaml and hand-crafted IR. This compiler attempts to leverage the power of MLIR and LLVM, enabling more optimizations.

As Moonbit language itself is unstable, this compiler might not always follow up the most recent changes.

Any contributions are very welcome.

### Compiler Digest

Unlike [Clang IR](https://github.com/llvm/clangir), we don't use a single dialect for every possible operations. Instead, we take advantage of various MLIR dialects that are already available, and treat `moon` dialect as a supplement.

As an example, take a look at the following source code.

```mbt
fn main {
  let x = 1 + 2 * 3
}
```

It will be compiled to:

```mlir
module {
  func.func @main() -> !moon.unit {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = arith.muli %c2_i32, %c3_i32 : i32
    %1 = arith.addi %c1_i32, %0 : i32
    %2 = "moon.get_unit"() : () -> !moon.unit
    return %2 : !moon.unit
  }
}
```

Note that the last statement is automatically the return value, hence the `get_unit` operation. These useless instructions are planned to be eliminated via further transforms.
