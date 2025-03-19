set(LLVM_TARGET_DEFINITIONS MoonOps.td)
mlir_tablegen(MoonOps.h.inc -gen-op-decls)
mlir_tablegen(MoonOps.cpp.inc -gen-op-defs)
mlir_tablegen(MoonTypes.h.inc -gen-typedef-decls -typedefs-dialect=moon)
mlir_tablegen(MoonTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=moon)
mlir_tablegen(MoonDialect.h.inc -gen-dialect-decls -dialect=moon)
mlir_tablegen(MoonDialect.cpp.inc -gen-dialect-defs -dialect=moon)
add_public_tablegen_target(MLIRMoonOpsIncGen)
add_dependencies(mlir-headers MLIRMoonOpsIncGen)

add_mlir_doc(MoonDialect MoonDialect Moon/ -gen-dialect-doc)

add_mlir_dialect_library(MLIRMoon
  MoonDialect.cpp
  MoonOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/dialect
  LINK_LIBS PUBLIC
)
