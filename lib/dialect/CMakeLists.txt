set(DIALECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS} ${DIALECT_DIR})

# Add your dialect files
add_mlir_dialect(MoonIR ${DIALECT_DIR})

# Ensure TableGen files are generated
mlir_tablegen(MoonIRDialectGen ${DIALECT_DIR}/MoonOps.td)
mlir_tablegen(MoonIRTypesGen ${DIALECT_DIR}/MoonTypes.td)

# Include generated files
target_include_directories(moonc
  PUBLIC
  ${DIALECT_DIR}
  ${CMAKE_BINARY_DIR}/include
)

# Register operations and types in the dialect
add_mlir_library(MoonIR
  ${CMAKE_CURRENT_SOURCE_DIR}/moonir.cpp
)
