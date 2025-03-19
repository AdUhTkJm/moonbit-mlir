set(DIALECT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mydialect)
set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS} ${DIALECT_DIR})

# Add your dialect files
add_mlir_dialect(MyDialect ${DIALECT_DIR})

# Ensure TableGen files are generated
mlir_tablegen(MyDialectDialectIncGen ${DIALECT_DIR}/MyDialectOps.td)
mlir_tablegen(MyDialectTypesIncGen ${DIALECT_DIR}/MyDialectTypes.td)

# Include generated files
target_include_directories(YourProject
  PUBLIC
  ${DIALECT_DIR}
  ${CMAKE_BINARY_DIR}/include
)

# Add the dialect to your project
add_mlir_dialect(MyDialect ${DIALECT_DIR})

# Register operations and types in the dialect
add_mlir_library(YourDialect
  ${CMAKE_CURRENT_SOURCE_DIR}/MyDialect.cpp
)
