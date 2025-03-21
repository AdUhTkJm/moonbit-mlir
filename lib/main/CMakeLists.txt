set(LLVM_LINK_COMPONENTS Support Support nativecodegen OrcJIT)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

add_executable(moonc main.cpp)

llvm_update_compile_flags(moonc)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(
  moonc
  PRIVATE ${dialect_libs}
          ${conversion_libs}
          ${extension_libs}
          MLIRAnalysis
          MLIRIR
          MLIRExecutionEngine
          MLIRParser
          MLIRPass
          MLIRMemRefDialect
          MLIRFunctionInterfaces
          MLIRSideEffectInterfaces
          MLIRCastInterfaces
          LibParse
          LibSema
          LibCG
          LibTransforms
          LibUtils)

mlir_check_link_libraries(moonc)
