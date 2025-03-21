set(TRANSFORMS_SOURCES
  MoonPasses.cpp
  RemoveUnit.cpp
  FPtrResolution.cpp
)

add_library(LibTransforms STATIC ${TRANSFORMS_SOURCES})

target_include_directories(LibTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(LibTransforms PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRIR
  MLIRDialect
  MLIRMoon
  MLIRPass
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRSCFToControlFlow
  MLIRTransforms
)
