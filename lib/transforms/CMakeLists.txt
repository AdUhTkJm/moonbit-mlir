set(TRANSFORMS_SOURCES
  MoonPasses.cpp
)

add_library(LibTransforms STATIC ${TRANSFORMS_SOURCES})

target_include_directories(LibTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(LibTransforms
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
