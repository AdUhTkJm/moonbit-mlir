set(CODEGEN_SOURCES
  CGModule.cpp
)

add_library(LibCG STATIC ${CODEGEN_SOURCES})

target_include_directories(LibCG PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(LibCG
  MLIRIR
  MLIRDialect
  MLIRMoon
)
