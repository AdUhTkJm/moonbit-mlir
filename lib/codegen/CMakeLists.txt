set(CODEGEN_SOURCES
  CGModule.cpp
)

add_library(CodegenLib STATIC ${CODEGEN_SOURCES})

target_include_directories(CodegenLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
