set(TBLGEN_SOURCES
  Diagnostics.cpp
  CLIParser.cpp
)

add_library(LibUtils STATIC ${TBLGEN_SOURCES})

target_include_directories(LibUtils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
