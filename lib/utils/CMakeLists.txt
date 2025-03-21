set(TBLGEN_SOURCES
  Diagnostics.cpp
)

add_library(LibUtils STATIC ${TBLGEN_SOURCES})

target_include_directories(LibUtils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
