set(TBLGEN_SOURCES
  Diagnostics.cpp
)

add_library(UtilsLib STATIC ${TBLGEN_SOURCES})

target_include_directories(UtilsLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
