set(SEMA_SOURCES
  Sema.cpp
  Types.cpp
)

add_library(LibSema STATIC ${SEMA_SOURCES})

target_include_directories(LibSema PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
