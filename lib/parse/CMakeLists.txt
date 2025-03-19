set(PARSE_SOURCES
  lexer.cpp
  parser.cpp
)

add_library(ParseLib STATIC ${PARSE_SOURCES})

target_include_directories(ParseLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
