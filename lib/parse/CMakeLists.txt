set(PARSE_SOURCES
  Lexer.cpp
  Parser.cpp
  ASTNode.cpp
)

add_library(LibParse STATIC ${PARSE_SOURCES})

target_include_directories(LibParse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
