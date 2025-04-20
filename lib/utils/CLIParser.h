#ifndef CLI_PARSER_H
#define CLI_PARSER_H

#include <string>
#include <vector>

namespace mbt {

struct CLIOptions {
  using option = unsigned char;

  struct {
    // Dumps MLIR after every pass.
    option dumpIR : 1;
    option dumpAST: 1;
  };

  std::string outputFile;
  std::vector<std::string> inputFiles;
};

CLIOptions parseCLIArgument(int argc, char **argv);

}

#endif
