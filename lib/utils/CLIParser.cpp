#include "CLIParser.h"
#include "llvm/Support/CommandLine.h"

using namespace mbt;
using namespace llvm;

cl::opt<std::string> outputFile("o", cl::desc("specify output file"), cl::value_desc("filename"));
cl::list<std::string> inputFiles(cl::Positional, cl::desc("<input file>"), cl::Required);

cl::opt<bool> dumpAST("dump-ast", cl::desc("dump AST"));
cl::opt<bool> dumpIR("dump-ir", cl::desc("dump MLIR after each pass"));
cl::alias dumpIRAlias("di", cl::aliasopt(dumpIR));

CLIOptions mbt::parseCLIArgument(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  return CLIOptions {
    .dumpIR = dumpIR,
    .dumpAST = dumpAST,
    .outputFile = outputFile,
    .inputFiles = inputFiles,
  };
}
