#include "CLIParser.h"
#include "lib/parse/Parser.h"
#include "lib/utils/Diagnostics.h"
#include "lib/codegen/CGModule.h"
#include "lib/sema/Sema.h"
#include "lib/transforms/MoonPasses.h"
#include "lib/transforms/LLVMLowering.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include <fstream>
#include <sstream>

using namespace mbt;

int main(int argc, char **argv) {
  auto options = mbt::parseCLIArgument(argc, argv);

  if (options.inputFiles.size() < 1) {
    std::cerr << "fatal error: no input files";
    return 1;
  }

  if (options.inputFiles.size() > 1) {
    std::cerr << "error: moonc does not support more than 1 input files currently";
    return 1;
  }

  std::ifstream ifs(options.inputFiles[0]);
  if (!ifs) {
    std::cerr << std::format("cannot open file: {}\n", argv[1]);
    return 1;
  }
  
  std::ostringstream ss;
  ss << ifs.rdbuf();
  std::string content = ss.str();

  Diagnostics::setInput(content);

  // Tokenizer.
  std::vector<Token> toks;
  Tokenizer tokenizer(argv[1], content);
  while (tokenizer.hasMore())
    toks.push_back(tokenizer.nextToken());
  if (toks.empty())
    return 0;

  if (toks.back().ty != Token::End) {
    Token last = toks.back();
    last.ty = Token::End; // Copy location information
    toks.push_back(last);
  }

  Diagnostics::reportAll();

  // Parser.
  Parser parser(toks);
  auto node = parser.parse();

  Diagnostics::reportAll();

  // Semantic analysis.
  mbt::TypeInferrer inferrer;
  inferrer.infer(node);
  inferrer.tidy(node);
  if (options.dumpAST)
    node->dump();

  Diagnostics::reportAll();

  mlir::MLIRContext ctx;
  CGModule cgm(ctx);
  mlir::ModuleOp theModule = cgm.emitModule(node);
  if (options.dumpIR)
    cgm.dump();

  registerMoonPasses(&ctx, theModule, /*dump=*/options.dumpIR);

  llvm::LLVMContext llvmCtx;
  auto llvmModule = mbt::translateToLLVM(llvmCtx, theModule);
  
  if (!options.outputFile.size()) {
    llvmModule->print(llvm::errs(), nullptr);
    return 0;
  }

  std::error_code errorCode;
  llvm::raw_fd_ostream dest(options.outputFile, errorCode, llvm::sys::fs::OF_None);
  llvmModule->print(dest, nullptr);
  
  return 0;
}
