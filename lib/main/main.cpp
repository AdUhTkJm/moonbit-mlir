#include "lib/parse/Parser.h"
#include "lib/utils/Diagnostics.h"
#include "lib/codegen/CGModule.h"
#include "lib/sema/Sema.h"
#include "lib/transforms/MoonPasses.h"
#include "lib/transforms/LLVMLowering.h"
#include "llvm/IR/Module.h"
#include <fstream>
#include <sstream>

using namespace mbt;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: moonc <file>\n";
    return 1;
  }

  std::ifstream ifs(argv[1]);
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
  // node->dump();

  Diagnostics::reportAll();

  mlir::MLIRContext ctx;
  CGModule cgm(ctx);
  mlir::ModuleOp theModule = cgm.emitModule(node);
  cgm.dump();

  registerMoonPasses(&ctx, theModule, /*dump=*/false);

  llvm::LLVMContext llvmCtx;
  auto llvmModule = mbt::translateToLLVM(llvmCtx, theModule);
  llvm::errs() << "------- Lowered -------\n";
  llvmModule->print(llvm::errs(), nullptr);
  
  return 0;
}
