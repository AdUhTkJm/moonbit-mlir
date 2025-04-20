#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "LLVMLowering.h"

using namespace mbt;

std::unique_ptr<llvm::Module> mbt::translateToLLVM(llvm::LLVMContext &ctx, mlir::ModuleOp op) {
  mlir::registerBuiltinDialectTranslation(*op.getContext());
  mlir::registerLLVMDialectTranslation(*op.getContext());

  return mlir::translateModuleToLLVMIR(op, ctx);
}
