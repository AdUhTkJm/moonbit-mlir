#ifndef LLVM_LOWERING_H
#define LLVM_LOWERING_H

#include "llvm/IR/LLVMContext.h"
#include "mlir/IR/BuiltinOps.h"

namespace mbt {

std::unique_ptr<llvm::Module> translateToLLVM(llvm::LLVMContext &ctx, mlir::ModuleOp op);

} // namespace mbt

#endif
