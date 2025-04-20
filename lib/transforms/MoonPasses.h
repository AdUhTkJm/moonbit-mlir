#ifndef MOON_PASSES_H
#define MOON_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mbt {

void registerMoonPasses(mlir::MLIRContext *ctx, mlir::ModuleOp theModule, bool dump);

// Removes all instances of values of mir::UnitType.
std::unique_ptr<mlir::Pass> createRemoveUnitPass();

// Resolves indirect calls to function pointers, if they are fixed at compile type.
std::unique_ptr<mlir::Pass> createFPtrResolutionPass();

// Lowers mbt::IntrinsicOp.
std::unique_ptr<mlir::Pass> createLowerIntrinsicPass();

// Lowers all MoonIR-specific operations.
std::unique_ptr<mlir::Pass> createMIRToLLVMPass();

} // namespace mbt

#endif
