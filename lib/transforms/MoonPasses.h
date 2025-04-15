#ifndef MOON_PASSES_H
#define MOON_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mbt {

void registerMoonPasses(mlir::MLIRContext *ctx, mlir::ModuleOp theModule);

// Removes all instances of values of mir::UnitType.
std::unique_ptr<mlir::Pass> createRemoveUnitPass();

// Resolves indirect calls to function pointers, if they are fixed at compile type.
std::unique_ptr<mlir::Pass> createFPtrResolutionPass();

} // namespace mbt

#endif
