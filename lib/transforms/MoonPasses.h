#ifndef MOON_PASSES_H
#define MOON_PASSES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "lib/dialect/MoonOps.h"

namespace mbt {

void registerMoonPasses(mlir::MLIRContext *ctx, mlir::ModuleOp theModule);

// Removes all instances of values of mir::UnitType.
std::unique_ptr<mlir::Pass> createRemoveUnitPass();

// Resolves indirect calls to function pointers, if they are fixed at compile type.
std::unique_ptr<mlir::Pass> createFPtrResolutionPass();

} // namespace mbt

#endif
