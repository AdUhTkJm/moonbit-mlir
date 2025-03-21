#ifndef MOON_PASSES_H
#define MOON_PASSES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mbt {

void registerMoonPasses(mlir::MLIRContext *ctx, mlir::ModuleOp theModule);

std::unique_ptr<mlir::Pass> createRemoveUnitPass();

} // namespace mbt

#endif
