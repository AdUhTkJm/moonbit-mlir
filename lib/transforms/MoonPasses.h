#ifndef MOON_PASSES_H
#define MOON_PASSES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace mbt {

void registerMoonPasses(mlir::MLIRContext *ctx, mlir::ModuleOp theModule);

} // namespace mbt

#endif
