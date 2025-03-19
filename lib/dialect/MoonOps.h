#ifndef MOON_OPS_H
#define MOON_OPS_H

#include "lib/dialect/MoonDialect.h"
#include "lib/dialect/MoonTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "lib/dialect/MoonOps.h.inc"

#endif
