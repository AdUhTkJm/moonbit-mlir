#include "MoonPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"

using namespace mbt;
using namespace mlir;

void mbt::registerMoonPasses(MLIRContext *ctx, ModuleOp theModule) {
  PassManager pm(ctx);

  pm.addPass(mbt::createRemoveUnitPass());

  // Lowers intrinsic calls.
  pm.addPass(mbt::createLowerIntrinsicPass());

  // This fptr resolution creates some dead values.
  // Eliminate them by DCE pass.
  pm.addPass(mbt::createFPtrResolutionPass());
  pm.addPass(mlir::createRemoveDeadValuesPass());

  // Convert MoonIR to LLVM IR.
  pm.addPass(mbt::createMIRToLLVMPass());

  // Convert MLIR builtin dialects to LLVM IR.
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());

  pm.enableVerifier();
  
  if (pm.run(theModule).failed())
    llvm_unreachable("pass failure");
}
