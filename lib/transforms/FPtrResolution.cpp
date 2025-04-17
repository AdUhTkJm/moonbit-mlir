#include "MoonPasses.h"
#include "lib/dialect/MoonOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

struct FPtrCallIndirectRewriter : public OpRewritePattern<func::CallIndirectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallIndirectOp op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult FPtrCallIndirectRewriter::matchAndRewrite(func::CallIndirectOp op,
                                                        PatternRewriter &rewriter) const {
  auto function = op.getOperand(0);
  
  // This is a deterministic function pointer.
  // Replace it with an ordinary call op.
  if (auto fptr = dyn_cast<mir::ClosureOp>(function.getDefiningOp())) {
    rewriter.replaceOpWithNewOp<func::CallOp>(op,
      fptr.getFunction(), op.getResultTypes(), op.getArgOperands());
    return success();
  }

  return failure();
}

class FPtrResolutionPass : public PassWrapper<FPtrResolutionPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override;

  llvm::StringRef getName() const override { return "fptr-resolution"; }
};

void FPtrResolutionPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  auto ctx = theModule.getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<FPtrCallIndirectRewriter>(ctx);

  (void) mlir::applyPatternsGreedily(theModule, std::move(patterns));
}

std::unique_ptr<mlir::Pass> mbt::createFPtrResolutionPass() {
  return std::make_unique<FPtrResolutionPass>();
}
