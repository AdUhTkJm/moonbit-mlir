#include "MoonPasses.h"
#include "lib/dialect/MoonOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

static const llvm::StringMap<llvm::StringRef> intrMap = {
  { "println", "puts" }
};

struct IntrinsicRewriter : public OpRewritePattern<mir::IntrinsicOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mir::IntrinsicOp op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult IntrinsicRewriter::matchAndRewrite(mir::IntrinsicOp op,
                                                 PatternRewriter &rewriter) const {
  auto intrinsicName = intrMap.at(op.getIntrinsicName());
  if (intrinsicName.starts_with("llvm.")) {
    auto attr = rewriter.getStringAttr(intrinsicName);
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(op, attr, op.getArgs());
    return success();
  }
  rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, op.getResultTypes(), intrinsicName, op.getArgs());
  return success();
}

class LowerIntrinsicPass : public PassWrapper<LowerIntrinsicPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override;

  llvm::StringRef getName() const override { return "lower-intrinsic"; }
};

void LowerIntrinsicPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  auto ctx = theModule.getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<IntrinsicRewriter>(ctx);

  (void) mlir::applyPatternsGreedily(theModule, std::move(patterns));
}

std::unique_ptr<mlir::Pass> mbt::createLowerIntrinsicPass() {
  return std::make_unique<LowerIntrinsicPass>();
}
