#include "MoonPasses.h"
#include "lib/dialect/MoonDialect.h"
#include "lib/dialect/MoonOps.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mbt;

struct MIRToLLVMTypeConverter : LLVMTypeConverter {
  using LLVMTypeConverter::LLVMTypeConverter;

  MIRToLLVMTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx) {
    // Add custom conversions if needed
    addConversion([](mir::PointerType type) {
      // Convert to LLVM type
      return LLVM::LLVMPointerType::get(type.getContext());
    });
  }
};

#define CONVERT_REWRITER(Name, Ty) \
  struct Lower##Name : public OpConversionPattern<Ty> {                      \
    using OpConversionPattern<Ty>::OpConversionPattern;                          \
                                                                             \
    LogicalResult matchAndRewrite(Ty op, OpAdaptor adaptor,                  \
                                  ConversionPatternRewriter &rewriter) const override;  \
  }

CONVERT_REWRITER(Alloca, mir::AllocaOp);
CONVERT_REWRITER(Load, mir::LoadOp);
CONVERT_REWRITER(Store, mir::StoreOp);

LogicalResult LowerAlloca::matchAndRewrite(mir::AllocaOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto llvmType = getTypeConverter()->convertType(op.getType());
  if (!llvmType)
    return failure();

  auto loc = op.getLoc();
  Value one = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

  rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(op, LLVM::LLVMPointerType::get(op->getContext()), llvmType, one);
  return success();
}

LogicalResult LowerLoad::matchAndRewrite(mir::LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  auto resultType = getTypeConverter()->convertType(op.getResult().getType());
  rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultType, adaptor.getBase());
  return success();
}

LogicalResult LowerStore::matchAndRewrite(mir::StoreOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(), adaptor.getBase());
  return success();
}

class MIRToLLVMPass : public PassWrapper<MIRToLLVMPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override;

  llvm::StringRef getName() const override { return "remove-unit"; }
};

void MIRToLLVMPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  auto ctx = theModule.getContext();
  MIRToLLVMTypeConverter converter(ctx);

  // First remove all units in instructions that use it.
  RewritePatternSet lowerPatterns(ctx);
  lowerPatterns.add<
    LowerAlloca,
    LowerLoad,
    LowerStore
  >(converter, ctx);

  ConversionTarget target(*ctx);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<mir::MoonDialect>();

  if (failed(mlir::applyPartialConversion(theModule, target, std::move(lowerPatterns))))
    signalPassFailure();
}

// Register the pass
std::unique_ptr<Pass> mbt::createMIRToLLVMPass() {
  return std::make_unique<MIRToLLVMPass>();
}
