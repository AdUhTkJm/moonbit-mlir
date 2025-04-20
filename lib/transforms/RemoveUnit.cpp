#include "MoonPasses.h"
#include "lib/dialect/MoonOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mbt;

#define REMOVE_UNIT_REWRITER(Name, Ty) \
  struct RemoveUnit##Name : public OpRewritePattern<Ty> {                    \
    using OpRewritePattern::OpRewritePattern;                                \
                                                                             \
    LogicalResult matchAndRewrite(Ty op,                                     \
                                  PatternRewriter &rewriter) const override; \
  }

REMOVE_UNIT_REWRITER(Intrinsic, mir::IntrinsicOp);
REMOVE_UNIT_REWRITER(Return, func::ReturnOp);
REMOVE_UNIT_REWRITER(Func, func::FuncOp);
REMOVE_UNIT_REWRITER(CallIndirect, func::CallIndirectOp);
REMOVE_UNIT_REWRITER(FPtr, mir::ClosureOp);
REMOVE_UNIT_REWRITER(GetUnit, mir::GetUnitOp);
REMOVE_UNIT_REWRITER(ScfYield, scf::YieldOp);
REMOVE_UNIT_REWRITER(ScfIf, scf::IfOp);

// ------------------------------------------------------------
// Unit Consumers
// ------------------------------------------------------------

LogicalResult RemoveUnitReturn::matchAndRewrite(func::ReturnOp op, PatternRewriter &rewriter) const {
  if (op.getNumOperands() < 1 || !isa<mir::UnitType>(op.getOperand(0).getType()))
    return failure();

  rewriter.replaceOpWithNewOp<func::ReturnOp>(op /*create an empty return op*/);
  return success();
}

LogicalResult RemoveUnitScfYield::matchAndRewrite(scf::YieldOp op, PatternRewriter &rewriter) const {
  if (op.getNumOperands() < 1 || !isa<mir::UnitType>(op.getOperand(0).getType()))
    return failure();

  rewriter.replaceOpWithNewOp<scf::YieldOp>(op /*create an empty yield op*/);
  return success();
}

LogicalResult RemoveUnitFunc::matchAndRewrite(func::FuncOp op, PatternRewriter &rewriter) const {
  if (op.getNumResults() < 1)
    return failure();
  
  // Remove `unit` return type in functions
  llvm::SmallVector<mlir::Type> results;
  auto retTy = op.getFunctionType().getResult(0);
  if (!isa<mir::UnitType>(retTy))
    results.push_back(retTy);
  
  // Remove `unit` parameters from functions
  llvm::SmallVector<mlir::Type> argTypes;
  for (auto &arg : op.getArguments()) {
    if (!isa<mir::UnitType>(arg.getType()))
      argTypes.push_back(arg.getType());
  }

  // Nothing has changed. Just return.
  if (argTypes.size() == op.getNumArguments() && results.size() == op.getNumResults())
    return failure();
    
  op.setType(mlir::FunctionType::get(rewriter.getContext(), argTypes, results));
  
  return success();
}

LogicalResult RemoveUnitFPtr::matchAndRewrite(mir::ClosureOp op, PatternRewriter &rewriter) const {
  StringAttr functionName = op.getFunction().getRootReference();
  Operation *lookedup = op->getParentOfType<ModuleOp>().lookupSymbol(op.getFunction());
  auto funcOp = cast<func::FuncOp>(lookedup);

  // The referenced function hasn't changed. Everything alright.
  if (funcOp.getFunctionType() == op.getResult().getType())
    return failure();
  
  // Update.
  auto symref = SymbolRefAttr::get(rewriter.getContext(), functionName);
  rewriter.replaceOpWithNewOp<mir::ClosureOp>(op, funcOp.getFunctionType(), symref);
  return success();
}

// ------------------------------------------------------------
// Unit Producers
// ------------------------------------------------------------

LogicalResult RemoveUnitIntrinsic::matchAndRewrite(mir::IntrinsicOp op, PatternRewriter &rewriter) const {
  if (op.getNumResults() < 1 || !isa<mir::UnitType>(op.getResult().getType()))
    return failure();
  
  llvm::SmallVector<mlir::Type, 1> resultTypes;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);
  rewriter.create<mir::IntrinsicOp>(op.getLoc(), resultTypes, op.getIntrinsicName(), op.getArgs());
  op.erase();

  return success();
}

LogicalResult RemoveUnitCallIndirect::matchAndRewrite(func::CallIndirectOp op, PatternRewriter &rewriter) const {
  if (op.getNumResults() < 1 || !isa<mir::UnitType>(op.getResult(0).getType()))
    return failure();
  
  llvm::SmallVector<mlir::Type, 1> resultTypes;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);
  rewriter.create<func::CallIndirectOp>(op.getLoc(), resultTypes, op.getOperands());
  op.erase();

  return success();
}

LogicalResult RemoveUnitGetUnit::matchAndRewrite(mir::GetUnitOp op, PatternRewriter &rewriter) const {
  op.erase();
  return success();
}

LogicalResult RemoveUnitScfIf::matchAndRewrite(scf::IfOp op, PatternRewriter &rewriter) const {
  if (op->getNumResults() < 1 || !isa<mir::UnitType>(op.getResult(0).getType()))
    return failure();

  bool withElse = !op.getElseRegion().empty();
  auto emptyIf = rewriter.create<scf::IfOp>(op.getLoc(), /*cond=*/op.getCondition(), withElse);
  
  // Move all blocks in both regions into the new IfOp.
  rewriter.inlineRegionBefore(op.getThenRegion(), emptyIf.getThenRegion(), emptyIf.getThenRegion().begin());
  if (withElse)
    rewriter.inlineRegionBefore(op.getElseRegion(), emptyIf.getElseRegion(), emptyIf.getElseRegion().begin());
  
  op.erase();
  return success();
}

class RemoveUnitPass : public PassWrapper<RemoveUnitPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override;

  llvm::StringRef getName() const override { return "remove-unit"; }
};

void RemoveUnitPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  auto ctx = theModule.getContext();

  // First remove all units in instructions that use it.
  RewritePatternSet unitConsumers(ctx);
  unitConsumers.add<
    RemoveUnitFunc,
    RemoveUnitFPtr,
    RemoveUnitReturn,
    RemoveUnitScfYield
  >(ctx);

  (void) mlir::applyPatternsGreedily(theModule, std::move(unitConsumers));

  // Now it's safe to erase instructions that produce a unit.
  RewritePatternSet unitProducers(ctx);
  unitProducers.add<
    RemoveUnitIntrinsic,
    RemoveUnitCallIndirect,
    RemoveUnitGetUnit,
    RemoveUnitScfIf
  >(ctx);

  (void) mlir::applyPatternsGreedily(theModule, std::move(unitProducers));
}

// Register the pass
std::unique_ptr<Pass> mbt::createRemoveUnitPass() {
  return std::make_unique<RemoveUnitPass>();
}
