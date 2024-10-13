#include "Dialect/FunctionUtils.h"
#include "Dialect/RemoteMem.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/IR/DataLayout.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTREMOTEMEMTOLLVM
#include "Lowering/Passes.h.inc"
using namespace mlir::cira;
namespace {
// =================================================================
// Patterns

class RemoteMemFuncLowering : public RemoteMemOpLoweringPattern<cira::OffloadOp> {
    using RemoteMemOpLoweringPattern<cira::OffloadOp>::RemoteMemOpLoweringPattern;
    LogicalResult matchAndRewrite(cira::OffloadOp op, cira::OffloadOpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        Value argSize = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
        Value retSize = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);

        if (adaptor.getInputs().size()) {

            auto ofldArgBuf = getOrCreateOffloadArgBuf(op->getParentOfType<ModuleOp>());
            Value ptrArg = rewriter.create<LLVM::AddressOfOp>(loc, ofldArgBuf);
            Value argBuf = rewriter.create<LLVM::LoadOp>(loc, ptrArg);

            // awayls point to the last byte being stored
            Value curMem = argBuf;

            // move cursor and store to offload arg buf
            for (auto const &[old, adp] : llvm::zip(op.getInputs(), adaptor.getInputs())) {
                Type currentType = adp.getType();
                Value toStore = adp;
                if (old.getType().isa<LLVM::LLVMPointerType>()) {
                    currentType = adp.getType().cast<LLVM::LLVMPointerType>().getElementType();
                    toStore = rewriter.create<LLVM::LoadOp>(loc, adp);
                }

                curMem = rewriter.create<LLVM::BitcastOp>(loc, LLVM::LLVMPointerType::get(currentType), curMem);
                rewriter.create<LLVM::StoreOp>(loc, toStore, curMem);
                curMem = rewriter.create<LLVM::GEPOp>(loc, curMem.getType(), curMem, ArrayRef<LLVM::GEPArg>(1));
            }

            Value startP = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), argBuf);
            Value endP = rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), curMem);
            argSize = rewriter.create<arith::SubIOp>(loc, rewriter.getI64Type(), endP, startP);
        }

        // currently we only permit 0/1 return
        Type relType;
        if (op.getRet().size()) {
            relType = getTypeConverter()->convertType(op.getResult(0).getType());
            retSize = getSizeInBytes(loc, relType, rewriter);
        }

        auto callRoutine = lookupOrCreateCallOffloadService(op->getParentOfType<ModuleOp>());
        Value retPtr = createLLVMCall(rewriter, loc, callRoutine, {adaptor.getFid(), argSize, retSize}).front();

        if (op.getRet().size()) {
            Value castRet = rewriter.create<LLVM::BitcastOp>(loc, LLVM::LLVMPointerType::get(relType), retPtr);
            Value retValue = rewriter.create<LLVM::LoadOp>(loc, castRet);
            rewriter.replaceOp(op, retValue);
        } else {
            rewriter.eraseOp(op);
        }

        return mlir::success();
    }
};

// =================================================================
} // namespace

namespace {
class ConvertRemoteMemToLLVMPass : public impl::ConvertRemoteMemToLLVMBase<ConvertRemoteMemToLLVMPass> {
public:
    ConvertRemoteMemToLLVMPass() = default;
    void runOnOperation() override {
        ModuleOp m = getOperation();

        // get local templates

        // get local caches
//        std::string cfgPath = cacheCFG;
//        std::unordered_map<int, mlir::cira::Cache *> caches;
//        mlir::cira::readCachesFromFile(caches, cfgPath);

        RemoteMemTypeLowerer typeConverter(&getContext());
        RewritePatternSet patterns(&getContext());
//        populateRemoteMemToLLVMPatterns(patterns);

        ConversionTarget target(getContext());
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
//        target.addIllegalOp<cira::function>();
        if (failed(applyPartialConversion(m, target, std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

void populateRemoteMemToLLVMPatterns(RewritePatternSet &patterns) {
    patterns.add<RemoteMemFuncLowering>(patterns);
}

std::unique_ptr<Pass> createRemoteMemToLLVMPass() { return std::make_unique<ConvertRemoteMemToLLVMPass>(); }
} // namespace mlir