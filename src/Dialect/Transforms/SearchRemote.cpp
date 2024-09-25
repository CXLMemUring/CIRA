//
// Created by yangyw on 8/5/24.
//
#include "Dialect/RemoteMem.h"
#include "Dialect/Transforms/Passes.h"
#include "Dialect/WorkloadAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/TypeName.h"
#include "llvm/Support/TypeSize.h"

#include <mlir/IR/BlockAndValueMapping.h>
#include <set>

namespace mlir {
#define GEN_PASS_DEF_RMEMSEARCHREMOTE
#include "Dialect/Transforms/Passes.h.inc"
}

using namespace mlir;

namespace {

class RMEMSearchRemotePass : public impl::RMEMSearchRemoteBase<RMEMSearchRemotePass> {
    std::set<Operation *> addrPathDFS(Operation *op, scf::ForOp loop) {
        std::set<Operation *> search;
        for (OpOperand &opd : op->getOpOperands()) {
            if (opd.get() == loop.getInductionVar())
                search.insert(op);
            else {
                Operation *def = opd.get().getDefiningOp();
                if (def && def->getBlock() == loop.getBody()) {
                    auto dfs = addrPathDFS(def, loop);
                    if (dfs.size()) {
                        dfs.insert(op);
                        search.merge(dfs);
                    }
                }
            }
        }
        return search;
    }

    llvm::SetVector<mlir::Value> analyzeValueUses(mlir::scf::ForOp forOp) {
        llvm::SetVector<mlir::Value> capturedValues;
        mlir::Region& loopBody = forOp.getRegion();

        forOp.walk([&](mlir::Operation *op) {
            for (mlir::Value operand : op->getOperands()) {
                mlir::Operation* definingOp = operand.getDefiningOp();

                // Check if the operand is defined outside the loop
                // and is not a result of an operation within the loop
                if (!loopBody.isAncestor(operand.getParentRegion()) &&
                    (definingOp == nullptr || !loopBody.isAncestor(definingOp->getParentRegion()))) {
                    capturedValues.insert(operand);
                }
            }
        });

        // Remove loop induction variable and loop-defined values from captured values
        capturedValues.remove(forOp.getInductionVar());
        for (mlir::Value arg : forOp.getRegionIterArgs()) {
            capturedValues.remove(arg);
        }

        return capturedValues;
    }

    mlir::func::FuncOp extractLoopBody(mlir::scf::ForOp forOp, const llvm::SetVector<mlir::Value> &capturedValues,
                                       mlir::OpBuilder &builder) {
        mlir::Block *body = forOp.getBody();

        // Prepare function type
        llvm::SmallVector<mlir::Type, 4> argTypes;
        argTypes.push_back(forOp.getInductionVar().getType()); // Induction variable
        for (mlir::Value val : capturedValues) {
            argTypes.push_back(val.getType());
        }
        auto funcType = builder.getFunctionType(argTypes, {});

        auto remoteFunc = builder.create<mlir::func::FuncOp>(forOp.getLoc(), "remote_loop_body", funcType);

        auto remoteBlock = remoteFunc.addEntryBlock();
        builder.setInsertionPointToStart(remoteBlock);

        // Create a mapping from old values to new function arguments
        mlir::BlockAndValueMapping mapping;
        mapping.map(forOp.getInductionVar(), remoteBlock->getArgument(0));
        for (size_t i = 0; i < capturedValues.size(); ++i) {
            mapping.map(capturedValues[i], remoteBlock->getArgument(i + 1));
        }

        // Clone loop body operations, remapping the values
        for (auto &op : body->getOperations()) {
            if (!mlir::isa<mlir::scf::YieldOp>(op)) {
                builder.clone(op, mapping);
            }
        }

        builder.create<mlir::func::ReturnOp>(forOp.getLoc());
        return remoteFunc;
    }

    void replaceWithRemoteCall(mlir::scf::ForOp forOp, mlir::func::FuncOp remoteFunc,
                               const llvm::SetVector<mlir::Value> &capturedValues, mlir::OpBuilder &builder) {
        builder.setInsertionPoint(forOp);

        llvm::SmallVector<mlir::Value, 4> callOperands;
        callOperands.push_back(forOp.getInductionVar());
        callOperands.append(capturedValues.begin(), capturedValues.end());

        auto call = builder.create<mlir::func::CallOp>(forOp.getLoc(), remoteFunc, callOperands);
        //  forOp.erase();
    }
    void runOnOperation() override {
        ModuleOp mop = getOperation();
        mlir::OpBuilder builder(mop.getContext());
        WorkloadComplexityAnalyzer analyzer{};
        auto dfsComplexity = [&](const std::set<Operation *> &dfs) {
            unsigned c = 0;
            for (auto const &op : dfs) {
                c += analyzer.visitOperation(op);
                if (c >= WorkloadComplexityAnalyzer::uncertain)
                    return WorkloadComplexityAnalyzer::uncertain;
            }
            return c;
        };
        mop.walk([&](mlir::scf::ForOp forOp) {
            if (dfsComplexity(addrPathDFS(forOp, forOp)) < 100) {
                llvm::SetVector<mlir::Value> capturedValues = analyzeValueUses(forOp);
                mlir::func::FuncOp remoteFunc = extractLoopBody(forOp, capturedValues, builder);
                replaceWithRemoteCall(forOp, remoteFunc, capturedValues, builder);
            }
        });
    }
};
} // namespace

std::unique_ptr<Pass> mlir::createRemoteMemSearchRemotePass() { return std::make_unique<RMEMSearchRemotePass>(); }
