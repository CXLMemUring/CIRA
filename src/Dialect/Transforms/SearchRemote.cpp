//
// Created by yangyw on 8/5/24.
//
#include "Dialect/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "Dialect/RemoteMem.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/TypeName.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <set>
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/TypeSize.h"

namespace mlir {
#define GEN_PASS_DEF_RMEMSEARCHREMOTE
#include "Dialect/Transforms/Passes.h.inc"
}

using namespace mlir;

namespace {

static inline std::pair<bool, Type> hasTargetTypeImpl(Type t, SetVector<Type> &callStack, DictionaryAttr &rule) {
    if (callStack.contains(t))
        return {false, t};
    callStack.insert(t);
    auto stackGurard = llvm::make_scope_exit([&] { callStack.pop_back(); });
    auto hasTargetTypeHandler = [&](Type type) {
        return hasTargetTypeImpl(type, callStack, rule);
    };
    return llvm::TypeSwitch<Type, std::pair<bool, Type>>(t)
        .Case<LLVM::LLVMStructType>([&](LLVM::LLVMStructType ST) -> std::pair<bool, Type> {
            if (ST.isIdentified()) {
                if (auto find = rule.get(ST.getName()))
                    return { ST.getName() == "struct.arc", find.cast<TypeAttr>().getValue()};
                // TODO: nested struct
            }
            return {false, ST};
        })
        .Case<LLVM::LLVMArrayType>([&](LLVM::LLVMArrayType aryType) -> std::pair<bool, Type> {
            auto ET = hasTargetTypeHandler(aryType.getElementType());
            if (ET.first) {
                llvm::errs() << "Cannot handle array of rmem target type\n";
            }
            return { false, LLVM::LLVMArrayType::get(ET.second, aryType.getNumElements()) };
        })
        .Case<LLVM::LLVMPointerType>([&](LLVM::LLVMPointerType PT) -> std::pair<bool, Type> {
            if (PT.isOpaque())
                return { false, PT };
            auto ET = hasTargetTypeHandler(PT.getElementType());
            auto LET = LLVM::LLVMPointerType::get(ET.second, PT.getAddressSpace());
            if (ET.first)
                return { false, cira::RemoteMemRefType::get(LET, 1) };
            return { false, LET };
        })
        .Default([](Type t) -> std::pair<bool, Type> { return {false, t} ; });
}

static inline Type hasTarget(Type t, DictionaryAttr &rule) {
    SetVector<Type> callStack;
    return hasTargetTypeImpl(t, callStack, rule).second;
}

class RMEMSearchRemotePass : public impl::RMEMSearchRemoteBase<RMEMSearchRemotePass> {
    void runOnOperation() override {
        ModuleOp mop = getOperation();
        OpBuilder b(mop);

        // SmallVector<NamedAttribute, 4> mappings;
        // mappings.emplace_back(b.getStringAttr("struct.node"), TypeAttr::get(b.getI1Type()));
        // mop->setAttr("rmem.type_rule", b.getDictionaryAttr(mappings));

        DictionaryAttr rule = mop->getAttrOfType<DictionaryAttr>("rmem.type_rule");
        // rule.dump();

        // Set attr of all users of current op
        // users should check operand and decide how to convert
        // this is different from "remote_target" that op does not have to
        // convert
        auto notifyRemoteUsers([&](Operation *op){
            for (Value rel : op->getResults()) {
                if (rel.use_empty()) continue;
                for (Operation *user : rel.getUsers()) {
                    user->setAttr("remote_check_use", b.getI8IntegerAttr(1));
                }
            }
        });

        // if result contains invalid type, must convert current op
        // also set rel_types for current op, but op will decide whether to follow this
        auto checkResultTypes([&](Operation *op, bool &isMarked, SmallVectorImpl<Type> &retTypes){
            for (Type r : op->getResultTypes()) {
                Type ifTargeted = hasTarget(r, rule);
                retTypes.emplace_back(ifTargeted);
                if (r != ifTargeted) {
                    // mark remote target
                    op->setAttr("remote_target", b.getI8IntegerAttr(1));
                    isMarked = true;
                }
            }
        });

        auto checkOperandTypes([&](Operation *op, bool &isMarked){
            for (Type t : op->getOperandTypes()) {
                if (hasTarget(t, rule) != t) {
                    // mark remote target
                    op->setAttr("remote_target", b.getI8IntegerAttr(1));
                    isMarked = true;
                    break;
                }
            }
        });

        mop->walk([&](Operation *op) {

            bool isMarked = false;
            SmallVector<Type, 4> retTypes;

            checkResultTypes(op, isMarked, retTypes);

            // prevent hand-written "rel_types" conflict
            if (isMarked && (op->getAttr("rel_types") == nullptr))
                op->setAttr("rel_types", b.getTypeArrayAttr(retTypes));
            else
                checkOperandTypes(op, isMarked);

            // notify users
            // This mechanism makes sure that implicit result type changes will
            // be acknowledged by its users e.g.
            // %1 = gep ptr<struct<(i32, i32)>>[0, 1] -> ptr<i32>
            // load/store %1 ...
            if (isMarked)
                notifyRemoteUsers(op);

            return WalkResult::advance();
        });
    }
};
}

std::unique_ptr<Pass> mlir::createRemoteMemSearchRemotePass() {
    return std::make_unique<RMEMSearchRemotePass>();
}
