//
// Created by yangyw on 8/4/24.
//
#include "Dialect/FunctionUtils.h"
#include "Dialect/RemoteMem.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::cira;

static constexpr llvm::StringRef kRemoteAccess = "remote_access_%s";
static int kRemoteAccessNum = 0;
static constexpr llvm::StringRef kInstrProfInc = "llvm.instrprof.increment";
static constexpr llvm::StringRef kInstrProfIncStep = "llvm.instrprof.increment.step";
static constexpr llvm::StringRef kDumpProfile = "__llvm_profile_write_file";

std::string getNextRemoteAccessName() {
    char buffer[50];  // Adjust size as needed
    std::snprintf(buffer, sizeof(buffer), kRemoteAccess, kRemoteAccessNum++);
    return std::string(buffer);
}
LLVM::LLVMFuncOp mlir::cira::lookupOrCreatRemoteAccess(ModuleOp moduleOp) {
    auto ctx = moduleOp.getContext();

    return cira::lookupOrCreateFn(
        moduleOp, getNextRemoteAccessName(),
        {getVoidPtrType(ctx), getIntBitType(ctx, 64), getIntBitType(ctx, 32), getIntBitType(ctx, 32)}, {});
}

LLVM::LLVMFuncOp mlir::cira::lookupOrCreateInstrInc(ModuleOp moduleOp) {
    auto ctx = moduleOp.getContext();
    return cira::lookupOrCreateFn(
        moduleOp, kInstrProfInc,
        {getVoidPtrType(ctx), getIntBitType(ctx, 64), getIntBitType(ctx, 32), getIntBitType(ctx, 32)}, {});
}

LLVM::LLVMFuncOp mlir::cira::lookupOrCreateInstrIncStep(ModuleOp moduleOp) {
    auto ctx = moduleOp.getContext();
    return cira::lookupOrCreateFn(
        moduleOp, kInstrProfIncStep,
        {getVoidPtrType(ctx), getIntBitType(ctx, 64), getIntBitType(ctx, 32), getIntBitType(ctx, 32)}, {});
}

LLVM::LLVMFuncOp mlir::cira::lookupOrCreateProfileWriteFn(ModuleOp moduleOp) {
    auto ctx = moduleOp.getContext();
    return cira::lookupOrCreateFn(moduleOp, kDumpProfile, {}, {getIntBitType(ctx, 32)});
}
LLVM::LLVMFuncOp mlir::cira::lookupOrCreateFn(ModuleOp moduleOp, StringRef name, ArrayRef<Type> inputTypes,
                                              Type resultType) {
    auto func = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (func)
        return func;
    OpBuilder b(moduleOp.getBodyRegion());
    return b.create<LLVM::LLVMFuncOp>(
        moduleOp->getLoc(), name,
        LLVM::LLVMFunctionType::get(resultType ? resultType : getVoidType(moduleOp.getContext()), inputTypes));
}
Operation::result_range mlir::cira::createLLVMCall(OpBuilder &builder, Location loc, LLVM::LLVMFuncOp fn,
                                                   ValueRange inputs) {

    return builder.create<LLVM::CallOp>(loc, fn, inputs)->getResults();
}
