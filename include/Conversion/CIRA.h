#ifndef REMOTE_TARGET_TO_REMOTE_MEM_H
#define REMOTE_TARGET_TO_REMOTE_MEM_H

#include <memory>
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
class Pass;
class Value;
class Location;
class RewritePatternSet;
class ConversionPatternRewriter;

#define GEN_PASS_DECL_CIRA
#include "Conversion/Passes.h.inc"

namespace cira {
class RemoteMemDialect;
class RemoteMemTypeConverter : public TypeConverter {
public:
    using TypeConverter::convertType;
    RemoteMemTypeConverter(MLIRContext *ctx, DictionaryAttr &rule);

    FunctionType convertFunctionSignature(FunctionType funcTy, SignatureConversion &result);
    LLVM::LLVMFunctionType convertLLVMFunctionSignature(LLVM::LLVMFunctionType funcTy, bool isVariadic, SignatureConversion &result);

    // Specialized conversion routine for function inputs
    Type convertCallingConventionType(Type type, bool needLLVMComp = false);

    // Specialized conversion routine for function results
    Type convertFunctionResult(Type type, bool needLLVMComp = false);
    MLIRContext &getContext();
    cira::RemoteMemDialect *getDialect() { return rmemDialect; };

    LogicalResult funcArgTypeConverter(Type type, SmallVectorImpl<Type> &result, bool needLLVMComp = false);

protected:
    cira::RemoteMemDialect *rmemDialect;
    DictionaryAttr &rule;

private:
    // Routine that recursively convert llvm.ptr to rmref
    llvm::Optional<Type> convertLLVMPointerType(LLVM::LLVMPointerType type);

    // Convert struct type
    llvm::Optional<LogicalResult> convertLLVMStructType(LLVM::LLVMStructType type, SmallVectorImpl<Type> &results, ArrayRef<Type> callStack);

    // Convert LLVM array
    llvm::Optional<Type> convertLLVMArrayType(LLVM::LLVMArrayType type);

    // Convert FunctionType
    llvm::Optional<Type> convertFunctionType(FunctionType type);

    // Convert LLVM FunctionType
    llvm::Optional<Type> convertLLVMFunctionType(LLVM::LLVMFunctionType type);

    // Convert MemRefType (recursive convert contained elementType if has nested llvm.ptr |struct | memref)
    llvm::Optional<Type> convertMemRefType(MemRefType type);

};

void populateSCFCIRAPatterns(MLIRContext *ctx, RewritePatternSet &patterns);
void populateCIRAPatterns(MLIRContext *ctx, RewritePatternSet &patterns);
std::unique_ptr<Pass> createCIRAPass();
}
}

#endif