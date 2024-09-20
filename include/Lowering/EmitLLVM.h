//
// Created by yangyw on 8/4/24.
//

#ifndef CIRA_EMITLLVM_H
#define CIRA_EMITLLVM_H
#include <memory>
#include "Dialect/RemoteMem.h"

namespace mlir {
#define GEN_PASS_DECL_EMITLLVM
#include "Lowering/Passes.h.inc"
class Pass;
class RewritePatternSet;
class LLVMTypeConverter;

void populateEmitLLVMPatterns (LLVMTypeConverter &llvmTypeConverter, RewritePatternSet &patterns);
std::unique_ptr<Pass> createEmitLLVMPass();
}
#endif // CIRA_EMITLLVM_H
