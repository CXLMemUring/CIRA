//
// Created by yangyw on 8/4/24.
//

#ifndef CIRA_REMOTEMEMTOLLVM_H
#define CIRA_REMOTEMEMTOLLVM_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

void populateRemoteMemToLLVMPatterns(mlir::RewritePatternSet &patterns);
#endif // CIRA_REMOTEMEMTOLLVM_H
