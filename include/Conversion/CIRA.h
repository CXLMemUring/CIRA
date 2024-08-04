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
class RemoteMemTypeConverter;

void populateSCFCIRAPatterns(MLIRContext *ctx, RewritePatternSet &patterns);
void populateCIRAPatterns(MLIRContext *ctx, RewritePatternSet &patterns);
std::unique_ptr<Pass> createCIRAPass();
}
}

#endif