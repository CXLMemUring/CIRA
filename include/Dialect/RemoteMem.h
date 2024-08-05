//
// Created by yangyw on 8/4/24.
//

#ifndef CIRA_REMOTEMEM_H
#define CIRA_REMOTEMEM_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class Value;
class Type;
namespace cira {
class RemoteMemDialect;
class RemoteMemRefType;
} // namespace cira
} // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/RemoteMem.h.inc"
#define GET_TYPEDEF_CLASSES
#include "Dialect/RemoteMemRef.h.inc"

#endif // CIRA_REMOTEMEM_H
