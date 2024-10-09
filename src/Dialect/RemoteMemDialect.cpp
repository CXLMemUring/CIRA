//
// Created by yangyw on 8/5/24.
//
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "Dialect/RemoteMem.h"
#include "Dialect/FunctionUtils.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace mlir;
using namespace mlir::cira;

//bool RemoteMemRefType::isValidElementType(Type elementType) {
//    if (!elementType) return false;
//    if (!elementType.isa<mlir::MemRefType, mlir::LLVM::LLVMPointerType, mlir::UnrankedMemRefType>()) return false;
//    return true;
//}
//LogicalResult RemoteMemRefType::verify(function_ref<InFlightDiagnostic()> emitError, Type elementType, unsigned canRemote) {
//    if (!RemoteMemRefType::isValidElementType(elementType))
//        return emitError() << "invalid pointer elementType: " << elementType;
//    return success();
//}
::mlir::Attribute RemoteMemDialect::parseAttribute(mlir::DialectAsmParser&, mlir::Type) const{}
void RemoteMemDialect::printAttribute(mlir::Attribute, mlir::DialectAsmPrinter&) const{}
void RemoteMemDialect::initialize() {
    registerTypes();
    addOperations<
#define GET_OP_LIST
#include "Dialect/RemoteMemRef.cpp.inc"
        >();
}
void RemoteMemDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/RemoteMem.cpp.inc"
        >();
}

#define GET_TYPEDEF_CLASSES
#include "Dialect/RemoteMemRef.cpp.inc"