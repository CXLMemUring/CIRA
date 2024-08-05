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

#include "Dialect/RemoteMem.cpp.inc"

void RemoteMemDialect::initialize() {
    registerTypes();
    addOperations<
#define GET_OP_LIST
#include "Dialect/RemoteMemRef.cpp.inc"
        >();
}

#define GET_TYPEDEF_CLASSES
#include "Dialect/RemoteMemRef.cpp.inc"