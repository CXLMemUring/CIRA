#ifndef LOWER_CIRA
#define LOWER_CIRA

include "mlir/Pass/PassBase.td"

def ConvertRemoteMemToLLVM : Pass<"lower-rmem-to-llvm", "ModuleOp"> {
  let summary = "Convert Remote Mem dialect to llvm dialect";
  let constructor = "mlir::createRemoteMemToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect", "rmem::RemoteMemDialect", "func::FuncDialect", "arith::ArithDialect"];
  let options = [
  ];
}

def EmitLLVM : Pass<"emit-llvm", "ModuleOp"> {
  let summary = "Convert everything into llvm-mlir";
  let constructor = "mlir::createEmitLLVMPass()";
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "cira::RemoteMemDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "scf::SCFDialect",
    "vector::VectorDialect",
    "memref::MemRefDialect"
  ];
}

#endif
