set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CIRALowering)
add_public_tablegen_target(CIRALoweringPassIncGen)
add_dependencies(mlir-headers CIRALoweringPassIncGen)
