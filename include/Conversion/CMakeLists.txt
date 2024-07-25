set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name DisaggregationConversion)
add_public_tablegen_target(RemoteMemConversionPassIncGen)