add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Lowering)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  DisaggMLIRUtils

  MLIRArithDialect
  MLIROptLib
  MLIRDialect
  MLIRIR
  # MLIRRemoteMem
)

add_llvm_executable(cira_remote driver.cpp)

llvm_update_compile_flags(cira_remote)
target_link_libraries(cira_remote PRIVATE ${LIBS})
