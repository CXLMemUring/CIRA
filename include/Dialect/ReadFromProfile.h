//
// Created by yangyw on 9/19/24.
//

#ifndef CIRA_READFROMPROFILE_H
#define CIRA_READFROMPROFILE_H

#include "llvm/ObjectYAML/YAML.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "Dialect/RemoteMem.h"

#include <set>
#include <vector>
#include <tuple>
#include <map>

struct ProfilingResult {
    std::string type;
    float size;
    uint64_t value;
};

template <>
struct llvm::yaml::MappingTraits<ProfilingResult> {
    static void mapping(IO &io, ProfilingResult &info) {
        io.mapRequired("type", info.type);
        io.mapRequired("size", info.size);
        io.mapOptional("value", info.value, 0);
    }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(ProfilingResult);

namespace mlir {


// namespace disagg {

struct AllocationAnnotationPass
    : public PassWrapper<AllocationAnnotationPass,
                         OperationPass<>> {
    AllocationAnnotationPass() = default;
    AllocationAnnotationPass(const AllocationAnnotationPass &pass) {}
    void runOnOperation() override;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<cira::RemoteMemDialect>();
    }

    // requires a input yaml list
    StringRef getArgument() const final { return "disagg-annotate-allocations"; }
    StringRef getDescription() const final { return "Allocate annotations"; }
    Option<std::string> memoryProfOption{
        *this, "memory-profiling",
        llvm::cl::desc("Heap Obejct Profiling Results")};
    Option<std::string> cpuProfOption{
        *this, "cpu-profiling",
        llvm::cl::desc("Heap Obejct Profiling Results")};
    Option<std::string> runtimeProfOption{
        *this, "runtime-profiling",
        llvm::cl::desc("Heap Obejct Profiling Results")};
    Option<float> memoryFactorOption{
        *this, "memory-factor",
        llvm::cl::desc("Memory factor comapred to full requirements")};
    Option<uint64_t> memorySizeOption{
        *this, "memory-size",
        llvm::cl::desc("Total Workng set size")};
    Option<bool> annotateOption{
        *this, "annotate",
        llvm::cl::desc("Perform profiling annotation")};

    std::vector<std::vector<ProfilingResult>> results{};
    std::map<uint64_t,ProfilingResult> allocationMap{};
    void parseProfilingResults();

    void propogateRemotableOperator(Operation* op, Type remoteType, Operation* parentOp = nullptr);
    void propogateRemotable();
    void duplicateFunctions();

    float cache_factor = 0.8;
    std::string configFile = "/tmp/cache.cfg";
    std::map <int, std::tuple<size_t, size_t, uint64_t>> configMap{};
    void normalizeConfig();

    std::map<std::pair<std::string,int>, std::map<llvm::hash_code, func::FuncOp>> funcOverrideMap{};
    func::FuncOp propogateRemotableFunction(func::FuncOp funcOp, int index, Type type);
};

std::unique_ptr<Pass> createAllocationAnnotationPass();

inline void registerAllocationAnnotationPass() {
    registerPass([]() -> std::unique_ptr<Pass> {
        return createAllocationAnnotationPass();
    });
}

}

bool objectLimitFilter(uint64_t number, mlir::Type type, uint64_t memoryLimit);
bool objectDependencyFilter(mlir::Operation *op);

#endif // CIRA_READFROMPROFILE_H
