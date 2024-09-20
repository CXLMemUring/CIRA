//
// Created by yangyw on 9/19/24.
//

#ifndef CIRA_LOWERING_PASSES_H
#define CIRA_LOWERING_PASSES_H

#include "Lowering/EmitLLVM.h"
#include "Lowering/RemoteMemToLLVM.h"
#include "mlir/Pass/Pass.h"
namespace mlir {

#define GEN_PASS_REGISTRATION
#include "Lowering/Passes.h.inc"
}

#endif // CIRA_LOWERING_PASSES_H
