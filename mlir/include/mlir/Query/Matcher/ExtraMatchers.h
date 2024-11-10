//===- Matchers.h - Various common matchers ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides extra matchers that are very useful for mlir-query
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_EXTRAMATCHERS_H
#define MLIR_IR_EXTRAMATCHERS_H

#include "MatchFinder.h"
#include "MatchersInternal.h"

namespace mlir {

namespace query {

namespace extramatcher {

namespace detail {

class DefinitionsMatcher {
public:
  DefinitionsMatcher(matcher::DynMatcher &&innerMatcher, unsigned hops)
      : innerMatcher(std::move(innerMatcher)), hops(hops) {}

private:
  bool matches(Operation *op, matcher::BoundOperationsGraphBuilder &Bound,
               unsigned tempHops) {
    tempStorage.push_back({op, tempHops});
    while (!tempStorage.empty()) {
      auto [currentOp, remainingHops] = tempStorage.pop_back_val();

      matcher::BoundOperationNode *currentNode =
          Bound.addNode(currentOp, true, true);
      if (remainingHops == 0) {
        continue;
      }

      for (auto operand : currentOp->getOperands()) {
        if (auto definingOp = operand.getDefiningOp()) {
          Bound.addEdge(currentOp, definingOp);
          if (!ccache.contains(operand)) {
            ccache.insert(operand);
            tempStorage.emplace_back(definingOp, remainingHops - 1);
          }
        } else if (auto blockArg = operand.dyn_cast<BlockArgument>()) {
          auto *block = blockArg.getOwner();

          if (block->isEntryBlock() &&
              isa<FunctionOpInterface>(block->getParentOp())) {
            // Do not proceed further if it's a function block argument
            continue;
          }

          Operation *parentOp = blockArg.getOwner()->getParentOp();
          if (parentOp) {
            Bound.addEdge(currentOp, parentOp);
            if (!!ccache.contains(blockArg)) {
              ccache.insert(blockArg);
              tempStorage.emplace_back(parentOp, remainingHops - 1);
            }
          }
        }
      }

      // Need at least 1 defining op
    }
    return ccache.size() >= 2;
  }

public:
  bool match(Operation *op, matcher::BoundOperationsGraphBuilder &Bound) {
    ccache.clear();
    tempStorage.clear();
    if (innerMatcher.match(op) && matches(op, Bound, hops)) {
      return true;
    }
    return false;
  }

private:
  llvm::DenseSet<mlir::Value> ccache;
  llvm::SmallVector<std::pair<Operation *, size_t>, 4> tempStorage;

private:
  matcher::DynMatcher innerMatcher;
  unsigned hops;
};
} // namespace detail

inline detail::DefinitionsMatcher
definedBy(mlir::query::matcher::DynMatcher innerMatcher) {
  return detail::DefinitionsMatcher(std::move(innerMatcher), 1);
}

inline detail::DefinitionsMatcher
getDefinitions(mlir::query::matcher::DynMatcher innerMatcher, unsigned hops) {
  assert(hops > 0 && "hops must be >= 1");
  return detail::DefinitionsMatcher(std::move(innerMatcher), hops);
}

} // namespace extramatcher

} // namespace query

} // namespace mlir

#endif // MLIR_IR_EXTRAMATCHERS_H