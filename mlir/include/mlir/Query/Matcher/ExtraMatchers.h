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
  enum class MatcherType { DefinedBy, GetDefinitions, GetAllDefinitions };
  DefinitionsMatcher(matcher::DynMatcher &&innerMatcher, unsigned hops,
                     MatcherType type)
      : innerMatcher(std::move(innerMatcher)), hops(hops), type(type) {}

private:
  llvm::StringRef getID() const;
  bool matches(Operation *op, matcher::BoundOperationsGraphBuilder &Bound,
               unsigned tempHops) {
    tempStorage.push_back({op, tempHops});
    while (!tempStorage.empty()) {
      auto [currentOp, remainingHops] = tempStorage.pop_back_val();

      matcher::BoundOperationNode *currentNode = Bound.addNode(currentOp);
      if (remainingHops == 0) {
        continue;
      }

      for (auto operand : currentOp->getOperands()) {
        if (auto definingOp = operand.getDefiningOp()) {
          Bound.addEdge(currentOp, definingOp);

          if (!ccache.contains(definingOp)) {
            ccache.insert(definingOp);
            tempStorage.emplace_back(definingOp, remainingHops - 1);
          }
        }
      }

      // Need at least 1 defining op
    }
    return true;
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
  llvm::DenseSet<Operation *> ccache;
  llvm::SmallVector<std::pair<Operation *, size_t>, 4> tempStorage;

private:
  matcher::DynMatcher innerMatcher;
  unsigned hops;
  MatcherType type;
};

llvm::StringRef DefinitionsMatcher::getID() const {
  switch (type) {
  case MatcherType::DefinedBy:
    return "definedBy";
  case MatcherType::GetDefinitions:
    return "getDefinitions";
  case MatcherType::GetAllDefinitions:
    return "getAllDefinitions";
  }
  llvm_unreachable("Unknown MatcherType");
}

} // namespace detail

inline detail::DefinitionsMatcher
definedBy(mlir::query::matcher::DynMatcher innerMatcher) {
  return detail::DefinitionsMatcher(
      std::move(innerMatcher), 1,
      detail::DefinitionsMatcher::MatcherType::DefinedBy);
}

inline detail::DefinitionsMatcher
getDefinitions(mlir::query::matcher::DynMatcher innerMatcher, unsigned hops) {
  assert(hops > 0 && "hops must be >= 1");
  return detail::DefinitionsMatcher(
      std::move(innerMatcher), hops,
      detail::DefinitionsMatcher::MatcherType::GetDefinitions);
}

} // namespace extramatcher

} // namespace query

} // namespace mlir

#endif // MLIR_IR_EXTRAMATCHERS_H