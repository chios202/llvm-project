//===- MatchersInternal.h - Structural query framework ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERSINTERNAL_H
#define MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERSINTERNAL_H

#include "mlir/IR/Matchers.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/MapVector.h"
#include <memory>
#include <stack>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace mlir::query::matcher {

struct BoundOperationNode {
  Operation *op;
  std::vector<BoundOperationNode *> parents;
  std::vector<BoundOperationNode *> children;

  bool highlightText_;
  bool detailedPrinting_;

  BoundOperationNode(Operation *operation, bool highlightText = false,
                     bool detailedPrinting = false)
      : op(operation), highlightText_(highlightText),
        detailedPrinting_(detailedPrinting) {}
};

class BoundOperationsGraphBuilder {
public:
  // Adds a node to the graph or retrieves it if it already exists
  BoundOperationNode *addNode(Operation *op, bool highlight = false,
                              bool detailedPrinting = false) {
    auto it = nodes.find(op);
    if (it != nodes.end()) {
      return it->second.get();
    }
    auto node =
        std::make_unique<BoundOperationNode>(op, highlight, detailedPrinting);
    BoundOperationNode *nodePtr = node.get();
    nodes[op] = std::move(node);
    return nodePtr;
  }

  bool addEdge(Operation *parentOp, Operation *childOp) {
    BoundOperationNode *parentNode = addNode(parentOp, false, false);
    BoundOperationNode *childNode = addNode(childOp, false, false);

    if (createsCycle(childNode, parentNode)) {
      return false;
    }

    parentNode->children.push_back(childNode);
    childNode->parents.push_back(parentNode);
    return true;
  }

  BoundOperationNode *getNode(Operation *op) const {
    auto it = nodes.find(op);
    return it != nodes.end() ? it->second.get() : nullptr;
  }

  // Access to all nodes in insertion order
  const llvm::MapVector<Operation *, std::unique_ptr<BoundOperationNode>> &
  getNodes() const {
    return nodes;
  }

private:
    llvm::MapVector<Operation *, std::unique_ptr<BoundOperationNode>> nodes;

  // Helper function to detect if adding an edge creates a cycle
  bool createsCycle(BoundOperationNode *startNode,
                    BoundOperationNode *targetNode) const {
    std::unordered_set<BoundOperationNode *> visited;
    std::stack<BoundOperationNode *> stack;
    stack.push(startNode);

    while (!stack.empty()) {
      BoundOperationNode *current = stack.top();
      stack.pop();

      if (current == targetNode) {
        // Cycle detected
        return true;
      }

      if (visited.find(current) != visited.end()) {
        continue;
      }
      visited.insert(current);

      for (BoundOperationNode *child : current->children) {
        stack.push(child);
      }
    }

    // No cycle detected
    return false;
  }
};

// Type trait to detect if a matcher has a match(Operation*) method
template <typename T, typename = void>
struct has_simple_match : std::false_type {};

template <typename T>
struct has_simple_match<T, std::void_t<decltype(std::declval<T>().match(
                               std::declval<Operation *>()))>>
    : std::true_type {};

// Type trait to detect if a matcher has a match(Operation*,
// BoundOperationsGraphBuilder&) method
template <typename T, typename = void>
struct has_bound_match : std::false_type {};

template <typename T>
struct has_bound_match<T, std::void_t<decltype(std::declval<T>().match(
                              std::declval<Operation *>(),
                              std::declval<BoundOperationsGraphBuilder &>()))>>
    : std::true_type {};

// Generic interface for matchers on an MLIR operation.
class MatcherInterface
    : public llvm::ThreadSafeRefCountedBase<MatcherInterface> {
public:
  virtual ~MatcherInterface() = default;
  virtual bool match(Operation *op) = 0;
  virtual bool match(Operation *op, BoundOperationsGraphBuilder &bound) = 0;
};

// MatcherFnImpl takes a matcher function object and implements
// MatcherInterface.
template <typename MatcherFn>
class MatcherFnImpl : public MatcherInterface {
public:
  MatcherFnImpl(MatcherFn &matcherFn) : matcherFn(matcherFn) {}

  bool match(Operation *op) override {
    if constexpr (has_simple_match<MatcherFn>::value)
      return matcherFn.match(op);
    return false;
  }

  bool match(Operation *op, BoundOperationsGraphBuilder &bound) override {
    if constexpr (has_bound_match<MatcherFn>::value)
      return matcherFn.match(op, bound);
    return false;
  }

private:
  MatcherFn matcherFn;
};

// Matcher wraps a MatcherInterface implementation and provides match()
// methods that redirect calls to the underlying implementation.
class DynMatcher {
public:
  // Takes ownership of the provided implementation pointer.
  DynMatcher(MatcherInterface *implementation, StringRef matcherName)
      : implementation(implementation), matcherName(matcherName.str()) {}

  template <typename MatcherFn>
  static std::unique_ptr<DynMatcher>
  constructDynMatcherFromMatcherFn(MatcherFn &matcherFn,
                                   StringRef matcherName) {
    auto impl = std::make_unique<MatcherFnImpl<MatcherFn>>(matcherFn);
    return std::make_unique<DynMatcher>(impl.release(), matcherName);
  }

  bool match(Operation *op) const { return implementation->match(op); }
  bool match(Operation *op, BoundOperationsGraphBuilder &bound) const {
    return implementation->match(op, bound);
  }

  void setFunctionName(StringRef name) { functionName = name.str(); }
  void setMatcherName(StringRef name) { matcherName = name.str(); }
  bool hasFunctionName() const { return !functionName.empty(); }
  StringRef getFunctionName() const { return functionName; }
  StringRef getMatcherName() const { return matcherName; }

private:
  llvm::IntrusiveRefCntPtr<MatcherInterface> implementation;
  std::string matcherName;
  std::string functionName;
};

} // namespace mlir::query::matcher

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERSINTERNAL_H