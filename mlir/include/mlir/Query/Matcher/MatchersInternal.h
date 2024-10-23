//===- MatchersInternal.h - Structural query framework ----------*- C++ -*-===//
//
// Implements the base layer of the matcher framework.
//
// Matchers are methods that return a Matcher which provides a method
// match(Operation *op)
//
// The matcher functions are defined in include/mlir/IR/Matchers.h.
// This file contains the wrapper classes needed to construct matchers for
// mlir-query.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERSINTERNAL_H
#define MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERSINTERNAL_H

#include "mlir/IR/Matchers.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include <type_traits>

namespace mlir::query::matcher {

class BoundOperations {
public:
  void bind(Operation *op) { operations.push_back(op); }

  std::vector<Operation *> &getOperations() { return operations; }

private:
  std::vector<Operation *> operations;
};

// Generic interface for matchers on an MLIR operation.
class MatcherInterface
    : public llvm::ThreadSafeRefCountedBase<MatcherInterface> {
public:
  virtual ~MatcherInterface() = default;

  virtual bool match(Operation *op) = 0;
  virtual bool match(Operation *op, BoundOperations &Bound) = 0;
};

// Helper traits to detect if MatcherFn has certain match methods.
namespace matcher_detail {

template <typename T, typename = void>
struct has_match_op : std::false_type {};

template <typename T>
struct has_match_op<T, std::void_t<decltype(std::declval<T &>().match(
                           std::declval<Operation *>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_match_op_bound : std::false_type {};

template <typename T>
struct has_match_op_bound<
    T, std::void_t<decltype(std::declval<T &>().match(
           std::declval<Operation *>(), std::declval<BoundOperations &>()))>>
    : std::true_type {};

} // namespace matcher_detail

// MatcherFnImpl takes a matcher function object and implements
// MatcherInterface.
template <typename MatcherFn>
class MatcherFnImpl : public MatcherInterface {
public:
  MatcherFnImpl(MatcherFn matcherFn) : matcherFn(std::move(matcherFn)) {}

  bool match(Operation *op) override { return match_impl(op); }

  bool match(Operation *op, BoundOperations &Bound) override {
    return match_impl(op, Bound);
  }

private:
  bool match_impl(Operation *op) {
    if constexpr (matcher_detail::has_match_op<MatcherFn>::value) {
      return matcherFn.match(op);
    } else if constexpr (matcher_detail::has_match_op_bound<MatcherFn>::value) {
      BoundOperations Bound;
      return matcherFn.match(op, Bound);
    }
    // No fallback needed; one of the match methods must be available.
  }

  bool match_impl(Operation *op, BoundOperations &Bound) {
    if constexpr (matcher_detail::has_match_op_bound<MatcherFn>::value) {
      return matcherFn.match(op, Bound);
    } else if constexpr (matcher_detail::has_match_op<MatcherFn>::value) {
      return matcherFn.match(op);
    }
    // No fallback needed; one of the match methods must be available.
  }

  MatcherFn matcherFn;
};

// Matcher wraps a MatcherInterface implementation and provides match()
// methods that redirect calls to the underlying implementation.
class DynMatcher {
public:
  // Takes ownership of the provided implementation pointer.
  DynMatcher(MatcherInterface *implementation, StringRef matcherName)
      : implementation(implementation), matcherName(matcherName) {}

  template <typename MatcherFn>
  static std::unique_ptr<DynMatcher>
  constructDynMatcherFromMatcherFn(MatcherFn &matcherFn,
                                   StringRef matcherName) {
    auto impl = new MatcherFnImpl<MatcherFn>(matcherFn);
    return std::make_unique<DynMatcher>(impl, matcherName);
  }

  bool match(Operation *op) { return implementation->match(op); }

  bool match(Operation *op, BoundOperations &Bound) {
    return implementation->match(op, Bound);
  }

  void setFunctionName(StringRef name) { functionName = name.str(); }
  bool hasFunctionName() const { return !functionName.empty(); }
  StringRef getFunctionName() const { return functionName; }
  StringRef getMatcherName() const { return matcherName; }

private:
  llvm::IntrusiveRefCntPtr<MatcherInterface> implementation;
  std::string functionName;
  std::string matcherName;
};

} // namespace mlir::query::matcher

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_MATCHERSINTERNAL_H
