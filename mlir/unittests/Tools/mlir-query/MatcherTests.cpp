//===- MatcherTests.cpp - MLIR Query matcher unit tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gtest/gtest.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Query/Matcher/MatchersInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Query/Matcher/MatchFinder.h"
#include "mlir/Query/Matcher/Registry.h"
#include "mlir/Query/Matcher/SliceMatchers.h"

using namespace mlir;
using namespace mlir::matchers;
using namespace mlir::query::matcher;
using namespace mlir::query::matcher::internal;

static const char *IR = R"MLIR(
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @slice_use_from_above(%arg0: tensor<5x5xf32>, %arg1: tensor<5x5xf32>) {
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.addf %in, %in : f32
      linalg.yield %2 : f32
    } -> tensor<5x5xf32>
    %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<5x5xf32> into tensor<25xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %c2 = arith.constant 2 : index
      %extracted = tensor.extract %collapsed[%c2] : tensor<25xf32>
      %2 = arith.addf %extracted, %extracted : f32
      linalg.yield %2 : f32
    } -> tensor<5x5xf32>
    return
  }
}
)MLIR";

class QueryMatcherTest : public ::testing::Test {
protected:
  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> ctx;

  void SetUp() override {
    registerAllDialects(registry);
    ctx = std::make_unique<mlir::MLIRContext>(registry);
  }

  OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef ir) {
    return parseSourceString<mlir::ModuleOp>(ir, ctx.get());
  }
};

TEST_F(QueryMatcherTest, m_Op) {
  OwningOpRef<ModuleOp> inputModule = parse(IR);
  EXPECT_TRUE(inputModule) << "failed to parse input IR";
  bool matched = false;

  inputModule->walk([&](Operation *op) {
    if (matchPattern(op, m_Op("arith.addf")))
      matched = true;
  });

  EXPECT_TRUE(matched);
}

TEST_F(QueryMatcherTest, m_Constant) {
  OwningOpRef<ModuleOp> inputModule = parse(IR);
  EXPECT_TRUE(inputModule) << "failed to parse input IR";
  bool matched = false;

  inputModule->walk([&](Operation *op) {
    if (matchPattern(op, m_Constant()))
      matched = true;
  });

  EXPECT_TRUE(matched);
}

TEST_F(QueryMatcherTest, m_Attr) {
  OwningOpRef<ModuleOp> inputModule = parse(IR);
  EXPECT_TRUE(inputModule) << "failed to parse input IR";
  bool matched = false;

  inputModule->walk(
      [&](Operation *op) { matchPattern(op, m_Attr("indexing_maps")); });

  EXPECT_TRUE(matched);
}

TEST_F(QueryMatcherTest, m_GetAllDefinitions) {
  MatchFinder finder;
  OwningOpRef<ModuleOp> inputModule = parse(IR);
  EXPECT_TRUE(inputModule) << "failed to parse input IR";

  Matcher m = m_GetAllDefinitions((m_Op("arith.addf")), 2);
  std::vector<MatchFinder::MatchResult> results =
      finder.collectMatches(inputModule->getOperation(), m);
  EXPECT_TRUE(results.size() == 2);
}

TEST_F(QueryMatcherTest, m_GetDefinitions) {
  MatchFinder finder;
  OwningOpRef<ModuleOp> inputModule = parse(IR);
  EXPECT_TRUE(inputModule) << "failed to parse input IR";

  Matcher m = m_GetDefinitions((m_Op("arith.addf")), 2, true, true, false);
  std::vector<MatchFinder::MatchResult> results =
      finder.collectMatches(inputModule->getOperation(), m);
  EXPECT_TRUE(results.size() == 2);
}

TEST_F(QueryMatcherTest, m_GetDefinitionsByPredicate) {
  MatchFinder finder;
  OwningOpRef<ModuleOp> inputModule = parse(IR);
  EXPECT_TRUE(inputModule) << "failed to parse input IR";

  Matcher m = m_GetDefinitionsByPredicate(
      (m_Op("arith.addf")), m_Op("linalg.generic"), true, false, false);
  std::vector<MatchFinder::MatchResult> results =
      finder.collectMatches(inputModule->getOperation(), m);
  EXPECT_TRUE(results.size() == 1);
}

TEST_F(QueryMatcherTest, m_GetUsersByPredicate) {
  MatchFinder finder;
  OwningOpRef<ModuleOp> inputModule = parse(IR);
  EXPECT_TRUE(inputModule) << "failed to parse input IR";

  auto filter = m_Op(m_Op("linalg.generic"), m_Constant());
  Matcher m = m_GetUsersByPredicate(m_Op("linalg.geneirc"), filter, true);

  std::vector<MatchFinder::MatchResult> results =
      finder.collectMatches(inputModule->getOperation(), m);
  EXPECT_TRUE(results.size() == 1);
}