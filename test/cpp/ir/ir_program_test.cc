// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include "paddle/ir/builtin_dialect.h"
#include "paddle/ir/builtin_op.h"
#include "paddle/ir/builtin_type.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/program.h"
#include "paddle/ir/utils.h"

TEST(program_test, program) {
  // (1) Init environment.
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Dialect *builtin_dialect =
      ctx->GetOrRegisterDialect<ir::BuiltinDialect>();

  // (2) Create an empty program object
  ir::Program *program = new ir::Program();
  EXPECT_EQ(program->ops().size() == 0, true);
  EXPECT_EQ(program->weights().size() == 0, true);

  // (3) Def a program:
  // a = GetParameterOp("a")
  std::string op1_name =
      builtin_dialect->name() + "." + std::string(ir::GetParameterOp::name());
  ir::OpInfoImpl *op1_info = ctx->GetRegisteredOpInfo(op1_name);
  std::map<ir::StrAttribute, ir::Attribute> op1_attribute_map{
      {ir::StrAttribute::get(ctx, "parameter_name"),
       ir::StrAttribute::get(ctx, "a")}};
  ir::DictionaryAttribute op1_attribute =
      ir::DictionaryAttribute::get(ctx, op1_attribute_map);
  ir::Operation *op1 = ir::Operation::create(
      {}, {ir::Float32Type::get(ctx)}, op1_attribute, op1_info, program);

  // b = GetParameterOp("b")
  std::string op2_name =
      builtin_dialect->name() + "." + std::string(ir::GetParameterOp::name());
  ir::OpInfoImpl *op2_info = ctx->GetRegisteredOpInfo(op2_name);
  std::map<ir::StrAttribute, ir::Attribute> op2_attribute_map{
      {ir::StrAttribute::get(ctx, "parameter_name"),
       ir::StrAttribute::get(ctx, "b")}};
  ir::DictionaryAttribute op2_attribute =
      ir::DictionaryAttribute::get(ctx, op2_attribute_map);
  ir::Operation *op2 = ir::Operation::create(
      {}, {ir::Float32Type::get(ctx)}, op2_attribute, op2_info, program);

  std::cout << op1->op_name() << std::endl;
  std::cout << op2->op_name() << std::endl;

  // c = AddOp(a, b)
  // SetParameterOp(c, "c")
}
