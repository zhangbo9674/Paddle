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

#include "paddle/fluid/paddle_dialect/paddle_dialect.h"
#include <iostream>
#include "paddle/ir/dialect_interface.h"

namespace paddle {
namespace dialect {
PaddleDialect::PaddleDialect(ir::IrContext *context)
    : ir::Dialect(name(), context, ir::TypeId::get<PaddleDialect>()) {
  initialize();
}

void PaddleDialect::initialize() {
  // RegisterTypes<GET_BUILT_IN_TYPE_LIST>();
  // RegisterAttributes<GET_BUILT_IN_ATTRIBUTE_LIST>();
  // RegisterOps<GET_BUILT_IN_OP_LIST>();
  std::cout << "initialize dialect" << std::endl;
}

}  // namespace dialect
}  // namespace paddle
