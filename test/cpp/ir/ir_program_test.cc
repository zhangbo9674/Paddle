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

#include "paddle/ir/program.h"

TEST(program_test, program) {
  // Create an empty program object
  ir::Program *program = new ir::Program();
  auto ops = program->ops();
  auto weights = program->weights();
  std::cout << ops.size() << std::endl;
  std::cout << weights.size() << std::endl;
  EXPECT_EQ(program->ops().size() == 0, true);
  EXPECT_EQ(program->weights().size() == 0, true);
}
