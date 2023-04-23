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

#pragma once

#include "paddle/ir/type.h"

namespace ir {
///
/// \brief Variable represents the weight in the calculation graph.
///
class Variable {
 public:
  TypeId type() const { return type_; }

  void* data() const { return data_; }

  bool is_mutable() const { return is_mutable_; }

 private:
  TypeId type_;

  void* data_;

  bool is_mutable_;
};

}  // namespace ir
