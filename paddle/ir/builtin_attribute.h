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

#include "paddle/ir/attribute.h"
#include "paddle/ir/builtin_attribute_storage.h"
#include "paddle/ir/utils.h"

namespace ir {
///
/// \brief All built-in attributes.
///
#define GET_BUILT_IN_ATTRIBUTE_LIST                                           \
  StrAttribute, BoolAttribute, FloatAttribute, DoubleAttribute, IntAttribute, \
      Int32_tAttribute, Int64_tAttribute

class StrAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(StrAttribute, StrAttributeStorage);

  bool operator<(const StrAttribute &right) const {
    return storage() < right.storage();
  }

  std::string data() const;

  uint32_t size() const;
};

class BoolAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(BoolAttribute, BoolAttributeStorage);

  bool data() const;
};

class FloatAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(FloatAttribute, FloatAttributeStorage);

  float data() const;
};

class DoubleAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(DoubleAttribute, DoubleAttributeStorage);

  double data() const;
};

class IntAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(IntAttribute, IntAttributeStorage);

  int data() const;
};

class Int32_tAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(Int32_tAttribute, Int32_tAttributeStorage);

  int32_t data() const;
};

class Int64_tAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(Int64_tAttribute, Int64_tAttributeStorage);

  int64_t data() const;
};

}  // namespace ir

namespace std {
template <>
struct hash<ir::StrAttribute> {
  std::size_t operator()(const ir::StrAttribute &obj) const {
    return std::hash<const ir::StrAttribute::Storage *>()(obj.storage());
  }
};

template <>
struct hash<ir::BoolAttribute> {
  std::size_t operator()(const ir::BoolAttribute &obj) const {
    return std::hash<const ir::BoolAttribute::Storage *>()(obj.storage());
  }
};

template <>
struct hash<ir::FloatAttribute> {
  std::size_t operator()(const ir::FloatAttribute &obj) const {
    return std::hash<const ir::FloatAttribute::Storage *>()(obj.storage());
  }
};

template <>
struct hash<ir::DoubleAttribute> {
  std::size_t operator()(const ir::DoubleAttribute &obj) const {
    return std::hash<const ir::DoubleAttribute::Storage *>()(obj.storage());
  }
};

template <>
struct hash<ir::IntAttribute> {
  std::size_t operator()(const ir::IntAttribute &obj) const {
    return std::hash<const ir::IntAttribute::Storage *>()(obj.storage());
  }
};

template <>
struct hash<ir::Int32_tAttribute> {
  std::size_t operator()(const ir::Int32_tAttribute &obj) const {
    return std::hash<const ir::Int32_tAttribute::Storage *>()(obj.storage());
  }
};

template <>
struct hash<ir::Int64_tAttribute> {
  std::size_t operator()(const ir::Int64_tAttribute &obj) const {
    return std::hash<const ir::Int64_tAttribute::Storage *>()(obj.storage());
  }
};
}  // namespace std
