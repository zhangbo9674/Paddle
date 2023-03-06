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

#include <type_traits>

namespace ir {
template <typename Target, typename From, typename Enabler = void>
struct isa_impl {
  static inline bool call(const From &Val) { return Target::classof(Val); }
};

template <typename Target, typename From>
struct isa_impl<
    Target,
    From,
    typename std::enable_if<std::is_base_of<Target, From>::value>::type> {
  static inline bool call(const From &) { return true; }
};

///
/// \brief The template function actually called by isa.
///
template <typename Target, typename From>
struct isa_wrap {
  static inline bool call(const From &Val) {
    return isa_impl<Target, From>::call(Val);
  }
};

///
/// \brief typequalified specialization of the isa_wrap template parameter From.
/// Specialized types include: const T, T*, const T*, T* const, const T* const.
///
template <typename Target, typename From>
struct isa_wrap<Target, const From> {
  static inline bool call(const From &Val) {
    return isa_impl<Target, From>::call(Val);
  }
};

template <typename Target, typename From>
struct isa_wrap<Target, From *> {
  static inline bool call(const From *Val) {
    if (Val == nullptr) {
      throw("isa<> used on a null pointer");
    }
    return isa_impl<Target, From>::call(*Val);
  }
};

template <typename Target, typename From>
struct isa_wrap<Target, From *const> {
  static inline bool call(const From *Val) {
    if (Val == nullptr) {
      throw("isa<> used on a null pointer");
    }
    return isa_impl<Target, From>::call(*Val);
  }
};

template <typename Target, typename From>
struct isa_wrap<Target, const From *> {
  static inline bool call(const From *Val) {
    if (Val == nullptr) {
      throw("isa<> used on a null pointer");
    }
    return isa_impl<Target, From>::call(*Val);
  }
};

template <typename Target, typename From>
struct isa_wrap<Target, const From *const> {
  static inline bool call(const From *Val) {
    if (Val == nullptr) {
      throw("isa<> used on a null pointer");
    }
    return isa_impl<Target, From>::call(*Val);
  }
};

///
/// \brief isa template function, used to determine whether the value is a
/// Target type. Using method: if (isa<Target_Type>(value)) { ... }.
///
template <class Target, class From>
inline bool isa(const From &Val) {
  return isa_wrap<typename std::remove_pointer<Target>::type, From>::call(Val);
}

///
/// \brief cast type deduction by From and To.
///
template <class To, class From>
struct cast_type_deduction {
  typedef To &return_type;
};
template <class To, class From>
struct cast_type_deduction<To, const From> {
  typedef const To &return_type;
};

template <class To, class From>
struct cast_type_deduction<To, From *> {
  typedef To *return_type;
};

template <class To, class From>
struct cast_type_deduction<To, const From *> {
  typedef const To *return_type;
};

template <class To, class From>
struct cast_type_deduction<To, const From *const> {
  typedef const To *return_type;
};

template <class To, class From>
struct cast_value {
  static typename cast_type_deduction<To, From>::return_type call(
      const From &Val) {
    typename cast_type_deduction<To, From>::return_type Res2 =
        (typename cast_type_deduction<To, From>::return_type) const_cast<
            From &>(Val);
    return Res2;
  }
};

///
/// \brief dyn_cast From to To.
///
template <class To, class From>
inline typename cast_type_deduction<To, From>::return_type dyn_cast_impl(
    From &Val) {  // NOLINT
  if (!isa<To>(Val)) {
    throw("dyn_cast_impl<To>() argument of incompatible type!");
  }
  return cast_value<To, From>::call(Val);
}

template <class To, class From>
inline typename cast_type_deduction<To, From *>::return_type dyn_cast_impl(
    From *Val) {
  if (!isa<To>(Val)) {
    throw("dyn_cast_impl<To>() argument of incompatible type!");
  }
  return cast_value<To, From *>::call(Val);
}

template <class To, class From>
inline typename cast_type_deduction<To, From>::return_type dyn_cast(
    From &Val) {  // NOLINT
  return dyn_cast_impl<To>(Val);
}

template <class To, class From>
inline typename cast_type_deduction<To, From *>::return_type dyn_cast(
    From *Val) {
  return dyn_cast_impl<To>(Val);
}

}  // namespace ir
