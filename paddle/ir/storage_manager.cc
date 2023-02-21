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

#include <memory>
#include <unordered_map>

#include "paddle/ir/storage_manager.h"

namespace ir {
// This is a structure for creating, caching, and looking up Storage of
// parameteric types.
struct ParametricStorageManager {
  using StorageBase = StorageManager::StorageBase;

  ParametricStorageManager() {}

  ~ParametricStorageManager() {
    for (const auto &instance : parametric_instances_) {
      VLOG(4) << "delete parametric_instances";
      delete instance.second;
    }
    parametric_instances_.clear();
  }

  // Get the storage of parametric type, if not in the cache, create and
  // insert the cache.
  StorageBase *GetOrCreate(std::size_t hash_value,
                           std::function<bool(const StorageBase *)> equal_func,
                           std::function<StorageBase *()> constructor) {
    if (parametric_instances_.count(hash_value) != 0) {
      auto pr = parametric_instances_.equal_range(hash_value);
      while (pr.first != pr.second) {
        if (equal_func(pr.first->second)) {
          VLOG(4) << "get cache, parameteric storage";
          return pr.first->second;
        }
        ++pr.first;
      }
    }
    VLOG(4) << "not cache, create a new parameteric storage";
    StorageBase *storage = constructor();
    parametric_instances_.emplace(hash_value, storage);
    return storage;
  }

 private:
  // In order to prevent hash conflicts, the unordered_multimap data structure
  // is used for storage.
  std::unordered_multimap<size_t, StorageBase *> parametric_instances_;
};

StorageManager::StorageManager() {}

StorageManager::~StorageManager() = default;

StorageManager::StorageBase *StorageManager::GetParametricStorageTypeImpl(
    TypeId type_id,
    std::size_t hash_value,
    std::function<bool(const StorageBase *)> equal_func,
    std::function<StorageBase *()> constructor) {
  std::lock_guard<ir::SpinLock> guard(parametric_instance_lock_);
  if (parametric_instance_.find(type_id) == parametric_instance_.end())
    throw("The input data pointer is null.");
  ParametricStorageManager &parametric_storage = *parametric_instance_[type_id];
  return parametric_storage.GetOrCreate(hash_value, equal_func, constructor);
}

StorageManager::StorageBase *StorageManager::GetParameterlessStorageTypeImpl(
    TypeId type_id) {
  std::lock_guard<ir::SpinLock> guard(parameterless_instances_lock_);
  VLOG(4) << "==> StorageManagerImpl::GetParameterlessStorage().";
  if (parameterless_instances_.find(type_id) == parameterless_instances_.end())
    throw("TypeId not found in IrContext.");
  StorageBase *parameterless_instance = parameterless_instances_[type_id];
  return parameterless_instance;
}

void StorageManager::RegisterParametricStorageTypeImpl(TypeId type_id) {
  std::lock_guard<ir::SpinLock> guard(parametric_instance_lock_);
  parametric_instance_.emplace(type_id,
                               std::make_unique<ParametricStorageManager>());
}

void StorageManager::RegisterParameterlessStorageTypeImpl(
    TypeId type_id, std::function<StorageBase *()> constructor) {
  std::lock_guard<ir::SpinLock> guard(parameterless_instances_lock_);
  VLOG(4) << "==> StorageManager::RegisterParameterlessStorageTypeImpl()";
  if (parameterless_instances_.find(type_id) != parameterless_instances_.end())
    throw("storage class already registered");
  parameterless_instances_.emplace(type_id, constructor());
}

}  // namespace ir
