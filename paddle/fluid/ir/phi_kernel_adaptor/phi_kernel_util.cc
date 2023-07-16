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

#include "paddle/fluid/ir/phi_kernel_adaptor/phi_kernel_util.h"

#include "paddle/fluid/ir/dialect/op_yaml_info_util.h"
#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/utils.h"
#include "paddle/phi/core/meta_tensor.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/phi/core/kernel_context.h"

#include "paddle/fluid/framework/string_array.h"
#include "paddle/fluid/framework/tensor_ref_array.h"
#include "paddle/fluid/ir/dialect/kernel_attribute.h"
#include "paddle/fluid/ir/dialect/kernel_type.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/fluid/ir/interface/op_yaml_info_parser.h"
#include "paddle/phi/core/enforce.h"

#include "glog/logging.h"

namespace ir {

using VariableNameMap =
    std::unordered_map<const paddle::framework::Variable*, std::string>;

paddle::framework::Variable* CreateVar(
    ir::Value value,
    paddle::framework::Scope* inner_scope,
    bool force_persisable,
    std::unordered_map<ir::Value, std::string>* value_2_var_name,
    std::unordered_map<const paddle::framework::Variable*, std::string>*
        variable_2_var_name,
    std::map<std::string, int>* var_name_2_id,
    std::vector<paddle::framework::Variable*>* variable_list) {
  Operation* def_op = value.GetDefiningOp();
  bool is_persisable = false;
  if (def_op->attributes().count("is_persisable")) {
    is_persisable = def_op->attributes()
                        .at("is_persisable")
                        .dyn_cast<ir::BoolAttribute>()
                        .data();
  }

  paddle::framework::Variable* var = nullptr;
  std::string name = "inner_var_" + std::to_string(variable_2_var_name->size());
  if (force_persisable || is_persisable) {
    VLOG(6) << "Create var: " << name << " in scope " << inner_scope->root();
    var = const_cast<paddle::framework::Scope*>(inner_scope->root())->Var(name);
  } else {
    VLOG(6) << "Create var: " << name << " in scope " << inner_scope;
    var = inner_scope->Var(name);
  }
  value_2_var_name->emplace(value, name);
  variable_2_var_name->emplace(var, name);
  auto id = var_name_2_id->size();
  var_name_2_id->emplace(name, id);
  variable_list->push_back(var);
  PADDLE_ENFORCE_EQ(
      variable_list->size(),
      var_name_2_id->size(),
      paddle::platform::errors::InvalidArgument(
          "The size of variable_list and var_name_2_id map should be equal"));
  return var;
}

void CheckInputVars(
    ir::Operation* op,
    const std::string& op_name,
    const std::unordered_map<ir::Value, std::string>& value_2_var_name) {
  size_t input_num = op->num_operands();
  if (input_num > 0) {
    for (size_t i = 0; i < input_num; ++i) {
      auto value = op->operand(i);
      if (value) {
        PADDLE_ENFORCE_NE(
            value_2_var_name.find(value),
            value_2_var_name.end(),
            phi::errors::PreconditionNotMet(
                "input should in name map, [%d] 'th input of [%s] op",
                i,
                op_name));
      }
    }
  }
}

void BuildValue(ir::Value value,
                paddle::framework::Scope* inner_scope,
                std::unordered_map<ir::Value, std::string>* value_2_var_name,
                std::unordered_map<const paddle::framework::Variable*,
                                   std::string>* variable_2_var_name,
                std::map<std::string, int>* var_name_2_id,
                std::vector<paddle::framework::Variable*>* variable_list) {
  paddle::framework::Variable* var = nullptr;
  VLOG(0) << "value 1";
  if (value_2_var_name->find(value) != value_2_var_name->end()) {
    var = inner_scope->FindVar(value_2_var_name->at(value));
    VLOG(0) << "value 2";
  } else {
    var = CreateVar(value,
                    inner_scope,
                    false,
                    value_2_var_name,
                    variable_2_var_name,
                    var_name_2_id,
                    variable_list);
    VLOG(0) << "value 3";
  }

  // Only support DenseTensor or Vector<DenseTensor>
  if (!value.type()) {
    var->GetMutable<phi::DenseTensor>();
    VLOG(0) << "value 4";
  } else if (value.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
    var->GetMutable<phi::DenseTensor>();
    VLOG(0) << "value 5";
  } else if (value.type().isa<paddle::dialect::AllocatedSelectedRowsType>()) {
    var->GetMutable<phi::SelectedRows>();
    VLOG(0) << "value 6";
  } else if (value.type().isa<ir::VectorType>()) {
    auto tensor_array = var->GetMutable<paddle::framework::VariableRefArray>();
    VLOG(0) << "value 7";
    for (size_t i = 0; i < value.type().dyn_cast<ir::VectorType>().size();
         i++) {
      VLOG(0) << "value 8";
      PADDLE_ENFORCE(value.type()
                         .dyn_cast<ir::VectorType>()[i]
                         .isa<paddle::dialect::AllocatedDenseTensorType>(),
                     paddle::platform::errors::Fatal(
                         "Element of VectorType output only support "
                         "DenseTensorType"));
      VLOG(0) << "value 9";
      auto var_i = CreateVar(value,
                             inner_scope,
                             false,
                             value_2_var_name,
                             variable_2_var_name,
                             var_name_2_id,
                             variable_list);
      VLOG(0) << "value 10";
      var_i->GetMutable<phi::DenseTensor>();
      VLOG(0) << "value 11";
      tensor_array->emplace_back(var_i);
      VLOG(0) << "value 12";
    }
  } else {
    PADDLE_THROW(phi::errors::PreconditionNotMet(
        "Output only support DenseTensorType or VectorType"));
    VLOG(0) << "value 13";
  }
}

void HandleForSpecialOp(
    ir::Operation* op,
    paddle::framework::Scope* inner_scope,
    std::unordered_map<ir::Value, std::string>* value_2_var_name,
    std::unordered_map<const paddle::framework::Variable*, std::string>*
        variable_2_var_name,
    std::map<std::string, int>* var_name_2_id,
    std::vector<paddle::framework::Variable*>* variable_list) {
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().data();
  }

  if (op_name == "pd.fetch") {
    // fetch is a very special op, with no output
    auto var = const_cast<paddle::framework::Scope*>(inner_scope->root())
                   ->Var("fetch");
    VLOG(0) << "fetch 1";
    VLOG(6) << "Create var: fetch in scope " << inner_scope->root();
    auto fetch_list = var->GetMutable<paddle::framework::FetchList>();
    VLOG(0) << "fetch 2";
    int index =
        op->attributes().at("col").dyn_cast<ir::Int32Attribute>().data();
    VLOG(0) << "fetch 3";
    fetch_list->resize(index + 1);
    VLOG(0) << "fetch 4";
  }

  if (op_name == "pd.feed") {
    auto value = op->result(0);
    VLOG(0) << "feed 1";
    auto var = CreateVar(value,
                         inner_scope,
                         false,
                         value_2_var_name,
                         variable_2_var_name,
                         var_name_2_id,
                         variable_list);
    VLOG(0) << "feed 2";
    // TODO(phlrain): need to update here, support StringTensor
    auto out_tensor = var->GetMutable<phi::DenseTensor>();
    VLOG(0) << "feed 3";
    auto feed_var =
        const_cast<paddle::framework::Scope*>(inner_scope->root())->Var("feed");
    VLOG(6) << "Create var: feed in scope " << inner_scope->root();
    VLOG(0) << "feed 4";
    int index =
        op->attributes().at("col").dyn_cast<ir::Int32Attribute>().data();
    VLOG(0) << "feed 5";
    auto feed_list = feed_var->Get<paddle::framework::FeedList>();
    VLOG(0) << "feed 6";
    auto& in_tensor = (PADDLE_GET(phi::DenseTensor, feed_list.at(index)));
    VLOG(0) << "feed 7";
    out_tensor->ShareDataWith(in_tensor);
    VLOG(0) << "feed 8";
    out_tensor->set_lod(in_tensor.lod());
    VLOG(0) << "feed 9";
  }

  if (op_name == "builtin.combine") {
    auto out_value = op->result(0);

    paddle::framework::Variable* var = nullptr;
    if (value_2_var_name->find(out_value) != value_2_var_name->end()) {
      var = inner_scope->FindVar(value_2_var_name->at(out_value));
    } else {
      var = CreateVar(out_value,
                      inner_scope,
                      false,
                      value_2_var_name,
                      variable_2_var_name,
                      var_name_2_id,
                      variable_list);
    }

    auto tensor_array = var->GetMutable<paddle::framework::VariableRefArray>();
    // clear tensor array
    tensor_array->clear();
    size_t input_num = op->num_operands();
    for (size_t i = 0; i < input_num; ++i) {
      auto value = op->operand(i);
      PADDLE_ENFORCE_EQ(
          value_2_var_name->count(value),
          true,
          phi::errors::PreconditionNotMet("can not found input of combine op"));
      tensor_array->emplace_back(
          inner_scope->FindVar(value_2_var_name->at(value)));
    }
  }

  if (op_name == "builtin.set_parameter") {
    VLOG(6) << "Handle for builtin.set_parameter:";
    auto param_name = op->attributes()
                          .at("parameter_name")
                          .dyn_cast<ir::StrAttribute>()
                          .data();

    auto value = op->operand(0);
    // change opreand name to param_name
    auto orig_name = value_2_var_name->at(value);

    if (inner_scope->root()->FindVar(param_name) == nullptr) {
      const_cast<paddle::framework::Scope*>(inner_scope->root())
          ->Rename(orig_name, param_name);
    }
    (*value_2_var_name)[value] = param_name;
  }

  if (op_name == "builtin.get_parameter") {
    VLOG(6) << "Handle for builtin.get_parameter:";
    auto param_name = op->attributes()
                          .at("parameter_name")
                          .dyn_cast<ir::StrAttribute>()
                          .data();
    auto value = op->result(0);
    value_2_var_name->emplace(value, param_name);
  }

  if (op_name == "builtin.slice") {
    VLOG(6) << "Handle for builtin.slice";
    auto out_value = op->result(0);
    auto in_value = op->operand(0);
    PADDLE_ENFORCE_EQ(value_2_var_name->count(in_value),
                      true,
                      phi::errors::PreconditionNotMet(
                          "input of buildin slice not in name map"));

    int index =
        op->attributes().at("index").dyn_cast<ir::Int32Attribute>().data();
    auto in_var = inner_scope->FindVar(value_2_var_name->at(in_value));
    auto variable_array = in_var->Get<paddle::framework::VariableRefArray>();

    PADDLE_ENFORCE_EQ(
        variable_2_var_name->count(variable_array[index]),
        true,
        phi::errors::PreconditionNotMet("[%d] the variable in build slice "
                                        "input MUST in variable name map",
                                        index));

    std::string var_name = variable_2_var_name->at(variable_array[index]);
    value_2_var_name->emplace(out_value, var_name);
  }
}

void HandleForInplaceOp(
    ir::Operation* op,
    paddle::framework::Scope* inner_scope,
    std::unordered_map<ir::Value, std::string>* value_2_var_name,
    std::unordered_map<const paddle::framework::Variable*, std::string>*
        variable_2_var_name,
    std::map<std::string, int>* var_name_2_id,
    std::vector<paddle::framework::Variable*>* variable_list) {
  if (op->num_results() < 1) return;
  VLOG(0) << "inplace 1";
  ir::IrContext* ctx = ir::IrContext::Instance();
  VLOG(0) << "inplace 2";
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().data();
  }
  VLOG(0) << "inplace 3";
  ir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);
  VLOG(0) << "inplace 4";
  paddle::dialect::OpYamlInfoParser yaml_parser(
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
          ->get_op_info_());
  VLOG(0) << "inplace 5";
  for (size_t i = 0; i < op->num_results(); ++i) {
    VLOG(0) << "inplace 6";
    ir::Value value = op->result(i);
    VLOG(0) << "inplace 7";
    std::string value_name = yaml_parser.OutputNames()[i];
    VLOG(0) << "inplace 8";
    if (yaml_parser.HasInplace(value_name)) {
      std::string inplace_name = yaml_parser.InplaceName(value_name);
      VLOG(0) << "inplace 9";
      ir::Value inplace_value =
          op->operand(yaml_parser.InputName2Id().at(inplace_name));
      VLOG(0) << "inplace 10";
      std::string var_name = value_2_var_name->at(inplace_value);
      VLOG(0) << "inplace: " << value_name << " -> " << inplace_name
              << " (var: " << var_name << ")";
      value_2_var_name->emplace(value, var_name);
      VLOG(0) << "inplace 11";
    } else {
      VLOG(0) << "inplace 12";
      BuildValue(value,
                 inner_scope,
                 value_2_var_name,
                 variable_2_var_name,
                 var_name_2_id,
                 variable_list);
      VLOG(0) << "inplace 13";
    }
  }
}

// NOTE(zhiqiu): the persistable is created in inner_scope's root, and other is
// created in inner_scope.
void BuildScope(const ir::Block& block,
                paddle::framework::Scope* inner_scope,
                std::unordered_map<ir::Value, std::string>* value_2_var_name,
                std::unordered_map<const paddle::framework::Variable*,
                                   std::string>* variable_2_var_name,
                std::map<std::string, int>* var_name_2_id,
                std::vector<paddle::framework::Variable*>* variable_list) {
  VLOG(0) << "***** [before build] scope"
          << "(" << inner_scope << ") ******\n"
          << paddle::framework::GenScopeTreeDebugInfo(
                 const_cast<paddle::framework::Scope*>(inner_scope->root()));

  // int count = value_2_var_name->size();
  for (auto it = block.begin(); it != block.end(); ++it) {
    ir::Operation* op = *it;
    VLOG(0) << "Build scope 1";

    std::string op_name = op->name();
    VLOG(0) << "Build scope 2 " << op_name;
    if (op->attributes().count("op_name")) {
      op_name =
          op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().data();
    }
    VLOG(0) << "build op:" << op_name;

    if (op_name == "pd.feed" || op_name == "pd.fetch" ||
        op_name == "builtin.combine" || op_name == "builtin.set_parameter" ||
        op_name == "builtin.get_parameter" || op_name == "builtin.slice") {
      HandleForSpecialOp(op,
                         inner_scope,
                         value_2_var_name,
                         variable_2_var_name,
                         var_name_2_id,
                         variable_list);
      VLOG(0) << "Build scope HandleForSpecialOp";
      continue;
    }

    CheckInputVars(op, op_name, *value_2_var_name);
    VLOG(0) << "Build scope CheckInputVars";
    if (op->num_results() < 1) continue;
    if (op->attributes().count("is_inplace") != 0 &&
        op->attributes()
            .at("is_inplace")
            .dyn_cast<ir::BoolAttribute>()
            .data()) {
      HandleForInplaceOp(op,
                         inner_scope,
                         value_2_var_name,
                         variable_2_var_name,
                         var_name_2_id,
                         variable_list);
      VLOG(0) << "Build scope HandleForInplaceOp";
      continue;
    } else {
      for (size_t i = 0; i < op->num_results(); ++i) {
        BuildValue(op->result(i),
                   inner_scope,
                   value_2_var_name,
                   variable_2_var_name,
                   var_name_2_id,
                   variable_list);
        VLOG(0) << "Build scope BuildValue";
      }
    }
  }

  VLOG(0) << "***** [after build] scope"
          << "(" << inner_scope << ") ******\n"
          << paddle::framework::GenScopeTreeDebugInfo(
                 const_cast<paddle::framework::Scope*>(inner_scope->root()));
}

}  // namespace ir
