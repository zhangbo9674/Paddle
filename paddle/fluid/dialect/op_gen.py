# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import pathlib
import sys

import yaml

sys.path.append(
    str(pathlib.Path(__file__).parents[2].joinpath('phi/api/yaml/generator'))
)
from api_base import BaseAPI


# Script parameter parsing
def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Generate Paddle Dialect OP Definition Files'
    )
    parser.add_argument('--op_yaml', type=str)
    parser.add_argument('--op_header_file', type=str)
    parser.add_argument('--op_source_file', type=str)
    parser.add_argument('--namespaces', type=str)
    args = parser.parse_args()
    return args


# string template for pd_op.h
NAMESPACE_GARD_TEMPLATE = """namespace {namespace} {{
{input}
}} // namespace {namespace}"""

H_FILE_TEMPLATE = """#pragma once

#include "paddle/ir/op_base.h"

{input}
"""

CC_FILE_TEMPLATE = """#include "{h_file}"
#include "paddle/fluid/dialect/pd_type.h"
#include "paddle/fluid/dialect/pd_attribute.h"
#include "paddle/ir/builtin_attribute.h"
#include "paddle/ir/builtin_type.h"
#include "paddle/ir/ir_context.h"
#include "paddle/phi/core/enforce.h"

{input}
"""

GET_PD_DIALECT_OP_LIST_TEMPALTE = """
// All defined pd_dialect operations in this file.
#define GET_PD_DIALECT_OP_LIST {}
"""

OP_DECLARE_ATTRIBUTES_TEMPLATE = """
class {op_name}Op : public ir::Op<{op_name}Op{interfaces}{traits}> {{
 public:
  using Op::Op;
  static const char *name() {{ return "{op_name}Op"; }}
  {attribute_declare}
  static constexpr uint32_t attributes_num_ = {attribute_num};
  static void verify(const std::vector<ir::Type> &inputs, const std::vector<ir::Type> &outputs, const ir::AttributeMap &attributes);
}};
"""
op_0_attribute_declare_str = "static const char **attributes_name_;"
op_n_attribute_declare_str = "static const char *attributes_name_[{}];"

OP_0_ATTRIBUTE_DEFINED_TEMPLATE = """
const char **{op_name}Op::attributes_name_ = nullptr;
"""
OP_N_ATTRIBUTE_DEFINED_TEMPLATE = """
const char *{op_name}Op::attributes_name_[{attribute_num}] = {{ {attribute_names} }};
"""

OP_VERIFY_TEMPLATE = """
void {op_name}Op::verify(const std::vector<ir::Type> &inputs, const std::vector<ir::Type> &outputs, const ir::AttributeMap &attributes) {{
  VLOG(4) << "Verifying inputs, outputs and attributes for: {op_name}.";

  // Verify inputs type:
  PADDLE_ENFORCE_EQ(inputs.size(), {inputs_size},
                    phi::errors::PreconditionNotMet("The size %d of inputs must be equal to {inputs_size}.", inputs.size()));
  {inputs_type_check}
  // Verify outputs type:
  PADDLE_ENFORCE_EQ(outputs.size(), {outputs_size},
                    phi::errors::PreconditionNotMet("The size %d of inputs must be equal to {outputs_size}.", outputs.size()));
  {outputs_type_check}
  // Verify if attributes contain attribute name in attributes_name_:
  {attributes_check}
}}
"""
# Example: index=1, standard=paddle::dialect::DenseTensorType
INPUT_TYPE_CHECK_TEMPLATE = """PADDLE_ENFORCE_EQ(inputs[{index}].isa<{standard}>(), true,
                    phi::errors::PreconditionNotMet("Type validation failed for the {index}th input."));
  """
INPUT_VECTORTYPE_CHECK_TEMPLATE = """PADDLE_ENFORCE_EQ(inputs[{index}].isa<ir::VectorType>(), true,
                    phi::errors::PreconditionNotMet("Type validation failed for the {index}th input."));
  PADDLE_ENFORCE_EQ(inputs[{index}].dyn_cast<ir::VectorType>().element_type().isa<{standard}>(), true,
                    phi::errors::PreconditionNotMet("Type validation failed for the {index}th input."));
  """
OUTPUT_TYPE_CHECK_TEMPLATE = """PADDLE_ENFORCE_EQ(outputs[{index}].isa<{standard}>(), true,
                    phi::errors::PreconditionNotMet("Type validation failed for the {index}th output."));
  """
OUTPUT_VECTORTYPE_CHECK_TEMPLATE = """PADDLE_ENFORCE_EQ(outputs[{index}].isa<ir::VectorType>(), true,
                    phi::errors::PreconditionNotMet("Type validation failed for the {index}th output."));
  PADDLE_ENFORCE_EQ(outputs[{index}].dyn_cast<ir::VectorType>().element_type().isa<{standard}>(), true,
                    phi::errors::PreconditionNotMet("Type validation failed for the {index}th output."));
  """
# Example: attribute_name=xxx
ATTRIBUTE_CHECK_TEMPLATE = """PADDLE_ENFORCE_EQ(attributes.at("{attribute_name}").isa<{standard}>(), true,
                    phi::errors::PreconditionNotMet("Type of attribute: {attribute_name} is not right."));
  """
ATTRIBUTE_VECTOR_CHECK_TEMPLATE = """PADDLE_ENFORCE_EQ(attributes.at("{attribute_name}").isa<ir::ArrayAttribute>(), true,
                    phi::errors::PreconditionNotMet("Type of attribute: {attribute_name} is not right."));
  for (size_t i = 0; i < attributes.at("{attribute_name}").dyn_cast<ir::ArrayAttribute>().size(); i++) {{
    PADDLE_ENFORCE_EQ(attributes.at("{attribute_name}").dyn_cast<ir::ArrayAttribute>()[i].isa<{standard}>(), true,
                      phi::errors::PreconditionNotMet("Type of attribute: {attribute_name} is not right."));
  }}
  """


class OpInfo(BaseAPI):
    def __init__(self, op_item_yaml):
        # api: string, api name
        # inputs:
        #     names : [], list of input names
        #     input_info : {input_name : type}
        # attrs:
        #     names : [], list of attribute names
        #     attr_info : { attr_name : (type, default_values)}
        # outputs:
        #     names : [], list of output names
        #     types : [], list of output types
        #     out_size_expr : [], expression for getting size of vector<Tensor>
        super().__init__(op_item_yaml)

        # input/output Paddle Type -> ir Type
        self.type_map = {
            'Tensor': 'paddle::dialect::DenseTensorType',
            'std::vector<Tensor>': 'ir::VectorType<paddle::dialect::DenseTensorType>',
            'const Tensor&': 'paddle::dialect::DenseTensorType',
            'const std::vector<Tensor>&': 'ir::VectorType<paddle::dialect::DenseTensorType>',
            'const paddle::optional<Tensor>&': 'paddle::dialect::DenseTensorType',
            'const paddle::optional<std::vector<Tensor>>&': 'ir::VectorType<paddle::dialect::DenseTensorType>',
            'paddle::optional<int>': 'ir::IntType',
            'paddle::optional<int32_t>': 'ir::Int32_tType',
            'paddle::optional<int64_t>': 'ir::Int64_tType',
            'paddle::optional<float>': 'ir::FloatType',
            'paddle::optional<double>': 'ir::DoubleType',
            'paddle::optional<bool>': 'ir::BoolType',
            'paddle::optional<const Place&>': 'ir::PlaceType',
            'paddle::optional<DataLayout>': 'ir::DataLayoutType',
            'paddle::optional<DataType>': 'ir::DataTypeType',
        }
        # attribute Paddle Type -> ir Attribute
        self.attr_type_map = {
            'const IntArray&': 'paddle::dialect::IntArrayAttribute',
            'const Scalar&': 'paddle::dialect::ScalarAttribute',
            'const std::vector<phi::Scalar>&': 'ir::ArrayAttribute<paddle::dialect::ScalarAttribute>',
            'int': 'ir::IntAttribute',
            'int32_t': 'ir::Int32_tAttribute',
            'int64_t': 'ir::Int64_tAttribute',
            'long': 'ir::LongAttribute',
            'size_t': 'ir::Size_tAttribute',
            'float': 'ir::FloatAttribute',
            'const std::vector<float>&': 'ir::ArrayAttribute<ir::FloatAttribute>',
            'double': 'ir::DoubleAttribute',
            'bool': 'ir::BoolAttribute',
            'const std::vector<bool>&': 'ir::ArrayAttribute<ir::BoolAttribute>',
            'const std::string&': 'ir::StrAttribute',
            'const std::vector<std::string>&': 'ir::ArrayAttribute<ir::StrAttribute>',
            'const Place&': 'paddle::dialect::PlaceAttribute',
            'DataLayout': 'paddle::dialect::DataLayoutAttribute',
            'DataType': 'paddle::dialect::DataTypeAttribute',
            'const std::vector<int64_t>&': 'ir::ArrayAttribute<ir::Int64_tAttribute>',
            'const std::vector<int>&': 'ir::ArrayAttribute<ir::IntAttribute>',
        }

    def get_op_name(self):
        return self.api

    def get_inputs_name_list(self):
        return self.inputs['names']

    def get_inputs_type_list(self, input_type_map={}):
        if len(input_type_map) == 0:
            input_type_map = self.type_map
        inputs_type_list = []
        for name in self.get_inputs_name_list():
            inputs_type_list.append(
                input_type_map[self.inputs['input_info'][name]]
            )
        return inputs_type_list

    def get_outputs_type_list(self, output_type_map={}):
        if len(output_type_map) == 0:
            output_type_map = self.type_map
        outputs_type_list = []
        for type in self.outputs['types']:
            outputs_type_list.append(output_type_map[type])
        return outputs_type_list

    def get_attributes_name_list(self):
        return self.attrs['names']

    def get_attributes_type_list(self, attribute_type_map={}):
        if len(attribute_type_map) == 0:
            attribute_type_map = self.attr_type_map
        attributes_type_list = []
        for name in self.get_attributes_name_list():
            attributes_type_list.append(
                attribute_type_map[self.attrs['attr_info'][name][0]]
            )
        return attributes_type_list

    def get_interfaces_list(self):
        """
        To be continue...
        """
        return []

    def get_traits_list(self):
        """
        To be continue...
        """
        return []


# Generate files
def GenerateOpDefFile(op_yaml_files, header_file, source_file, namespaces):
    # (1) Delete existing old files: pd_op.h.tmp, pd_op.cc.tmp
    if os.path.exists(header_file):
        os.remove(header_file)
    if os.path.exists(source_file):
        os.remove(source_file)

    # (2) Traverse the content of yaml files to generate op definitions
    ops_yaml = []
    for each_op_yaml in op_yaml_files:
        with open(each_op_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                ops_yaml.extend(api_list)
    ops_info = []
    for op_yaml in ops_yaml:
        ops_info.append(OpInfo(op_yaml))

    ops_name_list = []  # all op class name store in this list
    ops_declare_list = []  # all op class declare store in this list
    ops_defined_list = []  # all op class defined store in this list

    for op in ops_info:
        # get op info
        op_name = op.get_op_name()
        op_inputs_type = op.get_inputs_type_list()
        op_outputs_type = op.get_outputs_type_list()
        op_attributes_name = op.get_attributes_name_list()
        op_attributes_type = op.get_attributes_type_list()
        op_interfaces = op.get_interfaces_list()
        op_traits = op.get_traits_list()

        # auto code generate
        if len(op_interfaces) > 0:
            op_interfaces_str = "," + ",".join(op_interfaces)
        else:
            op_interfaces_str = ""
        if len(op_interfaces) > 0:
            op_traits_str = "," + ",".join(op_traits)
        else:
            op_traits_str = ""

        if len(op_attributes_name) == 0:
            op_declare_str = OP_DECLARE_ATTRIBUTES_TEMPLATE.format(
                op_name=op_name,
                interfaces=op_interfaces_str,
                traits=op_traits_str,
                attribute_declare=op_0_attribute_declare_str,
                attribute_num=0,
            )
            op_defined_str = OP_0_ATTRIBUTE_DEFINED_TEMPLATE.format(
                op_name=op_name
            )
        else:
            op_declare_str = OP_DECLARE_ATTRIBUTES_TEMPLATE.format(
                op_name=op_name,
                interfaces=op_interfaces_str,
                traits=op_traits_str,
                attribute_declare=op_n_attribute_declare_str.format(
                    len(op_attributes_name)
                ),
                attribute_num=len(op_attributes_name),
            )
            attribute_names_str = '"' + '", "'.join(op_attributes_name) + '"'
            op_defined_str = OP_N_ATTRIBUTE_DEFINED_TEMPLATE.format(
                op_name=op_name,
                attribute_num=len(op_attributes_name),
                attribute_names=attribute_names_str,
            )

        if len(op_inputs_type) == 0:
            inputs_type_check_str = (
                "// Inputs num is 0, not need to check inputs type."
            )
        else:
            inputs_type_check_str = ""
        for idx in range(len(op_inputs_type)):
            input_type = op_inputs_type[idx]
            if input_type.startswith("ir::VectorType"):
                inner_type = input_type[15:-1]
                inputs_type_check_str += INPUT_VECTORTYPE_CHECK_TEMPLATE.format(
                    index=idx, standard=inner_type
                )
            else:
                inputs_type_check_str += INPUT_TYPE_CHECK_TEMPLATE.format(
                    index=idx, standard=input_type
                )

        if len(op_outputs_type) == 0:
            outputs_type_check_str = (
                "// Outputs num is 0, not need to check outputs type."
            )
        else:
            outputs_type_check_str = ""
        for idx in range(len(op_outputs_type)):
            output_type = op_outputs_type[idx]
            if output_type.startswith("ir::VectorType"):
                inner_type = output_type[15:-1]
                outputs_type_check_str += (
                    OUTPUT_VECTORTYPE_CHECK_TEMPLATE.format(
                        index=idx, standard=inner_type
                    )
                )
            else:
                outputs_type_check_str += OUTPUT_TYPE_CHECK_TEMPLATE.format(
                    index=idx, standard=output_type
                )

        if len(op_attributes_name) == 0:
            attributes_check_str = (
                "// Attributes num is 0, not need to check attributes type."
            )
        else:
            attributes_check_str = ""
        for idx in range(len(op_attributes_name)):
            attribute_name = op_attributes_name[idx]
            attribute_type = op_attributes_type[idx]
            if attribute_type.startswith("ir::ArrayAttribute"):
                inner_attribute = attribute_type[19:-1]
                attributes_check_str += ATTRIBUTE_VECTOR_CHECK_TEMPLATE.format(
                    attribute_name=attribute_name, standard=inner_attribute
                )
            else:
                attributes_check_str += ATTRIBUTE_CHECK_TEMPLATE.format(
                    attribute_name=attribute_name, standard=attribute_type
                )

        op_verify_str = OP_VERIFY_TEMPLATE.format(
            op_name=op_name,
            inputs_size=len(op_inputs_type),
            outputs_size=len(op_outputs_type),
            inputs_type_check=inputs_type_check_str,
            outputs_type_check=outputs_type_check_str,
            attributes_check=attributes_check_str,
        )

        ops_name_list.append(op_name + "Op")
        ops_declare_list.append(op_declare_str)
        ops_defined_list.append(op_defined_str)
        ops_defined_list.append(op_verify_str)

    # (3) Generate head file str
    head_file_str = GET_PD_DIALECT_OP_LIST_TEMPALTE.format(
        ", ".join(ops_name_list)
    )  # Add GET_PD_DIALECT_OP_LIST
    head_file_str += "".join(ops_declare_list)  # Add op class
    for name in reversed(namespaces):
        head_file_str = NAMESPACE_GARD_TEMPLATE.format(
            namespace=name, input=head_file_str
        )  # Add namespaces
    head_file_str = H_FILE_TEMPLATE.format(input=head_file_str)  # Add head

    # (4) Generate source file str
    source_file_str = "".join(ops_defined_list)  # Add op define
    for name in reversed(namespaces):
        source_file_str = NAMESPACE_GARD_TEMPLATE.format(
            namespace=name, input=source_file_str
        )  # Add namespaces
    source_file_str = CC_FILE_TEMPLATE.format(
        h_file=header_file, input=source_file_str
    )  # Add head

    # (5) Generate pd_op.h.tmp, pd_op.cc.tmp
    with open(header_file, 'a') as f:
        f.write(head_file_str)
    with open(source_file, 'a') as f:
        f.write(source_file_str)


# main
if __name__ == "__main__":
    args = ParseArguments()

    op_yaml_files = args.op_yaml.split(",")
    header_file = args.op_header_file
    source_file = args.op_source_file
    if args.namespaces is not None:
        namespaces = args.namespaces.split(",")
    else:
        namespaces = []

    GenerateOpDefFile(op_yaml_files, header_file, source_file, namespaces)
