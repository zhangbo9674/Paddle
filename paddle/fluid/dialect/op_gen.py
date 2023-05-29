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

import yaml

# =====================================
# String Template for h file code gen
# =====================================
NAMESPACE_GARD_TEMPLATE = """namespace {namespace} {{
{input}
}} // namespace {namespace}"""

H_FILE_TEMPLATE = """#ifdef GET_OP_LIST
#undef GET_OP_LIST
{op_declare}
#else

#include <vector>

#inlcude "paddle/ir/builder.h"
#include "paddle/ir/op_base.h"
#include "paddle/ir/operation_utils.h"
#include "paddle/fluid/dialect/utils.h"
#include "paddle/fluid/dialect/pd_interface.h"

{input}
#endif
"""

GET_OP_LIST_TEMPALTE = """{}
"""

OP_DECLARE_TEMPLATE = """
class {op_name} : public ir::Op<{op_name}{interfaces}{traits}> {{
 public:
  using Op::Op;
  static const char *name() {{ return "{dialect_op_name}"; }}
  {attribute_declare}
  static constexpr uint32_t attributes_num = {attribute_num};
  static OpInfoTuple GetOpInfo();
  static void build({build_args});
  static void verify(const std::vector<ir::OpResult> &inputs, const std::vector<ir::Type> &outputs, const ir::AttributeMap &attributes);
{get_inputs_and_outputs}
}};
"""
op_0_attribute_declare_str = (
    "static constexpr const char **attributes_name = nullptr;"
)
op_n_attribute_declare_str = (
    "static const char *attributes_name[{attribute_num}];"
)

OP_GET_INPUT_TEMPLATE = """  ir::OpOperand {input_name}() {{ return operation()->GetOperandByIndex({input_index}); }}
"""
OP_GET_OUTPUT_TEMPLATE = """  ir::OpResult {output_name}() {{ return operation()->GetResultByIndex({output_index}); }}
"""

# =====================================
# String Template for cc file code gen
# =====================================
CC_FILE_TEMPLATE = """#include "{h_file}"
#include "paddle/fluid/dialect/pd_type.h"
#include "paddle/fluid/dialect/pd_attribute.h"
#include "paddle/ir/builtin_attribute.h"
#include "paddle/ir/builtin_type.h"
#include "paddle/ir/ir_context.h"
#include "paddle/phi/core/enforce.h"

{input}
"""

OP_N_ATTRIBUTE_DEFINED_TEMPLATE = """
const char *{op_name}::attributes_name[{attribute_num}] = {{ {attribute_names} }};
"""

# get op info
OP_INFO_TEMPLATE = """
OpInfoTuple {op_name}::GetOpInfo() {{
    std::vector<paddle::dialect::OpInputInfo> inputs = {{ {inputs} }};
    std::vector<paddle::dialect::OpAttributeInfo> attributes = {{ {attributes} }};
    std::vector<paddle::dialect::OpOutputInfo> outputs = {{ {outputs} }};
    return std::make_tuple(inputs, attributes, outputs);
}}
"""
CONSTRUCT_INPUT_INFO_TEMPLATE = (
    """OpInputInfo("{name}", "{typename}", {optional}, {no_need_buffer})"""
)
CONSTRUCT_OUTPUT_INFO_TEMPLATE = (
    """OpOutputInfo("{name}", "{typename}", {optional}, {intermediate})"""
)
CONSTRUCT_ATTRIBUTE_INFO_TEMPLATE = (
    """OpAttributeInfo("{name}", "{typename}", "{data_type}")"""
)

# build
OP_BUILD_TEMPLATE = """
static void build({build_args}) {{
    {build_outputs}
    {build_inputs}
    {build_attributes}
}}
"""
ATTRIBUTE_GEN_ADD_TEMPLATE = """
ir::Attribute {name} = {gen_func};
argument.addAttribute("{name}", {name});
"""

# verify
OP_VERIFY_TEMPLATE = """
void {op_name}::verify(const std::vector<ir::OpResult> &inputs, const std::vector<ir::Type> &outputs, const ir::AttributeMap &attributes) {{
  VLOG(4) << "Verifying inputs, outputs and attributes for: {op_name}.";

  // Verify inputs type:
  PADDLE_ENFORCE_EQ(inputs.size(), {inputs_size},
                    phi::errors::PreconditionNotMet("The size %d of inputs must be equal to {inputs_size}.", inputs.size()));
  {inputs_type_check}
  // Verify outputs type:
  PADDLE_ENFORCE_EQ(outputs.size(), {outputs_size},
                    phi::errors::PreconditionNotMet("The size %d of outputs must be equal to {outputs_size}.", outputs.size()));
  {outputs_type_check}
  // Verify if attributes contain attribute name in attributes_name:
  {attributes_check}
}}
"""
INPUT_TYPE_CHECK_TEMPLATE = """PADDLE_ENFORCE_EQ(inputs[{index}].type().isa<{standard}>(), true,
                    phi::errors::PreconditionNotMet("Type validation failed for the {index}th input."));
  """
INPUT_VECTORTYPE_CHECK_TEMPLATE = """if (inputs[{index}].type().isa<ir::VectorType>()) {{
    for (size_t i = 0; i < inputs[{index}].type().dyn_cast<ir::VectorType>().size(); i++) {{
      PADDLE_ENFORCE_EQ(inputs[{index}].type().dyn_cast<ir::VectorType>()[i].isa<{standard}>(), true,
                        phi::errors::PreconditionNotMet("Type validation failed for the {index}th input."));
    }}
  }} else {{
    PADDLE_ENFORCE_EQ(inputs[{index}].type().isa<{standard}>(), true,
                      phi::errors::PreconditionNotMet("Type validation failed for the {index}th input."));
  }}
  """
INPUT_OPTIONAL_TYPE_CHECK_TEMPLATE = """if (inputs[{index}]) {{
    PADDLE_ENFORCE_EQ(inputs[{index}].type().isa<{standard}>(), true,
                      phi::errors::PreconditionNotMet("Type validation failed for the {index}th input."));
  }}
  """
INPUT_OPTIONAL_VECTORTYPE_CHECK_TEMPLATE = """if (inputs[{index}]) {{
    if (inputs[{index}].type().isa<ir::VectorType>()) {{
      for (size_t i = 0; i < inputs[{index}].type().dyn_cast<ir::VectorType>().size(); i++) {{
        PADDLE_ENFORCE_EQ(inputs[{index}].type().dyn_cast<ir::VectorType>()[i].isa<{standard}>(), true,
                          phi::errors::PreconditionNotMet("Type validation failed for the {index}th input."));
      }}
    }} else {{
      PADDLE_ENFORCE_EQ(inputs[{index}].type().isa<{standard}>(), true,
                        phi::errors::PreconditionNotMet("Type validation failed for the {index}th input."));
    }}
  }}
  """
OUTPUT_TYPE_CHECK_TEMPLATE = """PADDLE_ENFORCE_EQ(outputs[{index}].isa<{standard}>(), true,
                    phi::errors::PreconditionNotMet("Type validation failed for the {index}th output."));
  """
OUTPUT_VECTORTYPE_CHECK_TEMPLATE = """if (outputs[{index}].isa<ir::VectorType>()) {{
    for (size_t i = 0; i < outputs[{index}].dyn_cast<ir::VectorType>().size(); i++) {{
      PADDLE_ENFORCE_EQ(outputs[{index}].dyn_cast<ir::VectorType>()[i].isa<{standard}>(), true,
                        phi::errors::PreconditionNotMet("Type validation failed for the {index}th output."));
    }}
  }} else {{
    PADDLE_ENFORCE_EQ(outputs[{index}].isa<{standard}>(), true,
                      phi::errors::PreconditionNotMet("Type validation failed for the {index}th output."));
  }}
  """
OUTPUT_OPTIONAL_TYPE_CHECK_TEMPLATE = """if (outputs[{index}]) {{
    PADDLE_ENFORCE_EQ(outputs[{index}].isa<{standard}>(), true,
                      phi::errors::PreconditionNotMet("Type validation failed for the {index}th output."));
  }}
  """
OUTPUT_OPTIONAL_VECTORTYPE_CHECK_TEMPLATE = """if (outputs[{index}]) {{
    if (outputs[{index}].isa<ir::VectorType>()) {{
      for (size_t i = 0; i < outputs[{index}].dyn_cast<ir::VectorType>().size(); i++) {{
        PADDLE_ENFORCE_EQ(outputs[{index}].dyn_cast<ir::VectorType>()[i].isa<{standard}>(), true,
                          phi::errors::PreconditionNotMet("Type validation failed for the {index}th output."));
      }}
    }} else {{
      PADDLE_ENFORCE_EQ(outputs[{index}].isa<{standard}>(), true,
                        phi::errors::PreconditionNotMet("Type validation failed for the {index}th output."));
    }}
  }}
  """
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


def to_phi_and_fluid_op_name(op_item):
    # Templat: - op : phi_name (fluid_name)
    names = op_item.split('(')
    if len(names) == 1:
        phi_fluid_name = names[0].strip()
        return phi_fluid_name, phi_fluid_name
    else:
        phi_name = names[0].strip()
        fluid_name = names[1].split(')')[0].strip()
        return phi_name, fluid_name


# =====================================
# Parse Op Compat From Yaml
# =====================================
class OpCompatParser:
    def __init__(self, ops_compat_yaml_file):
        self.ops_compat_yaml_file = ops_compat_yaml_file
        with open(self.ops_compat_yaml_file, "r") as f:
            self.ops_compat = yaml.safe_load(f)

    def get_compat(self, op_name):
        for compat in self.ops_compat:
            phi_name, fluid_name = to_phi_and_fluid_op_name(compat['op'])
            if op_name == phi_name:
                return compat
        return None


# =====================================
# Parse Op Information From Yaml
# =====================================
class OpInfoParser:
    def __init__(self, op_yaml_item, op_compat_item):
        self.op_yaml_item = op_yaml_item
        self.op_compat_item = op_compat_item
        self.op_phi_name = self.parse_op_phi_name()
        # parse inputs
        self.input_name_list = self.parse_input_name_list()
        self.input_type_list = self.parse_input_type_list()
        self.input_optional_list = self.parse_input_optional_list()
        self.input_no_need_buffer_list = self.parse_input_no_need_buffer_list()
        self.cross_check(
            self.input_name_list, self.input_type_list, self.input_optional_list
        )
        # parse outputs
        self.output_name_list = self.parse_output_name_list()
        self.output_type_list = self.parse_output_type_list()
        self.output_size_list = self.parse_output_size_list
        self.output_optional_list = self.parse_output_optional_list()
        self.output_intermediate_list = self.parse_output_intermediate_list()
        self.cross_check(
            self.output_name_list,
            self.output_type_list,
            self.output_optional_list,
        )
        # parse attributes
        self.attr_types_map = {
            'IntArray': ['paddle::dialect::IntArrayAttribute', 'IntArray'],
            'Scalar': ['paddle::dialect::ScalarAttribute', 'Scalar'],
            'Scalar(int)': ['paddle::dialect::ScalarAttribute', 'int'],
            'Scalar(int64_t)': ['paddle::dialect::ScalarAttribute', 'int64_t'],
            'Scalar(float)': ['paddle::dialect::ScalarAttribute', 'float'],
            'Scalar(dobule)': ['paddle::dialect::ScalarAttribute', 'dobule'],
            'Scalar[]': [
                'ir::ArrayAttribute<paddle::dialect::ScalarAttribute>',
                'std::vector<Scalar>',
            ],
            'int': ['ir::Int32_tAttribute', 'int'],
            'int32_t': ['ir::Int32_tAttribute', 'int32_t'],
            'int64_t': ['ir::Int64_tAttribute', 'int64_t'],
            'long': ['ir::LongAttribute', 'long'],
            'size_t': ['ir::Size_tAttribute', 'size_t'],
            'float': ['ir::FloatAttribute', 'float'],
            'float[]': [
                'ir::ArrayAttribute<ir::FloatAttribute>',
                'std::vector<float>',
            ],
            'double': ['ir::DoubleAttribute', 'double'],
            'bool': ['ir::BoolAttribute', 'bool'],
            'bool[]': [
                'ir::ArrayAttribute<ir::BoolAttribute>',
                'std::vecot<bool>',
            ],
            'str': ['ir::StrAttribute', 'std::string'],
            'str[]': ['ir::ArrayAttribute<ir::StrAttribute>', 'std::string'],
            'Place': ['paddle::dialect::PlaceAttribute', 'Place'],
            'DataLayout': [
                'paddle::dialect::DataLayoutAttribute',
                'DataLayout',
            ],
            'DataType': ['paddle::dialect::DataTypeAttribute', 'DataType'],
            'int64_t[]': [
                'ir::ArrayAttribute<ir::Int64_tAttribute>',
                'std::vector<int64_t>',
            ],
            'int[]': [
                'ir::ArrayAttribute<ir::Int32_tAttribute>',
                'std::vector<int>',
            ],
        }
        self.attribute_name_list = self.parse_attribute_name_list()
        self.attribute_type_list = self.parse_attribute_type_list()
        self.attribute_build_arg_type_list = (
            self.parse_attribute_build_arg_type_list()
        )
        self.attribute_data_type_list = self.parse_attribute_data_type_list()
        self.attribute_default_value_list = (
            self.parse_attribute_default_value_list()
        )
        self.cross_check(self.attribute_name_list, self.attribute_type_list)

        # parse infermeta
        self.infer_meta_map = self.parse_infer_meta_map()

    def cross_check(self, name_list, type_list, optional_list=None):
        assert len(name_list) == len(
            type_list
        ), "name list size != type list size."
        if optional_list is not None:
            assert len(type_list) == len(
                optional_list
            ), "type list size != optional list size."

    def parse_op_phi_name(self):
        if self.parse_op_inplace_info() is None:
            return [self.op_yaml_item['name']]
        else:
            if self.op_yaml_item['name'][-1] == "_":
                return [self.op_yaml_item['name']]
            else:
                return [
                    self.op_yaml_item['name'],
                    self.op_yaml_item['name'] + "_",
                ]

    def parse_op_inplace_info(self):
        if 'inplace' in self.op_yaml_item:
            return self.op_yaml_item['inplace']
        return None

    def parse_input_name_list(self):
        name_list = []
        for input_info in self.op_yaml_item['inputs']:
            name_list.append(input_info['name'])
        return name_list

    def parse_input_type_list(self):
        input_types_map = {
            'Tensor': 'paddle::dialect::DenseTensorType',
            'Tensor[]': 'ir::VectorType<paddle::dialect::DenseTensorType>',
        }
        type_list = []
        for input_info in self.op_yaml_item['inputs']:
            assert (
                input_info['typename'] in input_types_map
            ), f"{self.op_phi_name} : Input type error: the input type only support Tensor and Tensor[], but now is {input_info['typename']}."
            type_list.append(input_types_map[input_info['typename']])
        return type_list

    def parse_input_optional_list(self):
        optional_list = []
        for input_info in self.op_yaml_item['inputs']:
            if input_info['optional']:
                optional_list.append("true")
            else:
                optional_list.append("false")
        return optional_list

    def parse_input_no_need_buffer_list(self):
        no_need_buffer_list = []
        for input_info in self.op_yaml_item['inputs']:
            if input_info['no_need_buffer']:
                no_need_buffer_list.append("true")
            else:
                no_need_buffer_list.append("false")
        return no_need_buffer_list

    def parse_output_name_list(self):
        name_list = []
        for output_info in self.op_yaml_item['outputs']:
            name_list.append(output_info['name'])
        return name_list

    def parse_output_type_list(self):
        output_type_map = {
            'Tensor': 'paddle::dialect::DenseTensorType',
            'Tensor[]': 'ir::VectorType<paddle::dialect::DenseTensorType>',
        }
        type_list = []
        for output_info in self.op_yaml_item['outputs']:
            assert (
                output_info['typename'] in output_type_map
            ), f"{self.op_phi_name} : Output type error: the output type only support Tensor and Tensor[], but now is {output_info['typename']}."
            type_list.append(output_type_map[output_info['typename']])
        return type_list

    def parse_output_size_list(self):
        size_list = []
        for output_info in self.op_yaml_item['outputs']:
            if 'size' in output_info:
                size_list.append(output_info['size'])
            else:
                size_list.append(None)
        return size_list

    def parse_output_optional_list(self):
        optional_list = []
        for output_info in self.op_yaml_item['outputs']:
            if 'optional' in output_info:
                if output_info['optional']:
                    optional_list.append("true")
                else:
                    optional_list.append("false")
            else:
                optional_list.append("false")
        return optional_list

    def parse_output_intermediate_list(self):
        intermediate_list = []
        for output_info in self.op_yaml_item['outputs']:
            if 'intermediate' in output_info:
                if output_info['intermediate']:
                    intermediate_list.append("true")
                else:
                    intermediate_list.append("false")
            else:
                intermediate_list.append("false")
        return intermediate_list

    def parse_attribute_name_list(self):
        name_list = []
        for attribute_info in self.op_yaml_item['attrs']:
            name_list.append(attribute_info['name'])
        return name_list

    def parse_attribute_build_arg_type_list(self):
        type_list = []
        for attribute_info in self.op_yaml_item['attrs']:
            assert (
                attribute_info['typename'] in self.attr_types_map
            ), f"{self.op_phi_name} : Attr type error."

            # Scalar & IntArray has data_type
            temp_type = self.attr_types_map[attribute_info['typename']][1]
            if 'Scalar' in temp_type:
                if 'default_value' in attribute_info:
                    temp_type = attribute_info['default_value']
            if 'IntArray' in temp_type:
                if 'default_value' in attribute_info:
                    temp_type = attribute_info['default_value']
            temp_type.replace('Scalar', 'phi::Scalar')
            temp_type.replace('IntArray', 'phi::IntArray')
            temp_type.replace('Place', 'phi::Place')
            temp_type.replace('DataLayout', 'phi::DataLayout')
            temp_type.replace('DataType', 'phi::DataType')
            type_list.append(temp_type)
        return type_list

    def parse_attribute_type_list(self):
        type_list = []
        for attribute_info in self.op_yaml_item['attrs']:
            assert (
                attribute_info['typename'] in self.attr_types_map
            ), f"{self.op_phi_name} : Attr type error."
            type_list.append(self.attr_types_map[attribute_info['typename']][0])
        return type_list

    def parse_attribute_data_type_list(self):
        data_type_list = []
        for attribute_info in self.op_yaml_item['attrs']:
            if 'data_type' in attribute_info:
                data_type_list.append(attribute_info['data_type'])
            else:
                data_type_list.append("")
        return data_type_list

    def parse_attribute_default_value_list(self):
        default_value_list = []
        for attribute_info in self.op_yaml_item['attrs']:
            if 'default_value' in attribute_info:
                default_value_list.append(attribute_info['default_value'])
            else:
                default_value_list.append(None)
        return default_value_list

    def parse_infer_meta_map(self):
        if 'infer_meta' in self.op_yaml_item:
            return self.op_yaml_item['infer_meta']
        else:
            return None


def to_pascal_case(s):
    words = s.split("_")
    if s[-1] == "_":
        return "".join([word.capitalize() for word in words]) + "_"
    else:
        return "".join([word.capitalize() for word in words]) + ""


# =====================================
# Generate Op Definition Files
# =====================================
def GenBuildInputArgsStr(
    op_input_name_list,
    op_attribute_name_list,
    op_attribute_build_arg_type_list,
    op_attribute_default_value_list,
):
    '''
    Example: ir::Builder &builder, ir::OperationArgument &argument, ir::OpResult x_, phi::DataType dtype=phi::DataType::UNDEFINED, phi::Place place={}
    '''
    args_str = "ir::Builder &builder, ir::OperationArgument &argument"
    for input_name in op_input_name_list:
        args_str += ", ir::OpResult " + input_name + "_"
    for attr_idx in range(len(op_attribute_name_list)):
        args_str += (
            ", "
            + op_attribute_build_arg_type_list[attr_idx]
            + " "
            + op_attribute_name_list[attr_idx]
        )
        if op_attribute_default_value_list[attr_idx] is not None:
            args_str += "=" + op_attribute_default_value_list[attr_idx]
    return args_str


def GenBuildInputs(op_input_name_list):
    inputs_args_str = "_, ".join(op_input_name_list)
    build_input_str = """  std::vector<ir::OpResult> inputs = {{{inputs_args_str}}};
      argument.addOperands(inputs.begin(), inputs.end());
    """.format(
        inputs_args_str=inputs_args_str
    )
    return build_input_str


def GenBuildAttributes(op_attribute_name_list, op_attribute_type_list):
    STR_TEMPLATE = """  ir::Attribute attr_{attr_name} = {op_attribute_type}::get(ir::IrContext::Instance(), {attr_name});
      argument.addAttribute("{attr_name}", attribute);
    """
    attr_str = ""
    for idx in range(len(op_attribute_name_list)):
        attr_str += STR_TEMPLATE.format(
            attr_name=op_attribute_name_list[idx],
            op_attribute_type=op_attribute_type_list[idx],
        )
    return attr_str


def GenBuildOutputs(
    op_input_name_list,
    op_input_type_list,
    op_output_name_list,
    op_output_type_list,
    op_output_size_list,
    op_infer_meta_map,
):
    output_str = ""
    CREATE_INPUT_METATENSOR_TEMPLATE = """
    phi::DenseTensor dense_{name};
    dense_{name}.set_meta({name}.Meta());
    phi::MetaTensor meta_{name}(&dense_{name});
    """
    CREATE_INPUT_VEC_METATENSOR_TEMPLATE = """
    std::vector<phi::DenseTensor> vec_dense_{name}({name}.size(), phi::DenseTensor());
    std::vector<phi::MetaTensor> vec_meta_{name};
    for (size_t i=0; i < {name}.size(); i++) {{
      vec_dense_{name}[i].set_meta({name}[i].Meta());
      vec_meta_{name}.push_back(phi::MetaTensor(&vec_dense_{name}[i]));
    }}
    std::vector<const phi::MetaTensor*> vec_meta_{name}_ptr;
    for (size_t i=0; i < vec_meta_{name}.size(); i++) {{
       vec_meta_{name}_ptr.push_back(&vec_meta_{name}[i])
    }}
    """
    CREATE_OUTPUT_METATENSOR_TEMPLATE = """
    phi::DenseTensor dense_{name};
    phi::MetaTensor meta_{name}(&dense_{name});
    """
    CREATE_OUTPUT_VEC_METATENSOR_TEMPLATE = """
    std::vector<phi::DenseTensor> vec_dense_{name}(({output_size}), phi::DenseTensor());
    std::vector<phi::MetaTensor> vec_meta_{name};
    for (size_t i=0; i < ({output_size}); i++) {{
      vec_meta_{name}.push_back(phi::MetaTensor(&vec_dense_{name}[i]));
    }}
    std::vector<const phi::MetaTensor*> vec_meta_{name}_ptr;
    for (size_t i=0; i < vec_meta_{name}.size(); i++) {{
       vec_meta_{name}_ptr.push_back∂(&vec_meta_{name}[i])
    }}
    """
    # Prepar input type
    for idx in range(len(op_input_name_list)):
        # is a vector<Tensor>
        if 'ir::VectorType' in op_input_type_list[idx]:
            output_str += "ir::VectorType {name} = {name}_.type().dyn_cast<ir::VectorType>();\n".format(
                name=op_input_name_list[idx]
            )
        # is a Tensor
        else:
            output_str += "paddle::dialect::DenseTensorType {name} = {name}_type().dyn_cast<paddle::dialect::DenseTensorType>();\n".format(
                name=op_input_name_list[idx]
            )

    # Prepare inputs for infer meta
    infer_meta_args = []
    for idx in range(len(op_infer_meta_map['param'])):
        # is input
        if op_infer_meta_map['param'][idx] in op_input_name_list:
            # is a vector<Tensor>
            if (
                'ir::VectorType'
                in op_input_type_list[
                    op_input_name_list.index(op_infer_meta_map['param'][idx])
                ]
            ):
                output_str += CREATE_INPUT_VEC_METATENSOR_TEMPLATE.format(
                    name=op_infer_meta_map['param'][idx]
                )
                infer_meta_args.append(
                    "vec_meta_{name}_ptr".format(
                        name=op_infer_meta_map['param'][idx]
                    )
                )
            # is a Tensor
            else:
                output_str += CREATE_INPUT_METATENSOR_TEMPLATE.format(
                    name=op_infer_meta_map['param'][idx]
                )
                infer_meta_args.append(
                    "meta_{name}".format(name=op_infer_meta_map['param'][idx])
                )
        # is attribute
        else:
            infer_meta_args.append(op_infer_meta_map['param'][idx])

    # Prepare outputs for infer meta
    for idx in range(len(op_output_name_list)):
        # is a vector<Tensor>
        if 'ir::VectorType' in op_output_type_list[idx]:
            output_str += CREATE_OUTPUT_VEC_METATENSOR_TEMPLATE.format(
                name=op_output_name_list[idx],
                output_size=op_output_size_list[idx],
            )
            infer_meta_args.append(f"vec_meta_{op_output_name_list[idx]}_ptr")
        # is a Tensor
        else:
            output_str += CREATE_OUTPUT_METATENSOR_TEMPLATE.format(
                name=op_output_name_list[idx]
            )
            infer_meta_args.append(f"meta_{op_output_name_list[idx]}")

    # CREATE_INFER_META_FUNC_TEMPLATE = """
    # phi::{func}({args});
    # """
    # # use dense_{name} or vec_dense_{name} to create Outputs type
    # CREATE_OUTPUT_DENSE_TENSOR_TEMPLATE = """
    # ir::Type {name}_dense_tensor_type = paddle::dialect::DenseTensorType::get(ir::IrContext::Instance(), {dtype}, {dims}, {data_layout}, {lod}, {offset});
    # """"

    # CREATE_OUTPUT_VEC_DENSE_TENSOR_TEMPLATE = """
    # ir::Type {name}_vector_type = ir::VectorType::get(ir::IrContext::Instance(), {vector_type});
    # """


def OpGenerator(
    op_yaml_files,
    op_compat_yaml_file,
    namespaces,
    dialect_name,
    op_def_h_file,
    op_def_cc_file,
):
    # (1) Prepare: Delete existing old files: pd_op.h.tmp, pd_op.cc.tmp
    if os.path.exists(op_def_h_file):
        os.remove(op_def_h_file)
    if os.path.exists(op_def_cc_file):
        os.remove(op_def_cc_file)

    # (2) Prepare: Get all op item in all op_yaml_files
    op_compat_parser = OpCompatParser(op_compat_yaml_file)

    op_yaml_items = []
    for yaml_file in op_yaml_files:
        with open(yaml_file, "r") as f:
            ops = yaml.safe_load(f)
            op_yaml_items = op_yaml_items + ops
    op_info_items = []
    for op in op_yaml_items:
        op_info_items.append(
            OpInfoParser(op, op_compat_parser.get_compat(op['name']))
        )

    # (3) CodeGen: Traverse op_info_items and generate
    ops_name_list = []  # all op class name store in this list
    ops_declare_list = []  # all op class declare store in this list
    ops_defined_list = []  # all op class defined store in this list
    for op_info in op_info_items:
        # get op info
        op_input_name_list = op_info.input_name_list
        op_input_type_list = op_info.input_type_list
        op_input_optional_list = op_info.input_optional_list
        op_input_no_need_buffer_list = op_info.input_no_need_buffer_list
        op_output_name_list = op_info.output_name_list
        op_output_type_list = op_info.output_type_list
        op_output_size_list = op_info.output_size_list
        op_output_optional_list = op_info.output_optional_list
        op_output_intermediate_list = op_info.output_intermediate_list
        op_attribute_name_list = op_info.attribute_name_list
        op_attribute_type_list = op_info.attribute_type_list
        op_attribute_data_type_list = op_info.attribute_data_type_list
        op_attribute_build_arg_type_list = op_info.attribute_build_arg_type_list
        op_attribute_default_value_list = op_info.attribute_default_value_list
        op_infer_meta_map = op_info.infer_meta_map
        op_interfaces = ["GetOpInfoInterface"]
        op_traits = []

        # If op has inplace info, we will generate inplace op and non-inplace op.
        for op_name in op_info.op_phi_name:
            op_class_name = to_pascal_case(op_name) + "Op"
            op_dialect_name = dialect_name + "." + op_name

            # gen interface/trait str
            op_interfaces_str = ""
            if len(op_interfaces) > 0:
                op_interfaces_str = "," + ",".join(op_interfaces)
            op_traits_str = ""
            if len(op_traits) > 0:
                op_traits_str = "," + ",".join(op_traits)

            op_get_inputs_outputs_str = ""
            for idx in range(len(op_input_name_list)):
                op_get_inputs_outputs_str += OP_GET_INPUT_TEMPLATE.format(
                    input_name=op_input_name_list[idx],
                    input_index=idx,
                )
            for idx in range(len(op_output_name_list)):
                op_get_inputs_outputs_str += OP_GET_OUTPUT_TEMPLATE.format(
                    output_name=op_output_name_list[idx],
                    output_index=idx,
                )

            # gen op_declare_str/op_defined_str
            if len(op_attribute_name_list) == 0:
                op_declare_str = OP_DECLARE_TEMPLATE.format(
                    op_name=op_class_name,
                    dialect_op_name=op_dialect_name,
                    interfaces=op_interfaces_str,
                    traits=op_traits_str,
                    attribute_declare=op_0_attribute_declare_str,
                    attribute_num=0,
                    get_inputs_and_outputs=op_get_inputs_outputs_str,
                )
                op_defined_str = ""
            else:
                op_declare_str = OP_DECLARE_TEMPLATE.format(
                    op_name=op_class_name,
                    dialect_op_name=op_dialect_name,
                    interfaces=op_interfaces_str,
                    traits=op_traits_str,
                    attribute_declare=op_n_attribute_declare_str.format(
                        attribute_num=len(op_attribute_name_list)
                    ),
                    attribute_num=len(op_attribute_name_list),
                    get_inputs_and_outputs=op_get_inputs_outputs_str,
                )
                attribute_names_str = (
                    '"' + '", "'.join(op_attribute_name_list) + '"'
                )
                op_defined_str = OP_N_ATTRIBUTE_DEFINED_TEMPLATE.format(
                    op_name=op_class_name,
                    attribute_num=len(op_attribute_name_list),
                    attribute_names=attribute_names_str,
                )

            # generate get op info funciton: inputs
            inputs_info_str = ""
            if len(op_input_name_list) > 0:
                input_info_list = []
                for idx in range(len(op_input_name_list)):
                    input_info_list.append(
                        CONSTRUCT_INPUT_INFO_TEMPLATE.format(
                            name=op_input_name_list[idx],
                            typename=op_input_type_list[idx],
                            optional=op_input_optional_list[idx],
                            no_need_buffer=op_input_no_need_buffer_list[idx],
                        )
                    )
                inputs_info_str = ", ".join(input_info_list)

            # generate get op info funciton: outputs
            outputs_info_str = ""
            if len(op_output_name_list) > 0:
                output_info_list = []
                for idx in range(len(op_output_name_list)):
                    output_info_list.append(
                        CONSTRUCT_OUTPUT_INFO_TEMPLATE.format(
                            name=op_output_name_list[idx],
                            typename=op_output_type_list[idx],
                            optional=op_output_optional_list[idx],
                            intermediate=op_output_intermediate_list[idx],
                        )
                    )
                outputs_info_str = ", ".join(output_info_list)

            # generate get op info funciton: attributes
            attribute_info_str = ""
            if len(op_attribute_name_list) > 0:
                attribute_info_list = []
                for idx in range(len(op_attribute_name_list)):
                    attribute_info_list.append(
                        CONSTRUCT_ATTRIBUTE_INFO_TEMPLATE.format(
                            name=op_attribute_name_list[idx],
                            typename=op_attribute_type_list[idx],
                            data_type=op_attribute_data_type_list[idx],
                        )
                    )
                attribute_info_str = ", ".join(attribute_info_list)

            op_info_func_str = OP_INFO_TEMPLATE.format(
                op_name=op_class_name,
                inputs=inputs_info_str,
                attributes=attribute_info_str,
                outputs=outputs_info_str,
            )

            # generate op verify function: inputs_type_check_str
            if len(op_input_type_list) == 0:
                inputs_type_check_str = (
                    "// Inputs num is 0, not need to check inputs type."
                )
            else:
                inputs_type_check_str = ""
            for idx in range(len(op_input_type_list)):
                input_type = op_input_type_list[idx]
                is_optional = op_input_optional_list[idx]
                is_vector = False
                if input_type.startswith("ir::VectorType<"):
                    is_vector = True
                    input_type = input_type[15:-1]
                check_str = ""
                if is_optional == "true":
                    if is_vector:
                        check_str = (
                            INPUT_OPTIONAL_VECTORTYPE_CHECK_TEMPLATE.format(
                                index=idx, standard=input_type
                            )
                        )
                    else:
                        check_str = INPUT_OPTIONAL_TYPE_CHECK_TEMPLATE.format(
                            index=idx, standard=input_type
                        )
                else:
                    if is_vector:
                        check_str = INPUT_VECTORTYPE_CHECK_TEMPLATE.format(
                            index=idx, standard=input_type
                        )
                    else:
                        check_str = INPUT_TYPE_CHECK_TEMPLATE.format(
                            index=idx, standard=input_type
                        )
                inputs_type_check_str += check_str

            # generate op verify function: outputs_type_check_str
            if len(op_output_type_list) == 0:
                outputs_type_check_str = (
                    "// Outputs num is 0, not need to check outputs type."
                )
            else:
                outputs_type_check_str = ""
            for idx in range(len(op_output_type_list)):
                output_type = op_output_type_list[idx]
                is_optional = op_output_optional_list[idx]
                is_vector = False
                if output_type.startswith("ir::VectorType<"):
                    is_vector = True
                    output_type = output_type[15:-1]
                check_str = ""
                if is_optional == "true":
                    if is_vector:
                        check_str = (
                            OUTPUT_OPTIONAL_VECTORTYPE_CHECK_TEMPLATE.format(
                                index=idx, standard=output_type
                            )
                        )
                    else:
                        check_str = OUTPUT_OPTIONAL_TYPE_CHECK_TEMPLATE.format(
                            index=idx, standard=output_type
                        )
                else:
                    if is_vector:
                        check_str = OUTPUT_VECTORTYPE_CHECK_TEMPLATE.format(
                            index=idx, standard=output_type
                        )
                    else:
                        check_str = OUTPUT_TYPE_CHECK_TEMPLATE.format(
                            index=idx, standard=output_type
                        )
                outputs_type_check_str += check_str

            # generate op verify function: attributes_check_str
            if len(op_attribute_name_list) == 0:
                attributes_check_str = (
                    "// Attributes num is 0, not need to check attributes type."
                )
            else:
                attributes_check_str = ""
            for idx in range(len(op_attribute_name_list)):
                attribute_name = op_attribute_name_list[idx]
                attribute_type = op_attribute_type_list[idx]
                if attribute_type.startswith("ir::ArrayAttribute<"):
                    attribute_type = attribute_type[19:-1]
                    attributes_check_str += (
                        ATTRIBUTE_VECTOR_CHECK_TEMPLATE.format(
                            attribute_name=attribute_name,
                            standard=attribute_type,
                        )
                    )
                else:
                    attributes_check_str += ATTRIBUTE_CHECK_TEMPLATE.format(
                        attribute_name=attribute_name, standard=attribute_type
                    )

            # generate op verify function
            op_verify_str = OP_VERIFY_TEMPLATE.format(
                op_name=op_class_name,
                inputs_size=len(op_input_type_list),
                outputs_size=len(op_output_type_list),
                inputs_type_check=inputs_type_check_str,
                outputs_type_check=outputs_type_check_str,
                attributes_check=attributes_check_str,
            )

            ops_name_list.append(op_class_name)
            ops_declare_list.append(op_declare_str)
            ops_defined_list.append(op_defined_str)
            ops_defined_list.append(op_info_func_str)
            ops_defined_list.append(op_verify_str)

    # (4) Generate head file str
    op_namespaces_prev = ""
    for name in namespaces:
        op_namespaces_prev += name + "::"
    ops_name_with_namespace_list = []
    for name in ops_name_list:
        ops_name_with_namespace_list.append(op_namespaces_prev + name)
    op_list_str = GET_OP_LIST_TEMPALTE.format(
        ", ".join(ops_name_with_namespace_list)
    )  # Add GET_OP_LIST
    head_file_str = ""
    head_file_str += "".join(ops_declare_list)  # Add op class
    for name in reversed(namespaces):
        head_file_str = NAMESPACE_GARD_TEMPLATE.format(
            namespace=name, input=head_file_str
        )  # Add namespaces
    head_file_str = H_FILE_TEMPLATE.format(
        op_declare=op_list_str, input=head_file_str
    )  # Add head

    # (5) Generate source file str
    source_file_str = "".join(ops_defined_list)  # Add op define
    for name in reversed(namespaces):
        source_file_str = NAMESPACE_GARD_TEMPLATE.format(
            namespace=name, input=source_file_str
        )  # Add namespaces
    source_file_str = CC_FILE_TEMPLATE.format(
        h_file=op_def_h_file, input=source_file_str
    )  # Add head

    # (5) Generate pd_op.h.tmp, pd_op.cc.tmp
    with open(op_def_h_file, 'a') as f:
        f.write(head_file_str)
    with open(op_def_cc_file, 'a') as f:
        f.write(source_file_str)


# =====================================
# Script parameter parsing
# =====================================
def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Generate Dialect OP Definition Files By Yaml'
    )
    parser.add_argument('--op_yaml_files', type=str)
    parser.add_argument('--op_compat_yaml_file', type=str)
    parser.add_argument('--namespaces', type=str)
    parser.add_argument('--dialect_name', type=str)
    parser.add_argument('--op_def_h_file', type=str)
    parser.add_argument('--op_def_cc_file', type=str)
    return parser.parse_args()


# =====================================
# Main
# =====================================
if __name__ == "__main__":
    # parse arguments
    args = ParseArguments()
    op_yaml_files = args.op_yaml_files.split(",")
    op_compat_yaml_file = args.op_compat_yaml_file
    namespaces = []
    if args.namespaces is not None:
        namespaces = args.namespaces.split(",")
    dialect_name = args.dialect_name
    op_def_h_file = args.op_def_h_file
    op_def_cc_file = args.op_def_cc_file

    # auto code generate
    OpGenerator(
        op_yaml_files,
        op_compat_yaml_file,
        namespaces,
        dialect_name,
        op_def_h_file,
        op_def_cc_file,
    )
