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

# ===========================
fake_op_name = "FakeOp"
fake_op_h = """
class FakeOp : public ir::Op<FakeOp> {
 public:
  using Op::Op;
  static const char *name() { return "Fake"; }
  static const char **attributes_name_;
  static uint32_t attributes_num() { return 0; }
};
"""
fake_op_cc = """
const char **FakeOp::attributes_name_ = nullptr;
"""
fake_op_name1 = "FakeOp1"
fake_op_h1 = """
class FakeOp1 : public ir::Op<FakeOp1> {
 public:
  using Op::Op;
  static const char *name() { return "Fake1"; }
  static const char *attributes_name_[];
  static uint32_t attributes_num() { return 1; }
};
"""
fake_op_cc1 = """
const char *FakeOp1::attributes_name_[] = {"parameter_name"};
"""
# ===========================


# Script parameter parsing
def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Generate Paddle Dialect OP Definition Files'
    )
    parser.add_argument('--api_yaml', type=str)
    parser.add_argument('--op_header_file', type=str)
    parser.add_argument('--op_source_file', type=str)
    parser.add_argument('--namespaces', type=str)
    args = parser.parse_args()
    return args


# string template for pd_op.h
NAMESPACE_GARD_TEMPLATE = """
namespace {namespace} {{
{input}
}} // namespace {namespace}
"""

H_FILE_TEMPLATE = """
#pragma once

#include "paddle/ir/op_base.h"

{input}
"""

CC_FILE_TEMPLATE = """
#include "{h_file}"

{input}
"""

OP_DECLARE_0_ATTRIBUTES_TEMPLATE = """
class {op_name}Op : public ir::Op<{op_name}{interfaces}{traits}> {{
 public:
  using Op::Op;
  static const char *name() {{ return "{op_name}"; }}
  static const char **attributes_name_;
  static uint32_t attributes_num() {{ return 0; }}
}};
"""
OP_DECLARE_N_ATTRIBUTES_TEMPLATE = """
class {op_name}Op : public ir::Op<{op_name}{interfaces}{traits}> {{
 public:
  using Op::Op;
  static const char *name() {{ return "{op_name}"; }}
  static const char *attributes_name_[];
  static uint32_t attributes_num() {{ return {attribute_num}; }}
}};
"""

OP_DEFINED_0_ATTRIBUTES_TEMPLATE = """
const char **{op_name}Op::attributes_name_ = nullptr;
"""
OP_DEFINED_N_ATTRIBUTES_TEMPLATE = """
const char *{op_name}Op::attributes_name_[] = {{ {attribute_names} }};
"""


# Generate files
def GenerateOpDefFile(api_yaml_files, header_file, source_file, namespaces):
    # (1) Delete existing old files: pd_op.h.tmp, pd_op.cc.tmp
    if os.path.exists(header_file):
        os.remove(header_file)
    if os.path.exists(source_file):
        os.remove(source_file)

    # (2) Traverse the content of yaml files to generate op definitions
    op_name_list = []  # all op class name store in this list
    op_declare_list = []  # all op class declare store in this list
    op_defined_list = []  # all op class defined store in this list
    op_name_list.append(fake_op_name)
    op_name_list.append(fake_op_name1)
    op_declare_list.append(fake_op_h)
    op_declare_list.append(fake_op_h1)
    op_defined_list.append(fake_op_cc)
    op_defined_list.append(fake_op_cc1)

    ops_yaml = []
    for each_api_yaml in api_yaml_files:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                ops_yaml.extend(api_list)
    print("ops_yaml: \n", ops_yaml)

    # (3) Generate head file str
    head_file_str = """\n#define GET_PD_DIALECT_OP_LIST {}\n""".format(
        ", ".join(op_name_list)
    )  # Add GET_PD_DIALECT_OP_LIST
    head_file_str += "".join(op_declare_list)  # Add op class
    for name in reversed(namespaces):
        head_file_str = NAMESPACE_GARD_TEMPLATE.format(
            namespace=name, input=head_file_str
        )  # Add namespaces
    head_file_str = H_FILE_TEMPLATE.format(input=head_file_str)  # Add head

    # (4) Generate source file str
    source_file_str = "".join(op_defined_list)  # Add op define
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

    api_yaml_files = args.api_yaml.split(",")
    header_file = args.op_header_file
    source_file = args.op_source_file
    if args.namespaces is not None:
        namespaces = args.namespaces.split(",")
    else:
        namespaces = []

    GenerateOpDefFile(api_yaml_files, header_file, source_file, namespaces)
