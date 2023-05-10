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


def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Generate Paddle Dialect OP Definition Files'
    )
    parser.add_argument('--api_yaml', type=str)
    parser.add_argument('--op_header_file', type=str)
    parser.add_argument('--op_source_file', type=str)
    args = parser.parse_args()
    return args


def GenerateOpDefFile(header_file, source_file):
    if os.path.exists(header_file):
        os.remove(header_file)

    if os.path.exists(source_file):
        os.remove(source_file)

    h_file_contents = "#include <iostream>"

    cc_file_contents = " "

    with open(header_file, 'a') as f:
        f.write(h_file_contents)

    with open(source_file, 'a') as f:
        f.write(cc_file_contents)


if __name__ == "__main__":
    print("Generate Paddle Dialect OP Definition files ...")

    args = ParseArguments()
    api_yaml = args.api_yaml.split(",")
    header_file = args.op_header_file
    source_file = args.op_source_file

    GenerateOpDefFile(header_file, source_file)
