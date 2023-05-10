set(op_gen_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/paddle_dialect/op_definition/op_gen.py)
set(api_yaml_file ${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/ops.yaml)
set(legacy_api_yaml_file${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/legacy_ops.yaml
)

set(op_header_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/paddle_dialect/op_definition/op.h)
set(op_source_file
    ${PADDLE_SOURCE_DIR}/paddle/fluid/paddle_dialect/op_definition/op.cc)
set(op_header_file_tmp ${op_header_file}.tmp)
set(op_source_file_tmp ${op_source_file}.tmp)

# generate op
add_custom_command(
  OUTPUT ${op_header_file} ${op_source_file}
  COMMAND ${PYTHON_EXECUTABLE} -m pip install pyyaml
  COMMAND
    ${PYTHON_EXECUTABLE} ${op_gen_file} --api_yaml ${api_yaml_file}
    ${legacy_api_yaml_file} --op_header_file ${op_header_file_tmp}
    --op_source_file ${op_source_file_tmp}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${op_source_file_tmp}
          ${op_source_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${op_header_file_tmp}
          ${op_header_file}
  COMMENT "copy_if_different ${op_header_file} ${op_source_file}"
  DEPENDS ${api_yaml_file} ${legacy_api_yaml_file} ${op_gen_file}
  VERBATIM)

cc_library(paddle_dialect_op SRCS ${op_source_file})
