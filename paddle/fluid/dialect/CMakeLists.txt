set(PD_DIALECT_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/fluid/dialect")
set(PD_DIALECT_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/fluid/dialect")

# Generate pd_dialect files defining op using op_gen_file
set(op_gen_file ${PADDLE_SOURCE_DIR}/paddle/fluid/dialect/op_gen.py)
set(api_yaml_file ${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/ops.yaml)
set(legacy_api_yaml_file${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/legacy_ops.yaml
)

set(op_header_file ${PD_DIALECT_BINARY_DIR}/pd_op.h)
set(op_source_file ${PD_DIALECT_BINARY_DIR}/pd_op.cc)
set(op_header_file_tmp ${op_header_file}.tmp)
set(op_source_file_tmp ${op_source_file}.tmp)
set(op_namespace paddle,dialect)

add_custom_command(
  OUTPUT ${op_header_file} ${op_source_file}
  COMMAND ${PYTHON_EXECUTABLE} -m pip install pyyaml
  COMMAND
    ${PYTHON_EXECUTABLE} ${op_gen_file} --op_yaml ${api_yaml_file}
    ${legacy_api_yaml_file} --op_header_file ${op_header_file_tmp}
    --op_source_file ${op_source_file_tmp} --namespaces ${op_namespace}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${op_header_file_tmp}
          ${op_header_file}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${op_source_file_tmp}
          ${op_source_file}
  COMMENT "copy_if_different ${op_header_file} ${op_source_file}"
  DEPENDS ${op_gen_file} ${api_yaml_file} ${legacy_api_yaml_file}
  VERBATIM)

# All source files of pd_dialect, except for the source file of op, which is generated in the compilation directory.
file(GLOB PD_DIALECT_SRCS "*.cc")

cc_library(
  pd_dialect
  SRCS ${PD_DIALECT_SRCS}
  DEPS new_ir framework_proto dense_tensor)
target_include_directories(pd_dialect PRIVATE ${PD_DIALECT_BINARY_DIR})
