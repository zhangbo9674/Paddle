# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
try:
    from paddle.version import full_version as __version__  # noqa: F401
    from paddle.version import commit as __git_commit__  # noqa: F401
    from paddle.cuda_env import *  # noqa: F403
except ImportError:
    import sys

    sys.stderr.write(
        '''Warning with import paddle: you should not
     import paddle from the source directory; please install paddlepaddle*.whl firstly.'''
    )

# NOTE(SigureMo): We should place the import of base.core before other modules,
# because there are some initialization codes in base/core/__init__.py.
from .base import core  # noqa: F401
from .batch import batch

# Do the *DUPLICATED* monkey-patch for the tensor object.
# We need remove the duplicated code here once we fix
# the illogical implement in the monkey-patch methods later.
from .framework import monkey_patch_variable
from .framework import monkey_patch_math_tensor
from .pir import monkey_patch_opresult, monkey_patch_program

monkey_patch_variable()
monkey_patch_math_tensor()
monkey_patch_opresult()
monkey_patch_program()

from .framework import (
    disable_signal_handler,
    get_flags,
    set_flags,
    disable_static,
    enable_static,
    in_dynamic_mode,
)
from .base.dataset import *  # noqa: F403

from .framework.dtype import (
    iinfo,
    finfo,
    dtype,
    uint8,
    int8,
    int16,
    int32,
    int64,
    float16,
    float32,
    float64,
    bfloat16,
    bool,
    complex64,
    complex128,
)

Tensor = framework.core.eager.Tensor
Tensor.__qualname__ = 'Tensor'

import paddle.distributed.fleet  # noqa: F401
from paddle import (  # noqa: F401
    distributed,
    sysconfig,
    distribution,
    nn,
    optimizer,
    metric,
    regularizer,
    incubate,
    autograd,
    device,
    decomposition,
    jit,
    amp,
    dataset,
    inference,
    io,
    onnx,
    reader,
    static,
    vision,
    audio,
    geometric,
    sparse,
    quantization,
)

from .tensor.attribute import (
    is_complex,
    is_integer,
    rank,
    shape,
    real,
    imag,
    is_floating_point,
)

from .tensor.creation import (
    create_parameter,
    to_tensor,
    diag,
    diag_embed,
    diagflat,
    eye,
    linspace,
    logspace,
    ones,
    ones_like,
    zeros,
    zeros_like,
    arange,
    full,
    full_like,
    triu,
    triu_,
    tril,
    tril_,
    meshgrid,
    empty,
    empty_like,
    assign,
    complex,
    clone,
    tril_indices,
    triu_indices,
    polar,
    geometric_,
    cauchy_,
)

from .tensor.linalg import (  # noqa: F401
    matmul,
    dot,
    norm,
    transpose,
    transpose_,
    dist,
    t,
    t_,
    cdist,
    cross,
    cholesky,
    bmm,
    histogram,
    bincount,
    mv,
    eigvalsh,
)

from .tensor.logic import (  # noqa: F401
    equal,
    equal_,
    greater_equal,
    greater_equal_,
    greater_than,
    greater_than_,
    is_empty,
    less_equal,
    less_equal_,
    less_than,
    less_than_,
    logical_and,
    logical_and_,
    logical_not,
    logical_not_,
    logical_or,
    logical_or_,
    logical_xor,
    logical_xor_,
    bitwise_and,
    bitwise_and_,
    bitwise_not,
    bitwise_not_,
    bitwise_or,
    bitwise_or_,
    bitwise_xor,
    bitwise_xor_,
    not_equal,
    not_equal_,
    allclose,
    isclose,
    equal_all,
    is_tensor,
)


from .tensor.manipulation import (  # noqa: F401
    atleast_1d,
    atleast_2d,
    atleast_3d,
    cast,
    cast_,
    concat,
    broadcast_tensors,
    expand,
    broadcast_to,
    expand_as,
    tile,
    flatten,
    gather,
    gather_nd,
    reshape,
    reshape_,
    flip as reverse,
    scatter,
    scatter_,
    scatter_nd_add,
    scatter_nd,
    shard_index,
    slice,
    crop,
    split,
    vsplit,
    squeeze,
    squeeze_,
    stack,
    strided_slice,
    unique,
    unique_consecutive,
    unsqueeze,
    unsqueeze_,
    unstack,
    flip,
    rot90,
    unbind,
    roll,
    chunk,
    tolist,
    take_along_axis,
    put_along_axis,
    tensordot,
    as_complex,
    as_real,
    moveaxis,
    repeat_interleave,
    index_add,
    index_add_,
    index_put,
    index_put_,
    unflatten,
    as_strided,
    view,
    view_as,
    unfold,
    masked_fill,
    masked_fill_,
    index_fill,
    index_fill_,
    diagonal_scatter,
)

from .tensor.math import (  # noqa: F401
    abs,
    abs_,
    acos,
    acos_,
    asin,
    asin_,
    atan,
    atan_,
    atan2,
    ceil,
    cos,
    cos_,
    tan,
    tan_,
    cosh,
    cosh_,
    cumsum,
    cumsum_,
    cummax,
    cummin,
    cumprod,
    cumprod_,
    logcumsumexp,
    logit,
    logit_,
    exp,
    expm1,
    expm1_,
    floor,
    increment,
    log,
    log_,
    log2_,
    log2,
    log10,
    log10_,
    multiplex,
    pow,
    pow_,
    reciprocal,
    all,
    any,
    round,
    rsqrt,
    scale,
    sign,
    sin,
    sin_,
    sinh,
    sinh_,
    sqrt,
    square,
    square_,
    stanh,
    sum,
    multigammaln,
    multigammaln_,
    nan_to_num,
    nan_to_num_,
    nansum,
    nanmean,
    count_nonzero,
    tanh,
    tanh_,
    add_n,
    max,
    maximum,
    amax,
    min,
    minimum,
    amin,
    mm,
    divide,
    divide_,
    floor_divide,
    floor_divide_,
    remainder,
    remainder_,
    mod,
    mod_,
    floor_mod,
    floor_mod_,
    multiply,
    multiply_,
    renorm,
    renorm_,
    add,
    subtract,
    logsumexp,
    logaddexp,
    inverse,
    log1p,
    log1p_,
    erf,
    erf_,
    addmm,
    addmm_,
    clip,
    trace,
    diagonal,
    kron,
    isfinite,
    isinf,
    isnan,
    prod,
    broadcast_shape,
    conj,
    trunc,
    trunc_,
    digamma,
    digamma_,
    neg,
    neg_,
    lgamma,
    lgamma_,
    acosh,
    acosh_,
    asinh,
    asinh_,
    atanh,
    atanh_,
    lerp,
    erfinv,
    rad2deg,
    deg2rad,
    gcd,
    gcd_,
    lcm,
    lcm_,
    diff,
    angle,
    fmax,
    fmin,
    inner,
    outer,
    heaviside,
    frac,
    frac_,
    sgn,
    take,
    frexp,
    ldexp,
    ldexp_,
    trapezoid,
    cumulative_trapezoid,
    vander,
    nextafter,
    i0,
    i0_,
    i0e,
    i1,
    i1e,
    polygamma,
    polygamma_,
    hypot,
    hypot_,
    combinations,
)

from .tensor.random import (
    bernoulli,
    poisson,
    multinomial,
    standard_normal,
    normal,
    normal_,
    uniform,
    randn,
    rand,
    randint,
    randint_like,
    randperm,
)
from .tensor.search import (
    argmax,
    argmin,
    argsort,
    searchsorted,
    bucketize,
    masked_select,
    topk,
    where,
    where_,
    index_select,
    nonzero,
    sort,
    kthvalue,
    mode,
)

from .tensor.to_string import set_printoptions

from .tensor.einsum import einsum

from .framework import async_save, clear_async_save_task_queue  # noqa: F401

from .framework.random import (
    seed,
    get_cuda_rng_state,
    set_cuda_rng_state,
    get_rng_state,
    set_rng_state,
)
from .framework import (  # noqa: F401
    ParamAttr,
    CPUPlace,
    IPUPlace,
    CUDAPlace,
    CUDAPinnedPlace,
    CustomPlace,
    XPUPlace,
)

from .autograd import (
    grad,
    no_grad,
    enable_grad,
    set_grad_enabled,
    is_grad_enabled,
)
from .framework import (
    save,
    load,
)
from .distributed import DataParallel

from .framework import (
    set_default_dtype,
    get_default_dtype,
)

from .tensor.search import index_sample
from .tensor.stat import (
    mean,
    std,
    var,
    numel,
    median,
    nanmedian,
    quantile,
    nanquantile,
)
from .device import (  # noqa: F401
    get_cudnn_version,
    set_device,
    get_device,
    is_compiled_with_xpu,
    is_compiled_with_ipu,
    is_compiled_with_cinn,
    is_compiled_with_distribute,
    is_compiled_with_cuda,
    is_compiled_with_rocm,
    is_compiled_with_custom_device,
)

# high-level api
from . import (  # noqa: F401
    callbacks,
    hub,
    linalg,
    fft,
    signal,
    _pir_ops,
)
from .hapi import (
    Model,
    summary,
    flops,
)

import paddle.text  # noqa: F401
import paddle.vision  # noqa: F401

from .tensor.random import check_shape
from .nn.initializer.lazy_init import LazyGuard

# CINN has to set a flag to include a lib
if is_compiled_with_cinn():
    import os

    package_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_include_dir = os.path.join(package_dir, "libs")
    cuh_file = os.path.join(runtime_include_dir, "cinn_cuda_runtime_source.cuh")
    if os.path.exists(cuh_file):
        os.environ.setdefault('runtime_include_dir', runtime_include_dir)

disable_static()

from .pir_utils import IrGuard

ir_guard = IrGuard()
ir_guard._switch_to_pir()

__all__ = [
    'iinfo',
    'finfo',
    'dtype',
    'uint8',
    'int8',
    'int16',
    'int32',
    'int64',
    'float16',
    'float32',
    'float64',
    'bfloat16',
    'bool',
    'complex64',
    'complex128',
    'addmm',
    'addmm_',
    'allclose',
    'isclose',
    't',
    't_',
    'add',
    'subtract',
    'diag',
    'diagflat',
    'diag_embed',
    'isnan',
    'scatter_nd_add',
    'unstack',
    'get_default_dtype',
    'save',
    'multinomial',
    'get_cuda_rng_state',
    'get_rng_state',
    'rank',
    'empty_like',
    'eye',
    'cumsum',
    'cumsum_',
    'cummax',
    'cummin',
    'cumprod',
    'cumprod_',
    'logaddexp',
    'logcumsumexp',
    'logit',
    'logit_',
    'LazyGuard',
    'sign',
    'is_empty',
    'equal',
    'equal_',
    'equal_all',
    'is_tensor',
    'is_complex',
    'is_integer',
    'cross',
    'where',
    'where_',
    'log1p',
    'cos',
    'cos_',
    'tan',
    'tan_',
    'mean',
    'mode',
    'mv',
    'in_dynamic_mode',
    'min',
    'amin',
    'any',
    'slice',
    'normal',
    'normal_',
    'logsumexp',
    'full',
    'unsqueeze',
    'unsqueeze_',
    'argmax',
    'Model',
    'summary',
    'flops',
    'sort',
    'searchsorted',
    'bucketize',
    'split',
    'vsplit',
    'logical_and',
    'logical_and_',
    'full_like',
    'less_than',
    'less_than_',
    'kron',
    'clip',
    'Tensor',
    'crop',
    'ParamAttr',
    'stanh',
    'randint',
    'randint_like',
    'assign',
    'gather',
    'scale',
    'zeros',
    'rsqrt',
    'squeeze',
    'squeeze_',
    'to_tensor',
    'gather_nd',
    'isinf',
    'uniform',
    'floor_divide',
    'floor_divide_',
    'remainder',
    'remainder_',
    'floor_mod',
    'floor_mod_',
    'roll',
    'batch',
    'max',
    'amax',
    'logical_or',
    'logical_or_',
    'bitwise_and',
    'bitwise_and_',
    'bitwise_or',
    'bitwise_or_',
    'bitwise_xor',
    'bitwise_xor_',
    'bitwise_not',
    'bitwise_not_',
    'mm',
    'flip',
    'rot90',
    'bincount',
    'histogram',
    'multiplex',
    'CUDAPlace',
    'empty',
    'shape',
    'real',
    'imag',
    'is_floating_point',
    'complex',
    'reciprocal',
    'rand',
    'less_equal',
    'less_equal_',
    'triu',
    'triu_',
    'sin',
    'sin_',
    'dist',
    'cdist',
    'unbind',
    'meshgrid',
    'arange',
    'load',
    'numel',
    'median',
    'nanmedian',
    'quantile',
    'nanquantile',
    'no_grad',
    'enable_grad',
    'set_grad_enabled',
    'is_grad_enabled',
    'mod',
    'mod_',
    'abs',
    'abs_',
    'tril',
    'tril_',
    'pow',
    'pow_',
    'zeros_like',
    'maximum',
    'topk',
    'index_select',
    'CPUPlace',
    'matmul',
    'seed',
    'acos',
    'acos_',
    'logical_xor',
    'exp',
    'expm1',
    'expm1_',
    'bernoulli',
    'poisson',
    'sinh',
    'sinh_',
    'round',
    'DataParallel',
    'argmin',
    'prod',
    'broadcast_shape',
    'conj',
    'neg',
    'neg_',
    'lgamma',
    'lgamma_',
    'lerp',
    'erfinv',
    'inner',
    'outer',
    'square',
    'square_',
    'divide',
    'divide_',
    'ceil',
    'atan',
    'atan_',
    'atan2',
    'rad2deg',
    'deg2rad',
    'gcd',
    'gcd_',
    'lcm',
    'lcm_',
    'expand',
    'broadcast_to',
    'ones_like',
    'index_sample',
    'cast',
    'cast_',
    'grad',
    'all',
    'ones',
    'not_equal',
    'sum',
    'nansum',
    'nanmean',
    'count_nonzero',
    'tile',
    'greater_equal',
    'greater_equal_',
    'isfinite',
    'create_parameter',
    'dot',
    'increment',
    'erf',
    'erf_',
    'bmm',
    'chunk',
    'tolist',
    'tensordot',
    'greater_than',
    'greater_than_',
    'shard_index',
    'argsort',
    'tanh',
    'tanh_',
    'transpose',
    'transpose_',
    'cauchy_',
    'geometric_',
    'randn',
    'strided_slice',
    'unique',
    'unique_consecutive',
    'set_cuda_rng_state',
    'set_rng_state',
    'set_printoptions',
    'std',
    'flatten',
    'asin',
    'multiply',
    'multiply_',
    'disable_static',
    'masked_select',
    'var',
    'trace',
    'enable_static',
    'scatter_nd',
    'set_default_dtype',
    'disable_signal_handler',
    'expand_as',
    'stack',
    'sqrt',
    'randperm',
    'linspace',
    'logspace',
    'reshape',
    'reshape_',
    'atleast_1d',
    'atleast_2d',
    'atleast_3d',
    'reverse',
    'nonzero',
    'CUDAPinnedPlace',
    'logical_not',
    'logical_not_',
    'add_n',
    'minimum',
    'scatter',
    'scatter_',
    'floor',
    'cosh',
    'log',
    'log_',
    'log2',
    'log2_',
    'log10',
    'log10_',
    'concat',
    'check_shape',
    'trunc',
    'trunc_',
    'frac',
    'frac_',
    'digamma',
    'digamma_',
    'standard_normal',
    'diagonal',
    'broadcast_tensors',
    'einsum',
    'set_flags',
    'get_flags',
    'asinh',
    'acosh',
    'atanh',
    'as_complex',
    'as_real',
    'diff',
    'angle',
    'fmax',
    'fmin',
    'moveaxis',
    'repeat_interleave',
    'clone',
    'kthvalue',
    'renorm',
    'renorm_',
    'take_along_axis',
    'put_along_axis',
    'multigammaln',
    'multigammaln_',
    'nan_to_num',
    'nan_to_num_',
    'heaviside',
    'tril_indices',
    'index_add',
    "index_add_",
    "index_put",
    "index_put_",
    'sgn',
    'triu_indices',
    'take',
    'frexp',
    'ldexp',
    'ldexp_',
    'trapezoid',
    'cumulative_trapezoid',
    'polar',
    'vander',
    'unflatten',
    'as_strided',
    'view',
    'view_as',
    'unfold',
    'nextafter',
    'i0',
    'i0_',
    'i0e',
    'i1',
    'i1e',
    'polygamma',
    'polygamma_',
    'masked_fill',
    'masked_fill_',
    'hypot',
    'hypot_',
    'index_fill',
    "index_fill_",
    'diagonal_scatter',
    'combinations',
]
