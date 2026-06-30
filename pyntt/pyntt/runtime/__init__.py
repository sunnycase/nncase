"""PyNTT runtime entrypoints."""

from .errors import PyNTTArgumentError, PyNTTBackendError, PyNTTError, PyNTTSpecError
from .interpreter import PyNTTInterpreter
from .module import PyNTTModule
from .sharding import LocalShard, ShardedTensorSpec, local_shard_1d, sharded_tensor
from .tensor import allocate_outputs, resolve_shape_env, validate_inputs
from .tuning import select_tuning_parameter
from .triton import ensure_triton_allocator
from .workspace import RDataCache, WorkspacePool, allocate_workspace, materialize_rdata, materialize_rdata_table

__all__ = [
    "LocalShard",
    "PyNTTArgumentError",
    "PyNTTBackendError",
    "PyNTTError",
    "PyNTTInterpreter",
    "PyNTTModule",
    "PyNTTSpecError",
    "RDataCache",
    "ShardedTensorSpec",
    "WorkspacePool",
    "allocate_outputs",
    "allocate_workspace",
    "ensure_triton_allocator",
    "local_shard_1d",
    "materialize_rdata",
    "materialize_rdata_table",
    "resolve_shape_env",
    "select_tuning_parameter",
    "sharded_tensor",
    "validate_inputs",
]
