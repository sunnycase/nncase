"""PyNTT runtime entrypoints."""

from .errors import PyNTTArgumentError, PyNTTBackendError, PyNTTError, PyNTTSpecError
from .module import PyNTTModule
from .sharding import LocalShard, ShardedTensorSpec, local_shard_1d, sharded_tensor
from .tensor import allocate_outputs, validate_inputs
from .tuning import select_tuning_parameter
from .workspace import allocate_workspace, materialize_rdata, materialize_rdata_table

__all__ = [
    "LocalShard",
    "PyNTTArgumentError",
    "PyNTTBackendError",
    "PyNTTError",
    "PyNTTModule",
    "PyNTTSpecError",
    "ShardedTensorSpec",
    "allocate_outputs",
    "allocate_workspace",
    "local_shard_1d",
    "materialize_rdata",
    "materialize_rdata_table",
    "select_tuning_parameter",
    "sharded_tensor",
    "validate_inputs",
]
