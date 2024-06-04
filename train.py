import logging
import multiprocessing
from os import path
import time
from typing import Any, Literal, Optional, Sequence

import einshape as es
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from paxml import checkpoints
from paxml import tasks_lib
from praxis import base_hyperparams
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import normalizations
from praxis.layers import transformers
import patched_decoder
from utilsforecast.processing import make_future_dataframe

instantiate = base_hyperparams.instantiate
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor

num_devices = jax.local_device_count('cpu')

horizon_len = 128
output_patch_len = 32
context_len = 512
backend = "cpu"



model_p = pax_fiddle.Config(
        patched_decoder.PatchedTimeSeriesDecoder,
        name="patched_decoder",
        horizon_len=128,
        patch_len=32,
        model_dims=1280,
        hidden_dims=1280,
        residual_block_tpl=pax_fiddle.Config(patched_decoder.ResidualBlock),
        quantiles=patched_decoder.DEFAULT_QUANTILES,
        use_freq=True,
        stacked_transformer_params_tpl=pax_fiddle.Config(
            transformers.StackedTransformer,
            num_heads=16,
            num_layers=20,
            transformer_layer_params_tpl=pax_fiddle.Config(
                transformers.Transformer,
                ln_tpl=pax_fiddle.Config(
                    normalizations.RmsNorm,
                ),
            ),
        ),
    )


# make forward for model_p

# def forward(model_p: pax_fiddle.Config, inputs: JTensor, input_padding: JTensor, freq: Optional[JTensor] = None) -> NestedMap:
#     """Forward pass for the model."""
#     model = base_hyperparams.instantiate(model_p)
#     def _decode(inputs):
#       assert self._model is not None
#       assert self._train_state is not None
#       return model.apply(
#           self._train_state.mdl_vars,
#           inputs,
#           horizon_len=self.horizon_len,
#           output_patch_len=self.output_patch_len,
#           max_len=self.context_len,
#           rngs={
#               base_layer.PARAMS: self._key1,
#               base_layer.RANDOM: self._key2,
#           },
#           method=self._model.decode,
#       )

#     _pmapped_decode = jax.pmap(
#         _decode,
#         axis_name="batch",
#         devices=jax.devices("cpu"),
#         backend="cpu",
#         axis_size=1,
#     )

t =  {
        "input_ts": jnp.zeros(
            (
                8,
                512 + 128,
            ),
            dtype=jnp.float32,
        ),
        "input_padding": jnp.zeros(
            (
                8,
                512 + 128,
            ),
            dtype=jnp.float32,
        ),
        "freq": jnp.zeros(
            (
                8,
                1,
            ),
            dtype=jnp.int32,
        ),
    }

_model = instantiate(model_p)
var_weight_hparams = _model.abstract_init_with_metadata(
    t, do_eval=True
)

mesh_shape = [1, 1, 1]
mesh_name = ["replica", "data", "mdl"]
train_state_partition_specs = tasks_lib.create_state_partition_specs(
    var_weight_hparams,
    mesh_shape=mesh_shape,
    mesh_axis_names=mesh_name,
    discard_opt_states=True,
    learners=None,
)
train_state_local_shapes = tasks_lib.create_state_unpadded_shapes(
    var_weight_hparams,
    discard_opt_states=True,
    learners=None,
)
repo_id = "google/timesfm-1.0-200m"
checkpoint_type = checkpoints.CheckpointType.FLAX


_train_state = checkpoints.restore_checkpoint(
    train_state_local_shapes,
    checkpoint_dir = path.join(snapshot_download(repo_id), "checkpoints"),
    checkpoint_type=checkpoint_type,
    state_specs=train_state_partition_specs,
    step=None,
)

_key1, _key2 = jax.random.split(jax.random.PRNGKey(42))
# Initialize and jit the decode fn.
def _decode(inputs):
    assert _model is not None
    assert _train_state is not None
    return _model.apply(
        _train_state.mdl_vars,
        inputs,
        horizon_len=horizon_len,
        output_patch_len=output_patch_len,
        max_len=context_len,
        rngs={
            base_layer.PARAMS: _key1,
            base_layer.RANDOM: _key2,
        },
        method=_model.decode,
    )

_pmapped_decode = jax.pmap(
    _decode,
    axis_name="batch",
    devices=jax.devices(backend),
    backend=backend,
    axis_size=num_devices,
)

_eval_context = base_layer.JaxContext.HParams(do_eval=True)


per_core_batch_size = 1
with base_layer.JaxContext.new_context(hparams=_eval_context):
      _ = _pmapped_decode(
          NestedMap({
              "input_ts": jnp.zeros(
                  (
                      num_devices,
                      per_core_batch_size,
                      context_len,
                  ),
                  dtype=jnp.float32,
              ),
              "input_padding": jnp.zeros(
                  (
                      num_devices,
                      per_core_batch_size,
                      context_len + horizon_len,
                  ),
                  dtype=jnp.float32,
              ),
              "date_features": None,
              "freq": jnp.zeros(
                  (num_devices, per_core_batch_size, 1),
                  dtype=jnp.int32,
              ),
          })
      )

print(_)