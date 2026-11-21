# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFly speculative-decoding configuration shared by model-specific subclasses.

`SpeculativeConfig` resolves a target model type to its subclass through
`config_class_for` and calls these classmethods while building the draft
`ModelConfig`. The module must stay importable without torch-heavy imports,
since `vllm.config` loads it in every process.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from transformers import PreTrainedConfig

    from vllm.config import ModelConfig, SpeculativeConfig

# Target text model type -> qualified name of its DFly configuration class.
_CONFIG_BY_TARGET_MODEL_TYPE = {
    "hy_v4": "vllm.models.hy_v4.dfly_config.HYV4DFlySpeculatorConfig",
    "hy_v4_flash": "vllm.models.hy_v4_flash.dfly_config.HYV4FlashDFlySpeculatorConfig",
}


def config_class_for(target: "ModelConfig | None") -> type["DFlySpeculatorConfig"]:
    """The DFly configuration class registered for the target's model type."""
    if target is None:
        raise ValueError("DFly requires target_model_config")
    model_type = target.hf_text_config.model_type
    qualname = _CONFIG_BY_TARGET_MODEL_TYPE.get(model_type)
    if qualname is None:
        raise ValueError(
            f"No DFly configuration is registered for target model type {model_type!r}"
        )
    return resolve_obj_by_qualname(qualname)


class DFlySpeculatorConfig:
    """DFly behavior shared by model-specific configuration subclasses."""

    base_method = "dfly"
    speculator_class: str | None = None

    @classmethod
    def default_model(cls, spec: "SpeculativeConfig") -> str:
        if spec.target_model_config is None:
            raise ValueError("DFly requires target_model_config")
        return spec.target_model_config.model

    @classmethod
    def draft_hf_overrides(
        cls, spec: "SpeculativeConfig"
    ) -> Callable[["PreTrainedConfig"], "PreTrainedConfig"]:
        raise NotImplementedError

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        return

    @classmethod
    def resolve_num_speculative_tokens(cls, spec: "SpeculativeConfig") -> None:
        n = spec.num_speculative_tokens
        if n is None:
            block_size = getattr(
                spec.target_model_config.hf_config, "dfly_block_size", None
            )
            if type(block_size) is not int or not 2 <= block_size <= 17:
                raise ValueError(
                    "DFly needs num_speculative_tokens: the checkpoint declares no "
                    "dfly_block_size in [2, 17]"
                )
            # The checkpoint block includes the separate anchor query.
            n = block_size - 1
        if type(n) is not int or not 1 <= n <= 16:
            raise ValueError("DFly requires num_speculative_tokens in [1, 16]")
        spec.num_speculative_tokens = n

    @classmethod
    def apply_options(cls, spec: "SpeculativeConfig") -> None:
        if spec.enable_adaptive_verification:
            raise ValueError("DFly does not support adaptive verification")
        if spec.rejection_sample_method != "standard":
            raise ValueError("DFly requires standard target verification")
        if spec.uses_dynamic_speculative_decoding():
            raise ValueError("DFly does not support dynamic speculative decoding")
