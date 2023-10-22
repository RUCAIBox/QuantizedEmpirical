import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import PeftConfig, PeftType, transpose
from transformers.activations import ACT2FN
from models.quant import QuantLinearAdapter, QuantLinear


TRANSFORMERS_MODELS_TO_ADAPTER_TYPE_MAPPING = {
    "bloom": {"dense_h_to_4h": "mh_adapter", "dense_4h_to_h": "output_adapter"},
    "gptj": {"fc_in":"mh_adapter", "fc_out":"output_adapter"},
    "gpt_neo": {"c_fc":"mh_adapter", "c_proj":"output_adapter"},
    "llama": {"gate_proj": "mh_adapter", "up_proj":"mh_adapter", "down_proj":"output_adapter"},
    "opt": {"fc1":"mh_adapter", "fc2":"output_adapter"},
    "chatglm": {"dense_h_to_4h": "mh_adapter", "dense_4h_to_h": "output_adapter"},
}

def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb

@dataclass
class GPTQBottleneckConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Bottleneck`].

    Args:
        bottleneck_size (`int`): The size of the bottleneck.
        non_linearity (`str`): The non-linearity to apply to the bottleneck.
        dropout (`float`, optional): The dropout probability of the bottleneck. Default to 0.0
        bias ('str'): Bias type for Bottleneck. Can be 'none', 'all' or 'adapter_only'. Default to 'none'.
        use_parallel_adapter (:obj:`bool`, optional): Whether to use parallel adapter. Defaults to False.
        scaling (:obj:`float` or :obj:`str`, optional):
            Scaling factor to use for scaled addition of adapter outputs as done by He et al. (2021). Can be either a
            constant factor (float) or the string "learned", in which case the scaling factor is learned. Defaults to
            1.0.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Adapter to.
        init_weights (:obj:`str`, optional): Initialization method for the weights of the adapter modules.
            Currently, this can be either "bert" (default) or "mam_adapter".
        modules_to_save (`List[str]`):List of modules apart from Bottleneck adapter layers to be set as trainable
            and saved in the final checkpoint.
    """
    bottleneck_size : int = field(default=256, metadata={"help": "The size of the bottleneck"})
    non_linearity : str = field(default="tanh", metadata={"help": "The non-linearity to apply to the bottleneck"})
    adapter_dropout : float = field(default=0.0, metadata={"help": "The dropout probability of the bottleneck, default to 0.0"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Adapter."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    use_parallel_adapter: bool = field(default=False, metadata={"help": "Whether to use parallel adapter"})
    use_adapterp: bool = field(default=False, metadata={"help": "Whether to use adapterp"})
    scaling: Union[float, str] = 1.0
    bias: str = field(default="none", metadata={"help": "Bias type for Bottleneck. Can be 'none', 'all' or 'adapter_only'"})
    init_weights: str = field(default="bert", metadata={"help": "Initialization method for the weights of the adapter modules."})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from Adapter layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.GPTQBOTTLENECK


class GPTQBottleneckModel(torch.nn.Module):
    """
    Creates Bottleneck adapter model for a pretrained trainsformers model.

    Args:
        model ('transformers.PreTrainedModel'): The pretrained model to be adapted.
        config (`BottleneckConfig`): The configuration of the Bottleneck adapter.
    
    Returns:
        `torch.nn.Module`: The Bottleneck adapter model.
    
    Example::

        >>> from transformers import AutoModelForCausalLM, BottleneckConfig
        >>> from peft import BottleneckModel, BottleneckConfig
        >>> config = BottleneckConfig(
            peft_type="BOTTLNECK", task="CAUSAL_LM", target_modules=["gate_proj", "up_proj", "down_proj"],
            bottleneck_size=256, non_linearity="tanh",
        )
        >>> model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf") 
        >>> bottleneck_model = BottleneckModel(config, model)

    **Attribute**:
        - **model** (`transformers.PreTrainedModel`): The pretrained model to be adapted.
        - **peft_config** (`BottleneckConfig`): The configuration of the Bottleneck adapter.
    """

    def __init__(self, config, model):
        super().__init__()
        self.model = model
        self.peft_config = config
        self._find_and_replace()
        mark_only_adapter_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Adapter with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "bottleneck_size": self.peft_config.bottleneck_size,
            "non_linearity": self.peft_config.non_linearity,
            "adapter_dropout": self.peft_config.adapter_dropout,
            "scaling": self.peft_config.scaling,
            "init_weights": self.peft_config.init_weights,
            "adapter_type": self.peft_config.adapter_type,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                # determine the type of adapter to be used, this will effect the forward pass
                if self.peft_config.use_parallel_adapter:
                    adapter_type = "parallel_adapter"
                else:
                    adapter_type = TRANSFORMERS_MODELS_TO_ADAPTER_TYPE_MAPPING[self.model.config.model_type][target_name]
                kwargs.update({"adapter_type": adapter_type})
                    
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    raise NotImplementedError
                    # kwargs.update(
                    #     {
                    #         "has_fp16_weights": target.state.has_fp16_weights,
                    #         "memory_efficient_backward": target.state.memory_efficient_backward,
                    #         "threshold": target.state.threshold,
                    #         "index": target.index,
                    #     }
                    # )
                    # if adapter_type == "mh_adapter":
                    #     new_module = Linear8bitLt(target.in_features, target.in_features, bias=bias, **kwargs)
                    # elif adapter_type == "output_adapter":
                    #     new_module = Linear8bitLt(target.out_features, target.out_features, bias=bias, **kwargs)
                    # elif adapter_type == "parallel_adapter":
                    #     new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear):
                    if adapter_type == "mh_adapter":
                        new_module = QuantLinearAdapter(self.peft_config.bits, self.peft_config.groupsize, target.in_features, target.in_features, bias=bias, **kwargs)
                    elif adapter_type == "output_adapter":
                        new_module = QuantLinearAdapter(self.peft_config.bits, self.peft_config.groupsize, target.out_features, target.out_features, bias=bias, **kwargs)
                    elif adapter_type == "parallel_adapter":
                        new_module = QuantLinearAdapter(self.peft_config.bits, self.peft_config.groupsize, target.in_features, target.out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)
    
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "adapter_" in name:
                module.to(old_module.weight.device)
        
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        raise NotImplementedError
        # for module in self.model.modules():
        #     if isinstance(module, AdapterLayer):
        #         module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


def mark_only_adapter_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "adapter_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    else:
        raise NotImplementedError