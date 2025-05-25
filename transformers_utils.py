import os
import collections
import re
import shutil
import tempfile
import gc
import itertools
from copy import deepcopy

import torch
import torch.nn as nn

from transformers.utils import logging
from transformers.utils.quantization_config import QuantizationMethod

from transformers.pytorch_utils import id_tensor_storage
from transformers.modeling_utils import (
    set_initialized_submodules, 
    expand_device_map, 
    _load_state_dict_into_model, 
    get_disk_only_shard_files, 
    load_state_dict, 
    _load_state_dict_into_meta_model,
    check_support_param_buffer_assignment,
    is_fsdp_enabled,
    is_local_dist_rank_0,
)
from accelerate.utils import (
    find_tied_parameters, 
    set_module_tensor_to_device, 
    load_offloaded_weights, 
    save_offload_index
)
from transformers.integrations import is_deepspeed_zero3_enabled

logger = logging.get_logger(__name__)

## transformers=4.46.0

## Don't convert Inserted Adapter to Low Bit Linear
def get_keys_to_not_convert(model):
    r"""
    An utility function to get the key of the module to keep in full precision if any For example for CausalLM modules
    we may want to keep the lm_head in full precision for numerical stability reasons. For other architectures, we want
    to keep the tied weights of the model. The function will return a list of the keys of the modules to not convert in
    int8.

    Parameters:
    model (`torch.nn.Module`):
        Input model
    """
    # Create a copy of the model and tie the weights, then
    # check if it contains tied weights
    tied_model = deepcopy(model)  # this has 0 cost since it is done inside `init_empty_weights` context manager`
    tied_model.tie_weights()

    tied_params = find_tied_parameters(tied_model)
    # For compatibility with Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = sum(list(tied_params.values()), []) + list(tied_params.keys())
    else:
        tied_keys = sum(tied_params, [])
    has_tied_params = len(tied_keys) > 0

    add_module = []
    for n, p in model.named_modules():
        if 'adapter' in n:
            add_module.append(n)

    # If there is not tied weights, we want to keep the lm_head（output_embedding) in full precision
    if not has_tied_params:
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            list_last_module = [name for name, module in model.named_modules() if id(module) == id(output_emb)]
            list_module = list_last_module + add_module
            return list_module

    # otherwise, no tied weights, no output embedding defined, simply keep the last module in full precision
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]
    # add last module together with tied weights
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = list(set(tied_keys)) + list(intersection) + add_module

    # remove ".weight" from the keys
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    return filtered_module_names


@classmethod
def _load_pretrained_model(
    cls,
    model,
    state_dict,
    loaded_keys,
    resolved_archive_file,
    pretrained_model_name_or_path,
    ignore_mismatched_sizes=False,
    sharded_metadata=None,
    _fast_init=True,
    low_cpu_mem_usage=False,
    device_map=None,
    offload_folder=None,
    offload_state_dict=None,
    dtype=None,
    hf_quantizer=None,
    keep_in_fp32_modules=None,
    gguf_path=None,
    weights_only=True,
):
    is_safetensors = False
    is_quantized = hf_quantizer is not None
    state_dict_folder = None
    state_dict_index = None

    if device_map is not None and "disk" in device_map.values():
        archive_file = (
            resolved_archive_file[0] if isinstance(resolved_archive_file, (list, tuple)) else resolved_archive_file
        )
        is_safetensors = archive_file.endswith(".safetensors")
        if offload_folder is None and not is_safetensors:
            raise ValueError(
                "The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder`"
                " for them. Alternatively, make sure you have `safetensors` installed if the model you are using"
                " offers the weights in this format."
            )
        if offload_folder is not None:
            os.makedirs(offload_folder, exist_ok=True)
        if offload_state_dict is None:
            offload_state_dict = True

    is_sharded_safetensors = is_safetensors and sharded_metadata is not None

    # tie the model weights before retrieving the state_dict
    model.tie_weights()

    # Retrieve missing & unexpected_keys
    model_state_dict = model.state_dict()
    expected_keys = list(model_state_dict.keys())
    prefix = model.base_model_prefix

    if hf_quantizer is not None:
        expected_keys = hf_quantizer.update_expected_keys(model, expected_keys, loaded_keys)

    def _fix_key(key):
        if "beta" in key:
            return key.replace("beta", "bias")
        if "gamma" in key:
            return key.replace("gamma", "weight")

        # to avoid logging parametrized weight norm renaming
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            if "weight_g" in key:
                return key.replace("weight_g", "parametrizations.weight.original0")
            if "weight_v" in key:
                return key.replace("weight_v", "parametrizations.weight.original1")
        else:
            if "parametrizations.weight.original0" in key:
                return key.replace("parametrizations.weight.original0", "weight_g")
            if "parametrizations.weight.original1" in key:
                return key.replace("parametrizations.weight.original1", "weight_v")
        return key

    original_loaded_keys = loaded_keys
    loaded_keys = [_fix_key(key) for key in loaded_keys]

    if len(prefix) > 0:
        has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
        expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
    else:
        has_prefix_module = False
        expects_prefix_module = False

    # key re-naming operations are never done on the keys
    # that are loaded, but always on the keys of the newly initialized model
    remove_prefix_from_model = not has_prefix_module and expects_prefix_module
    add_prefix_to_model = has_prefix_module and not expects_prefix_module

    if remove_prefix_from_model:
        _prefix = f"{prefix}."
        expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
        expected_keys = [s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys]
    elif add_prefix_to_model:
        expected_keys = [".".join([prefix, s]) for s in expected_keys]

    missing_keys = sorted(set(expected_keys) - set(loaded_keys))
    unexpected_keys = set(loaded_keys) - set(expected_keys)

    # Remove nonpersistent buffers from unexpected keys: they are not in the state dict but will be in the model
    # buffers
    model_buffers = {n for n, _ in model.named_buffers()}
    if remove_prefix_from_model:
        model_buffers = {key[len(_prefix) :] if key.startswith(_prefix) else key for key in model_buffers}
    elif add_prefix_to_model:
        model_buffers = {".".join([prefix, key]) for key in model_buffers}
    unexpected_keys = sorted(unexpected_keys - model_buffers)

    model.tie_weights()
    if device_map is None and not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
        ptrs = collections.defaultdict(list)
        for name, tensor in model.state_dict().items():
            id_tensor = id_tensor_storage(tensor)
            ptrs[id_tensor].append(name)

        # These are all the pointers of shared tensors.
        tied_params = [names for _, names in ptrs.items() if len(names) > 1]
    else:
        # id function doesn't work for meta tensor so we need this function
        tied_params = find_tied_parameters(model)

    for group in tied_params:
        if remove_prefix_from_model:
            group = [key[len(_prefix) :] if key.startswith(_prefix) else key for key in group]
        elif add_prefix_to_model:
            group = [".".join([prefix, key]) for key in group]
        missing_in_group = [k for k in missing_keys if k in group]
        if len(missing_in_group) > 0 and len(missing_in_group) < len(group):
            missing_keys = [k for k in missing_keys if k not in missing_in_group]

    # Some models may have keys that are not in the state by design, removing them before needlessly warning
    # the user.
    if cls._keys_to_ignore_on_load_missing is not None:
        for pat in cls._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

    if cls._keys_to_ignore_on_load_unexpected is not None:
        for pat in cls._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
    if hf_quantizer is not None:
        missing_keys = hf_quantizer.update_missing_keys(model, missing_keys, prefix)

    # retrieve weights on meta device and put them back on CPU.
    # This is not ideal in terms of memory, but if we don't do that not, we can't initialize them in the next step
    if low_cpu_mem_usage:
        for key in missing_keys:
            if key in list(model_state_dict.keys()):
                key = key
            elif f"{prefix}.{key}" in list(model_state_dict.keys()):
                key = f"{prefix}.{key}"
            elif key.startswith(prefix) and ".".join(key.split(".")[1:]) in list(model_state_dict.keys()):
                key = ".".join(key.split(".")[1:])
            param = model_state_dict[key]

            # upcast in fp32 if any
            target_dtype = dtype
            if (
                keep_in_fp32_modules is not None
                and dtype == torch.float16
                and any(
                    module_to_keep_in_fp32 in key.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules
                )
            ):
                target_dtype = torch.float32

            if param.device == torch.device("meta"):
                value = torch.empty(*param.size(), dtype=target_dtype)
                if (
                    not is_quantized
                    or (getattr(hf_quantizer, "requires_parameters_quantization", False))
                    or not hf_quantizer.check_quantized_param(
                        model, param_value=value, param_name=key, state_dict={}
                    )
                ):
                    set_module_tensor_to_device(model, key, "cpu", value)
                else:
                    hf_quantizer.create_quantized_param(model, value, key, "cpu", state_dict, unexpected_keys)

    # retrieve uninitialized modules and initialize before maybe overriding that with the pretrained weights.
    if _fast_init:
        if not ignore_mismatched_sizes:
            if remove_prefix_from_model:
                _loaded_keys = [f"{prefix}.{k}" for k in loaded_keys]
            elif add_prefix_to_model:
                _loaded_keys = [k[len(prefix) + 1 :] for k in loaded_keys]
            else:
                _loaded_keys = loaded_keys
            not_initialized_submodules = set_initialized_submodules(model, _loaded_keys)
            # If we're about to tie the output embeds to the input embeds we don't need to init them
            if hasattr(model.config, "tie_word_embeddings") and model.config.tie_word_embeddings:
                output_embeddings = model.get_output_embeddings()
                if output_embeddings is not None:
                    # Still need to initialize if there is a bias term since biases are not tied.
                    if not hasattr(output_embeddings, "bias") or output_embeddings.bias is None:
                        output_embeddings._is_hf_initialized = True
        else:
            not_initialized_submodules = dict(model.named_modules())
        # This will only initialize submodules that are not marked as initialized by the line above.
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            not_initialized_parameters = list(
                set(
                    itertools.chain.from_iterable(
                        submodule.parameters(recurse=False) for submodule in not_initialized_submodules.values()
                    )
                )
            )
            with deepspeed.zero.GatheredParameters(not_initialized_parameters, modifier_rank=0):
                model.apply(model._initialize_weights)
        else:
            model.apply(model._initialize_weights)

    # Set some modules to fp32 if any
    if keep_in_fp32_modules is not None:
        for name, param in model.named_parameters():
            if any(module_to_keep_in_fp32 in name.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules):
                # param = param.to(torch.float32) does not work here as only in the local scope.
                param.data = param.data.to(torch.float32)

    # Make sure we are able to load base models as well as derived models (with heads)
    start_prefix = ""
    model_to_load = model
    if len(cls.base_model_prefix) > 0 and not hasattr(model, cls.base_model_prefix) and has_prefix_module:
        start_prefix = cls.base_model_prefix + "."
    if len(cls.base_model_prefix) > 0 and hasattr(model, cls.base_model_prefix) and not has_prefix_module:
        model_to_load = getattr(model, cls.base_model_prefix)
        base_model_expected_keys = list(model_to_load.state_dict().keys())
        if any(key in expected_keys_not_prefixed and key not in base_model_expected_keys for key in loaded_keys):
            raise ValueError(
                "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                "properly saved?"
            )
        if device_map is not None:
            device_map = {k.replace(f"{cls.base_model_prefix}.", ""): v for k, v in device_map.items()}

    def _find_mismatched_keys(
        state_dict,
        model_state_dict,
        loaded_keys,
        add_prefix_to_model,
        remove_prefix_from_model,
        ignore_mismatched_sizes,
    ):
        mismatched_keys = []
        if ignore_mismatched_sizes:
            for checkpoint_key in loaded_keys:
                # If the checkpoint is sharded, we may not have the key here.
                if checkpoint_key not in state_dict:
                    continue
                model_key = checkpoint_key
                if remove_prefix_from_model:
                    # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                    model_key = f"{prefix}.{checkpoint_key}"
                elif add_prefix_to_model:
                    # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                    model_key = ".".join(checkpoint_key.split(".")[1:])

                if (
                    model_key in model_state_dict
                    and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                ):
                    if (
                        state_dict[checkpoint_key].shape[-1] == 1
                        and state_dict[checkpoint_key].numel() * 2 == model_state_dict[model_key].numel()
                    ):
                        # This skips size mismatches for 4-bit weights. Two 4-bit values share an 8-bit container, causing size differences.
                        # Without matching with module type or paramter type it seems like a practical way to detect valid 4bit weights.
                        pass
                    else:
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
        return mismatched_keys

    if resolved_archive_file is not None:
        folder = os.path.sep.join(resolved_archive_file[0].split(os.path.sep)[:-1])
    else:
        folder = None
    if device_map is not None and is_safetensors:
        param_device_map = expand_device_map(device_map, original_loaded_keys, start_prefix)
        str_dtype = str(dtype).replace("torch.", "") if dtype is not None else "float32"
        if sharded_metadata is None:
            archive_file = (
                resolved_archive_file[0]
                if isinstance(resolved_archive_file, (list, tuple))
                else resolved_archive_file
            )
            weight_map = {p: archive_file for p in original_loaded_keys}
        else:
            weight_map = {p: os.path.join(folder, f) for p, f in sharded_metadata["weight_map"].items()}
        offload_index = {
            p[len(start_prefix) :]: {"safetensors_file": f, "weight_name": p, "dtype": str_dtype}
            for p, f in weight_map.items()
            if p.startswith(start_prefix) and param_device_map[p[len(start_prefix) :]] == "disk"
        }
    else:
        offload_index = None

    if state_dict is not None:
        # Whole checkpoint
        mismatched_keys = _find_mismatched_keys(
            state_dict,
            model_state_dict,
            original_loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        )

        # For GGUF models `state_dict` is never set to None as the state dict is always small
        if gguf_path:
            error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
                model_to_load,
                state_dict,
                start_prefix,
                expected_keys,
                device_map=device_map,
                offload_folder=offload_folder,
                offload_index=offload_index,
                state_dict_folder=state_dict_folder,
                state_dict_index=state_dict_index,
                dtype=dtype,
                hf_quantizer=hf_quantizer,
                is_safetensors=is_safetensors,
                keep_in_fp32_modules=keep_in_fp32_modules,
                unexpected_keys=unexpected_keys,
            )
        else:
            # Sharded checkpoint or whole but low_cpu_mem_usage==True
            assign_to_params_buffers = check_support_param_buffer_assignment(
                model_to_load, state_dict, start_prefix
            )
            error_msgs = _load_state_dict_into_model(
                model_to_load, state_dict, start_prefix, assign_to_params_buffers
            )

    else:
        # This should always be a list but, just to be sure.
        if not isinstance(resolved_archive_file, list):
            resolved_archive_file = [resolved_archive_file]

        error_msgs = []
        mismatched_keys = []
        if not is_safetensors:
            offload_index = {} if device_map is not None and "disk" in device_map.values() else None
        if offload_state_dict:
            state_dict_folder = tempfile.mkdtemp()
            state_dict_index = {}
        else:
            state_dict_folder = None
            state_dict_index = None

        if is_sharded_safetensors:
            disk_only_shard_files = get_disk_only_shard_files(
                device_map, sharded_metadata=sharded_metadata, start_prefix=start_prefix
            )
            disk_only_shard_files = [os.path.join(folder, f) for f in disk_only_shard_files]
        else:
            disk_only_shard_files = []

        if len(resolved_archive_file) > 1:
            resolved_archive_file = logging.tqdm(resolved_archive_file, desc="Loading checkpoint shards")
        assign_to_params_buffers = None
        for shard_file in resolved_archive_file:
            # Skip the load for shards that only contain disk-offloaded weights when using safetensors for the offload.
            if shard_file in disk_only_shard_files:
                continue
            map_location = None
            if (
                device_map is not None
                and hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method == QuantizationMethod.TORCHAO
                and hf_quantizer.quantization_config.quant_type == "int4_weight_only"
            ):
                map_location = torch.device([d for d in device_map.values() if d not in ["cpu", "disk"]][0])
            state_dict = load_state_dict(
                shard_file, is_quantized=is_quantized, map_location=map_location, weights_only=weights_only
            )

            # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
            # matching the weights in the model.
            mismatched_keys += _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )
            if low_cpu_mem_usage:
                if is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized:
                    for key, param in model_to_load.state_dict().items():
                        if param.device == torch.device("meta"):
                            set_module_tensor_to_device(
                                model_to_load, key, "cpu", torch.empty(*param.size(), dtype=dtype)
                            )
                else:
                    new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
                        model_to_load,
                        state_dict,
                        start_prefix,
                        expected_keys,
                        device_map=device_map,
                        offload_folder=offload_folder,
                        offload_index=offload_index,
                        state_dict_folder=state_dict_folder,
                        state_dict_index=state_dict_index,
                        dtype=dtype,
                        hf_quantizer=hf_quantizer,
                        is_safetensors=is_safetensors,
                        keep_in_fp32_modules=keep_in_fp32_modules,
                        unexpected_keys=unexpected_keys,
                    )
                    error_msgs += new_error_msgs
            else:
                # Sharded checkpoint or whole but low_cpu_mem_usage==True
                if assign_to_params_buffers is None:
                    assign_to_params_buffers = check_support_param_buffer_assignment(
                        model_to_load, state_dict, start_prefix
                    )
                error_msgs += _load_state_dict_into_model(
                    model_to_load, state_dict, start_prefix, assign_to_params_buffers
                )

            # force memory release
            del state_dict
            gc.collect()

        if offload_index is not None and len(offload_index) > 0:
            if model != model_to_load:
                # We need to add the prefix of the base model
                prefix = cls.base_model_prefix
                if not is_safetensors:
                    for weight_name in offload_index:
                        shutil.move(
                            os.path.join(offload_folder, f"{weight_name}.dat"),
                            os.path.join(offload_folder, f"{prefix}.{weight_name}.dat"),
                        )
                offload_index = {f"{prefix}.{key}": value for key, value in offload_index.items()}
            if not is_safetensors:
                save_offload_index(offload_index, offload_folder)
                offload_index = None

        if offload_state_dict:
            # Load back temporarily offloaded state dict
            load_offloaded_weights(model_to_load, state_dict_index, state_dict_folder)
            shutil.rmtree(state_dict_folder)

    if len(error_msgs) > 0:
        error_msg = "\n\t".join(error_msgs)
        if "size mismatch" in error_msg:
            error_msg += (
                "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
            )
        raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

    missing_keys = list(filter(lambda x: 'adapter' not in x,  missing_keys))

    if len(unexpected_keys) > 0:
        archs = [] if model.config.architectures is None else model.config.architectures
        warner = logger.warning if model.__class__.__name__ in archs else logger.info
        warner(
            f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
            f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
            f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
            " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
            " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
            f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
            " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
        )
    else:
        logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
            " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
        )
    elif len(mismatched_keys) == 0:
        logger.info(
            f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
            f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
            " training."
        )
    if len(mismatched_keys) > 0:
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        logger.warning(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
            f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
            " to use it for predictions and inference."
        )

    return model, missing_keys, unexpected_keys, mismatched_keys, offload_index, error_msgs