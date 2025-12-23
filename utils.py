import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from libs.model import *
from libs.model.language_model.pave_qwen2 import PAVEQwen2ForCausalLM
from libs.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def prepare_video_model(training_args, model_args, data_args, compute_dtype, attn_implementation="flash_attention_2"):
    
    # Load config and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # === FIX: Robust check for model class ===
    # Check if 'model_class' attribute exists, otherwise use the class name itself
    current_model_class_name = getattr(model_args, 'model_class', '')
    if not current_model_class_name:
        current_model_class_name = type(model_args).__name__

    print(f"[DEBUG] Detected Model Class Name: {current_model_class_name}")

    # Load model
    if 'VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B' in current_model_class_name:
        print(f"Loading PAVEQwen2ForCausalLM from {model_args.model_name_or_path}...")
        model = PAVEQwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=compute_dtype,
            attn_implementation=attn_implementation
        )
    else:
        # Fallback for other models
        print(f"Loading generic AutoModelForCausalLM from {model_args.model_name_or_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=compute_dtype,
            attn_implementation=attn_implementation
        )

    # Enable LoRA if needed
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # Initialize vision modules
    if hasattr(model, "get_model") and hasattr(model.get_model(), "initialize_vision_modules"):
        model.get_model().initialize_vision_modules(model_args=model_args)
        
    # Initialize vision projector
    if hasattr(model, "get_model") and hasattr(model.get_model(), "init_video_feat_connector"):
        # Ensure we pass the config which contains fast_feat_type
        model.get_model().init_video_feat_connector(model_args)

    return model, tokenizer

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    
    # Remove multimodal modules from LoRA
    return list(lora_module_names)