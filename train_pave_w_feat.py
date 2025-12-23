# Adopted from https://github.com/haotian-liu/LLaVA. The training entrance.
import os
import pathlib
import torch
import sys
import datetime

# === 移除 ipdb 防止分布式死锁 ===
# import ipdb 

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}]", *args, flush=True)

def debug_print(msg):
    # 强制打印调试信息，带上 Rank ID
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}][Rank {local_rank}] DEBUG: {msg}", flush=True)

from libs.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from libs.utils.model_trainer import VideoModelTrainer
from libs import conversation_lib as conversation_lib
from libs.model import *
from libs.utils.train_utils import parse_argument_classes, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_videotrainer, get_peft_state_non_lora_maybe_zero_3_with_state_dict
from libs.model.dynamic_gate import DynamicGate, ModalFusion

from libs.dataset.base_dataset import make_video_supervised_data_module
from utils import prepare_video_model

def get_inner_model(model):
    """递归解包 PeftModel/DistributedDataParallel 找到真正的模型"""
    inner = model
    if hasattr(inner, 'module'):
        inner = inner.module
    if hasattr(inner, 'base_model'):
        inner = inner.base_model
    if hasattr(inner, 'model'): 
        inner = inner.model
    if hasattr(inner, 'get_model'):
        inner = inner.get_model()
    return inner

def train_pave_func(attn_implementation=None):
    global local_rank
    
    # 1. 解析参数
    model_args, data_args, training_args = parse_argument_classes(sys.argv[1:])
    local_rank = training_args.local_rank
    debug_print("Arguments parsed. Starting model preparation...")

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # 2. 准备模型
    model, tokenizer = prepare_video_model(training_args, model_args, data_args, compute_dtype, attn_implementation)
    debug_print("Model and Tokenizer prepared.")

    # === FIX 1: Enable Input Grads for Gradient Checkpointing + LoRA ===
    # This is CRITICAL for the "element 0 of tensors does not require grad" error
    if training_args.gradient_checkpointing:
        debug_print("Gradient Checkpointing enabled. Enabling input gradients...")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # === FIX 2: Assign image_processor to data_args ===
    inner_model = get_inner_model(model)
    vision_tower = None
    if hasattr(inner_model, 'get_vision_tower'):
        vision_tower = inner_model.get_vision_tower()
    
    if isinstance(vision_tower, list):
        vision_tower = vision_tower[0]

    if vision_tower is not None and hasattr(vision_tower, 'image_processor'):
        data_args.image_processor = vision_tower.image_processor
    else:
        debug_print("[WARNING] Could not find image_processor in vision tower.")

    # === FIX 3: Ensure mm_use_im_start_end exists in data_args ===
    if not hasattr(data_args, 'mm_use_im_start_end'):
        config = getattr(model, 'config', None)
        if config and hasattr(config, 'mm_use_im_start_end'):
            data_args.mm_use_im_start_end = config.mm_use_im_start_end
        else:
            data_args.mm_use_im_start_end = False
        debug_print(f"Set missing data_args.mm_use_im_start_end to {data_args.mm_use_im_start_end}")

    # 3. 初始化动态门控
    if hasattr(inner_model, 'init_dynamic_gate'):
        rank0_print("Initializing dynamic gate network (Found via unwrapping)...")
        inner_model.init_dynamic_gate()
        
        try:
            device = model.device
            if hasattr(inner_model, 'fusion'):
                rank0_print(f"Moving dynamic gate to {device}...")
                inner_model.fusion.to(device)
        except Exception as e:
            rank0_print(f"Warning: Failed to move fusion layer to device: {e}")
    else:
        debug_print(f"init_dynamic_gate method NOT found on {type(inner_model)}. Skipped.")

    # 4. 准备数据集
    debug_print("Loading Dataset...")
    data_module = make_video_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    debug_print("Dataset Loaded successfully.")

    # 5. 初始化 Trainer
    debug_print("Initializing VideoModelTrainer...")
    trainer = VideoModelTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    debug_print("VideoModelTrainer Initialized.")
    
    # 6. 开始训练
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("Starting training from scratch...")
        torch.use_deterministic_algorithms(False) 
        
        debug_print("Entring trainer.train()... DeepSpeed engine will init now.")
        debug_print("NOTE: We switched to 'adamw_torch' to avoid JIT compilation hangs.")
        
        trainer.train()
    
    debug_print("Training finished.")

    # 7. 保存模型
    trainer.save_state()
    model.config.use_cache = True
    if training_args.lora_enable: 
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3_with_state_dict(
            model.state_dict(), special_key=['temporal_aggregator', 'self_attn.v_kv_proj', 'self_attn.gate_proj']
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        print(f"Process launched. Rank: {os.environ['LOCAL_RANK']}", flush=True)
    train_pave_func(attn_implementation="flash_attention_2")