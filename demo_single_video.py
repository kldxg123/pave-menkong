import os
import torch
import numpy as np
from decord import VideoReader, cpu
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import SiglipImageProcessor

# 引入项目依赖
from libs.conversation_lib import conv_templates, SeparatorStyle
from libs.mm_utils import tokenizer_vision_token, KeywordsStoppingCriteria
from libs.constants import DEFAULT_IMAGE_TOKEN
from libs.model.language_model.pave_qwen2 import PAVEQwen2Config, PAVEQwen2ForCausalLM

# ==========================================
# 1. 强制绕过 Transformers 参数检查 (Monkey Patch)
# ==========================================
from transformers.generation import utils as generation_utils
def _validate_model_kwargs_skip(self, model_kwargs):
    pass 
generation_utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs_skip

# ==========================================
# 2. 配置路径
# ==========================================
VIDEO_PATH = "/home/app-ahr/PAVE/data/test/1.MP4"
MODEL_BASE = "./models/llava-onevision-qwen2-7b-ov"
MODEL_LORA = "./checkpoints/pave_v5_1_3_lora_avsd_7B_imagebind"
SIGLIP_PATH = "./models/siglip-so400m-patch14-384"  # 如果本地没有，代码会自动尝试下载

# 对话 Prompt (你可以修改这里的问题)
USER_QUERY = "请详细描述这个视频的内容。" 

# ==========================================
# 3. 核心工具函数
# ==========================================
def load_video_tensor(video_path, image_processor, num_frames=32):
    """读取视频 -> 抽帧 -> SigLIP 处理 -> Tensor"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    print(f"[处理] 正在读取视频: {video_path}")
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    
    # 均匀采样 32 帧
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy() # Shape: (T, H, W, C)
    
    # 使用 SigLIP 预处理
    # return_tensors="pt" 返回 PyTorch Tensor
    # pixel_values Shape: (1, T, C, H, W) 或 (T, C, H, W) 取决于 processor 版本
    inputs = image_processor.preprocess(frames, return_tensors="pt")
    video_tensor = inputs["pixel_values"]
    
    # PAVE 期望形状: [T, C, H, W] -> [1, T, C, H, W] (Batch Dim)
    # 注意：具体形状取决于模型内部处理，通常 SigLIP 输出是 [T, 3, 384, 384]
    # 我们先转为 half 精度并移到 GPU
    return video_tensor.half().cuda().unsqueeze(0)

def get_dummy_audio_feat():
    """生成全 0 的音频特征占位符 (因为单视频只有图像)"""
    # 模拟 Shape: [100, 1024] -> PAVE 格式 [1, 1024, 100, 1, 1]
    # Permute(1,0) -> [1024, 100]
    dummy = torch.zeros((100, 1024), dtype=torch.float16, device='cuda')
    return dummy.permute(1, 0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

# ==========================================
# 4. 模型加载逻辑 (带 Config 自动修复)
# ==========================================
def load_pave_model():
    print(">>> 正在加载模型组件...")
    
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=False)
    
    # 2. Config (强制 Eager 模式)
    base_cfg = PAVEQwen2Config.from_pretrained(MODEL_BASE)
    base_cfg._attn_implementation = "eager"
    
    # 3. Base Model
    model = PAVEQwen2ForCausalLM.from_pretrained(
        MODEL_BASE, 
        config=base_cfg, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    # 4. LoRA Config Patching
    lora_cfg = PAVEQwen2Config.from_pretrained(MODEL_LORA)
    # 自动补全缺失属性
    if not hasattr(lora_cfg, 'vision_tower'):
        lora_cfg.vision_tower = getattr(lora_cfg, 'mm_vision_tower', SIGLIP_PATH)
    for k, v in {'mm_vision_select_layer': -2, 'mm_projector_type': 'mlp2x_gelu', 'pretrain_mm_mlp_adapter': None, 'mm_use_im_start_end': False}.items():
        if not hasattr(lora_cfg, k): setattr(lora_cfg, k, v)
    
    # 5. 初始化视觉模块
    model.get_model().initialize_vision_modules(model_args=lora_cfg, fsdp=None)
    
    # 6. 加载 Projector 权重
    non_lora_path = os.path.join(MODEL_LORA, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        print(f"    加载投影层权重: {non_lora_path}")
        w = torch.load(non_lora_path, map_location="cpu")
        clean_w = {k.replace("base_model.model.", "").replace("model.", ""): v for k, v in w.items()}
        model.load_state_dict(clean_w, strict=False)

    # 7. 加载 LoRA
    print("    加载 LoRA 权重...")
    model = PeftModel.from_pretrained(model, MODEL_LORA).merge_and_unload()
    
    # 8. Tokenizer Resize
    if hasattr(model, 'initialize_vision_tokenizer'):
        model.initialize_vision_tokenizer(lora_cfg, tokenizer=tokenizer)
    
    # 9. Image Processor
    try:
        image_processor = SiglipImageProcessor.from_pretrained(SIGLIP_PATH)
    except:
        print("[警告] 本地 SigLIP 未找到，尝试从 HuggingFace 下载...")
        image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        
    return tokenizer, model, image_processor

# ==========================================
# 5. 主程序
# ==========================================
def main():
    # --- 1. 加载模型 ---
    tokenizer, model, image_processor = load_pave_model()
    model.eval()
    
    # 获取真实词表大小用于安全钳制
    REAL_VOCAB_SIZE = model.get_input_embeddings().weight.shape[0]
    
    # --- 2. 准备数据 ---
    # 视频特征
    video_tensor = load_video_tensor(VIDEO_PATH, image_processor)
    # 音频特征 (Dummy)
    audio_feat = get_dummy_audio_feat()
    feat_frame_num = audio_feat.shape[2]
    
    # --- 3. 构建 Prompt ---
    # 使用 Qwen2Tokenizer 模版
    conv = conv_templates["Qwen2Tokenizer"].copy()
    # 格式: <image>\nUser Question
    qs = DEFAULT_IMAGE_TOKEN + "\n" + USER_QUERY
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    print(f"\n[Prompt] {prompt}")
    
    # --- 4. Tokenize & Clamp ---
    input_ids = tokenizer_vision_token(prompt, tokenizer, DEFAULT_IMAGE_TOKEN, return_tensors='pt').unsqueeze(0).cuda()
    
    # [安全锁] 防止 ID 越界
    if input_ids.max() >= REAL_VOCAB_SIZE:
        print(f"[Info] Clamping input_ids (Max: {input_ids.max()} -> {REAL_VOCAB_SIZE-1})")
        input_ids = torch.clamp(input_ids, max=REAL_VOCAB_SIZE - 1)
        
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    
    # --- 5. 生成 ---
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    
    print(">>> 开始生成...")
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                video_feats=audio_feat, # 音频/ImageBind特征
                feat_frame_nums=torch.tensor([feat_frame_num]).cuda(),
                images=video_tensor,    # 视频像素特征
                image_sizes=[200],      # PAVE 内部标识
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,        # Greedy Search 保证稳定
                temperature=0.0,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
        
        # --- 6. 解码输出 ---
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if output_text.endswith(stop_str):
            output_text = output_text[:-len(stop_str)]
            
        print("="*40)
        print(f"视频: {VIDEO_PATH}")
        print(f"问题: {USER_QUERY}")
        print(f"模型回答: {output_text}")
        print("="*40)
        
    except Exception as e:
        print(f"\n[错误] 生成失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()