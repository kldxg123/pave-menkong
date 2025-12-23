import os
import json
import torch
import numpy as np
import time
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel
import torch.nn as nn
from typing import Optional, Tuple, Union, List

# ==============================================================================
# 0. 配置区域
# ==============================================================================
CONFIG = {
    "model_path": "./checkpoints/pave_v5_1_3_lora_avsd_7B_imagebind",
    "model_base": "./models/llava-onevision-qwen2-7b-ov",
    "video_folder": "./data/video_instruction_tuning/avsd/Charades_vu17_test_480",
    "feature_folder": "./data/video_instruction_tuning/avsd/Charades_vu17_test_audio_imagebind_feat",
    "annotation_file": "./data/video_instruction_tuning/avsd/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json",
    "pred_save_file": "./data/video_instruction_tuning/prediction/pave_prediction_v14.json",
    "num_frames": 32,
    "batch_size": 1,
    "conv_mode": "conv_llava_ov_qwen"
}

# 动态导入
try:
    from libs.conversation_lib import conv_templates, SeparatorStyle
    from libs.mm_utils import tokenizer_vision_token, KeywordsStoppingCriteria
    from libs.model.language_model.pave_qwen2 import PAVEQwen2Config, PAVEQwen2ForCausalLM
    from libs.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
except ImportError as e:
    print(f"[Fatal] 导入失败: {e}")
    exit(1)

SAFE_IMAGE_TOKEN_ID = 151646 

# ==============================================================================
# 1. 核心逻辑：手动构建 Embedding (带详细日志)
# ==============================================================================
def get_model_inputs(model, vision_tower, input_ids, video_feat, images, feat_frame_nums):
    t_start = time.time()
    
    base_model = model.get_model()
    mm_projector = base_model.mm_projector
    embed_tokens = base_model.embed_tokens
    
    target_dtype = mm_projector[0].weight.dtype
    target_device = mm_projector[0].weight.device
    
    # 1. 计算视频特征
    if video_feat is not None:
        video_feat = video_feat.to(dtype=target_dtype, device=target_device)
        if hasattr(base_model, 'video_feat_projector') and base_model.video_feat_projector is not None:
            video_feat = base_model.video_feat_projector(video_feat)

    # 2. 计算图像特征
    images = images.to(dtype=target_dtype, device=target_device)
    
    if images.ndim == 5:
        # [Batch, Frames, Channels, H, W]
        b, t, c, h, w = images.shape
        images_reshaped = images.view(b * t, c, h, w)
        
        # Vision Tower Forward
        print(f"   [Debug] Vision Tower Input: {images_reshaped.shape}")
        image_features = vision_tower(images_reshaped)
        print(f"   [Debug] Vision Tower Output: {image_features.shape}")
        
        image_features = mm_projector(image_features)
        
        # Restore shape
        num_patches = image_features.shape[1]
        output_dim = image_features.shape[-1]
        image_features = image_features.view(b, t, num_patches, output_dim)
        
    else:
        image_features = vision_tower(images)
        image_features = mm_projector(image_features)
    
    # 3. 拼接逻辑
    input_ids = input_ids[0]
    
    parts = []
    current_part = []
    for idx, token_id in enumerate(input_ids):
        if token_id == IMAGE_TOKEN_INDEX:
            if current_part:
                parts.append(torch.tensor(current_part, device=input_ids.device))
                current_part = []
            parts.append(IMAGE_TOKEN_INDEX)
        else:
            current_part.append(token_id)
    if current_part:
        parts.append(torch.tensor(current_part, device=input_ids.device))

    final_embeds = []
    cur_image_idx = 0
    
    for part in parts:
        if isinstance(part, int) and part == IMAGE_TOKEN_INDEX:
            curr_image_feat = image_features[cur_image_idx] # [Frames, Patches, Dim]
            
            # Video Feat Fusion
            if video_feat is not None:
                curr_video_feat = video_feat[0] 
                
                # Squeeze extra dims
                while curr_video_feat.ndim > 2:
                    curr_video_feat = curr_video_feat.squeeze(0)
                
                target_len = curr_image_feat.shape[0]
                if curr_video_feat.shape[0] != target_len:
                    if curr_video_feat.shape[0] != target_len and curr_video_feat.shape[1] == target_len:
                         curr_video_feat = curr_video_feat.t()

                    temp_feat = curr_video_feat.unsqueeze(0).permute(0, 2, 1)
                    temp_feat = torch.nn.functional.interpolate(temp_feat.float(), size=target_len, mode='linear', align_corners=False).to(curr_image_feat.dtype)
                    curr_video_feat = temp_feat.permute(0, 2, 1).squeeze(0)
                
                combined_feat = curr_image_feat + curr_video_feat.unsqueeze(1)
                final_embeds.append(combined_feat.flatten(0, 1))
            else:
                final_embeds.append(curr_image_feat.flatten(0, 1))
                
            cur_image_idx += 1
        else:
            safe_part = torch.clamp(part, min=0, max=model.config.vocab_size - 1)
            text_embed = embed_tokens(safe_part)
            final_embeds.append(text_embed)
            
    inputs_embeds = torch.cat(final_embeds, dim=0).unsqueeze(0)
    print(f"   [Debug] Embedding Constrution Time: {time.time() - t_start:.2f}s")
    return inputs_embeds

# ==============================================================================
# 2. 工具函数
# ==============================================================================
from transformers.generation import utils as generation_utils
def _validate_model_kwargs_skip(self, model_kwargs): pass
generation_utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs_skip

def load_video_frames(video_path, num_frames=32):
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
        return vr.get_batch(indices).asnumpy()
    except Exception:
        return np.zeros((num_frames, 336, 336, 3), dtype=np.uint8)

class AVSDTestDataset(Dataset):
    def __init__(self, config, image_processor):
        self.config = config
        self.image_processor = image_processor
        try:
            with open(config["annotation_file"], "r") as f:
                self.data = json.load(f).get("dialogs", [])
        except:
            self.data = []
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        vid = item['image_id']
        video_path = os.path.join(self.config["video_folder"], f"{vid}.mp4")
        if os.path.exists(video_path): frames = load_video_frames(video_path, self.config["num_frames"])
        else: frames = np.zeros((32, 336, 336, 3), dtype=np.uint8)
        video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        feat_path = os.path.join(self.config["feature_folder"], f"{vid}.pt")
        audio_feat = torch.load(feat_path) if os.path.exists(feat_path) else torch.zeros((100, 1024), dtype=torch.float16)
        if audio_feat.requires_grad: audio_feat.requires_grad = False
        return {"vid": vid, "video": [video_tensor], "audio_feature": audio_feat, "conversation": item['dialog']}

# ==============================================================================
# 3. 主程序
# ==============================================================================
def main():
    print(">>> [Init] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_base"], use_fast=False)

    print(">>> [Init] Loading Model...")
    base_cfg = PAVEQwen2Config.from_pretrained(CONFIG["model_base"])
    
    # [优化] 尝试使用 sdpa 加速注意力计算，防止长序列卡死
    try:
        base_cfg._attn_implementation = "sdpa" 
        print(">>> [Optimization] Using SDPA attention implementation.")
    except:
        base_cfg._attn_implementation = "eager"
        print(">>> [Warning] SDPA not available, falling back to Eager (Slower).")

    model = PAVEQwen2ForCausalLM.from_pretrained(CONFIG["model_base"], config=base_cfg, device_map="auto", torch_dtype=torch.float16)

    print(">>> [Init] Loading LoRA...")
    lora_cfg = PAVEQwen2Config.from_pretrained(CONFIG["model_path"])
    if not hasattr(lora_cfg, 'vision_tower'): lora_cfg.vision_tower = getattr(lora_cfg, 'mm_vision_tower', "./models/siglip-so400m-patch14-384")
    for k, v in {'mm_vision_select_layer': -2, 'mm_projector_type': 'mlp2x_gelu', 'pretrain_mm_mlp_adapter': None, 'mm_use_im_start_end': False}.items():
        if not hasattr(lora_cfg, k): setattr(lora_cfg, k, v)
    
    print(">>> [Init] Initializing Vision Modules...")
    model.get_model().initialize_vision_modules(model_args=lora_cfg, fsdp=None)
    
    global_vision_tower = model.get_model().get_vision_tower()
    if global_vision_tower is None and hasattr(model.get_model(), 'vision_tower'):
        global_vision_tower = model.get_model().vision_tower
    if global_vision_tower is None: raise ValueError("Fatal Error: Could not initialize Vision Tower!")
    global_vision_tower.to(device="cuda", dtype=torch.float16)

    non_lora_path = os.path.join(CONFIG["model_path"], "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        w = torch.load(non_lora_path, map_location="cpu")
        model.load_state_dict({k.replace("base_model.model.", "").replace("model.", ""): v for k, v in w.items()}, strict=False)

    print(">>> [Fix] Moving Projectors to CUDA...")
    device = "cuda"
    dtype = torch.float16
    global_vision_tower.to(device=device, dtype=dtype)
    if model.get_model().mm_projector is not None:
        model.get_model().mm_projector.to(device=device, dtype=dtype)
    if hasattr(model.get_model(), 'video_feat_projector') and model.get_model().video_feat_projector is not None:
        model.get_model().video_feat_projector.to(device=device, dtype=dtype)

    model = PeftModel.from_pretrained(model, CONFIG["model_path"]).merge_and_unload()
    model.get_model().vision_tower = global_vision_tower
    
    from transformers import SiglipImageProcessor
    proc_path = "./models/siglip-so400m-patch14-384"
    if not os.path.exists(proc_path): proc_path = "google/siglip-so400m-patch14-384"
    image_processor = SiglipImageProcessor.from_pretrained(proc_path)
    model.eval()

    dataset = AVSDTestDataset(CONFIG, image_processor)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    results = []

    print(f">>> Start Inference (Total samples: {len(dataset)})...")
    
    # 手动控制循环，以便我们可以捕获超时或长时间等待
    pbar = tqdm(loader)
    for i, sample in enumerate(pbar):
        vid = sample['vid']
        t_sample_start = time.time()
        
        try:
            # 1. 数据准备
            video_tensor = sample['video'][0].half().cuda().unsqueeze(0)
            audio_feat = sample['audio_feature'].cuda().half().permute(1, 0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            feat_frame_num = audio_feat.shape[2]

            conv = conv_templates[CONFIG["conv_mode"]].copy()
            for k, turn in enumerate(sample['conversation']):
                qs = DEFAULT_IMAGE_TOKEN + "\n" + turn['question'][0] if k == 0 else turn['question'][0]
                conv.append_message(conv.roles[0], qs)
                if k < len(sample['conversation']) - 1:
                    conv.append_message(conv.roles[1], turn['answer'][0])

            prompt = conv.get_prompt() + conv.roles[1] + "\n"
            input_ids = tokenizer_vision_token(prompt, tokenizer, DEFAULT_IMAGE_TOKEN, return_tensors='pt').unsqueeze(0).cuda()

            # 2. 构建 Embedding
            # print(f"\n[Debug {vid}] Building Embeddings...")
            inputs_embeds = get_model_inputs(
                model,
                global_vision_tower, 
                input_ids, 
                video_feat=audio_feat, 
                images=video_tensor, 
                feat_frame_nums=torch.tensor([feat_frame_num]).cuda()
            )
            
            # [重要] 打印维度检查
            if i == 0:
                print(f"\n>>> [Check Dimension] Inputs Embeds Shape: {inputs_embeds.shape}")
                print(f">>> (If seq_len > 20000, generation will be slow!)")

            # 3. 生成
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            
            safe_ids_for_stop = input_ids.clone()
            safe_ids_for_stop[safe_ids_for_stop == IMAGE_TOKEN_INDEX] = 0
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, safe_ids_for_stop)

            # print(f"[Debug {vid}] Generating...")
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    max_new_tokens=256,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            # 4. 解码
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if output_text.endswith(stop_str): output_text = output_text[:-len(stop_str)]
            results.append({"image_id": vid, "caption": output_text})

            # 5. 耗时监控
            elapsed = time.time() - t_sample_start
            pbar.set_description(f"Processing {vid} ({elapsed:.1f}s)")
            
            if i < 2 or i % 50 == 0:
                print(f"\n[Result {vid}]: {output_text[:100]}...")

        except Exception as e:
            print(f"\n[ERROR] {vid}: {e}")
            if "CUDA" in str(e): torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(CONFIG["pred_save_file"]), exist_ok=True)
    with open(CONFIG["pred_save_file"], "w") as f:
        json.dump(results, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    main()