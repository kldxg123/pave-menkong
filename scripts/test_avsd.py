import os
import json
import torch
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 0. 配置区域
# ==============================================================================
CONFIG = {
    "model_path": "./checkpoints/pave_v5_1_3_lora_avsd_7B_imagebind",
    "model_base": "./models/llava-onevision-qwen2-7b-ov",
    "video_folder": "./data/video_instruction_tuning/avsd/Charades_vu17_test_480",
    "feature_folder": "./data/video_instruction_tuning/avsd/Charades_vu17_test_audio_imagebind_feat",
    "annotation_file": "./data/video_instruction_tuning/avsd/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json",
    "pred_save_file": "./data/video_instruction_tuning/prediction/pave_prediction_final.json",
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

# 定义安全的 Image Token ID (使用 config.json 中的 <image> id)
SAFE_IMAGE_TOKEN_ID = 151646 

# ==============================================================================
# 1. 核心补丁：重写 PAVE 的输入处理逻辑
# ==============================================================================
def patched_prepare_inputs_labels_for_multimodal(
    self, input_ids, position_ids, attention_mask, past_key_values, labels,
    images, video_feat=None, video_feat_fps=None, feat_frame_nums=None, 
    second_feat=None, second_feat_fps=None, second_feat_frame_nums=None
):
    """
    这是一个运行时补丁，用于替换原有的 prepare_inputs_labels_for_multimodal。
    它不仅支持 -200，还支持 SAFE_IMAGE_TOKEN_ID (151646)，从而允许我们在 generate 中传入全正数。
    """
    vision_tower = self.get_vision_tower()
    
    # === 分支 1: 纯文本 / 解码阶段 ===
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        # [安全] 强制计算正确的 Position IDs，防止 -1
        if past_key_values is not None and input_ids.shape[1] == 1:
            current_seq_len = past_key_values[0][0].shape[-2]
            position_ids = torch.tensor([[current_seq_len]], dtype=torch.long, device=input_ids.device)
            if attention_mask is not None:
                target_len = current_seq_len + 1
                if attention_mask.shape[1] < target_len:
                    pad = torch.ones((attention_mask.shape[0], target_len - attention_mask.shape[1]), 
                                   dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, pad], dim=1)
        elif position_ids is None and attention_mask is not None:
            position_ids = (torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1).clamp(min=0)
            
        return None, position_ids, attention_mask, past_key_values, input_ids, labels

    # === 分支 2: 多模态 Prefill ===
    # 处理视频/辅助特征投影
    if video_feat is not None:
        dtype = self.get_model().mm_projector.weight.dtype
        device = self.get_model().mm_projector.weight.device
        video_feat = video_feat.to(dtype=dtype, device=device)
        video_feat = self.get_model().video_feat_projector(video_feat)

    if second_feat is not None:
        dtype = self.get_model().mm_projector.weight.dtype
        device = self.get_model().mm_projector.weight.device
        second_feat = second_feat.to(dtype=dtype, device=device)
        second_feat = self.get_model().second_feat_projector(second_feat)

    # 提取图像特征
    if type(images) is list or images.ndim == 5:
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.get_model().get_vision_tower()(concat_images)
        image_features = self.get_model().mm_projector(image_features)
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
    else:
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)

    embedding_layer = self.get_model().embed_tokens
    new_input_embeds = []
    new_labels = [] if labels is not None else None
    cur_image_idx = 0

    for batch_idx, cur_input_ids in enumerate(input_ids):
        # [关键逻辑] 同时检测 -200 和 151646 作为分割点
        # 优先检测 -200，如果没有则检测 SAFE_IMAGE_TOKEN_ID
        is_neg_token = (cur_input_ids == IMAGE_TOKEN_INDEX)
        is_safe_token = (cur_input_ids == SAFE_IMAGE_TOKEN_ID)
        
        if is_neg_token.sum() > 0:
            split_mask = is_neg_token
        elif is_safe_token.sum() > 0:
            split_mask = is_safe_token
        else:
            split_mask = torch.zeros_like(cur_input_ids, dtype=torch.bool)

        num_images = split_mask.sum()

        # 无图像情况
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = embedding_layer(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            if labels is not None: new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        # 分割
        image_token_indices = [-1] + torch.where(split_mask)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        if labels is not None:
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            if labels is not None:
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])

        # 拼接文本部分进行 Embedding
        if len(cur_input_ids_noim) > 0:
            flat_ids = torch.cat(cur_input_ids_noim)
            cur_input_embeds = embedding_layer(flat_ids)
            split_sizes = [x.shape[0] for x in cur_input_ids_noim]
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        else:
            cur_input_embeds_no_im = [torch.empty((0, embedding_layer.weight.shape[1]), device=embedding_layer.weight.device)] * (num_images + 1)

        new_input_embeds_curr = []
        new_labels_curr = []

        for i in range(num_images + 1):
            new_input_embeds_curr.append(cur_input_embeds_no_im[i])
            if labels is not None: new_labels_curr.append(cur_labels_noim[i])
            
            if i < num_images:
                curr_image_feat = image_features[cur_image_idx]
                # 视频/音频特征融合逻辑
                if video_feat is not None:
                    curr_video_feat = video_feat[batch_idx]
                    if curr_video_feat.shape[0] != curr_image_feat.shape[0]:
                        if curr_video_feat.ndim > 2: curr_video_feat = curr_video_feat.flatten(2).permute(1, 0)
                        temp_feat = curr_video_feat.unsqueeze(0).permute(0, 2, 1)
                        target_len = curr_image_feat.shape[0]
                        temp_feat = F.interpolate(temp_feat.float(), size=target_len, mode='linear', align_corners=False).to(curr_image_feat.dtype)
                        curr_video_feat = temp_feat.permute(0, 2, 1).squeeze(0)
                    
                    combined_feat = curr_image_feat + curr_video_feat
                    if second_feat is not None:
                         curr_second = second_feat[batch_idx]
                         if curr_second.ndim > 2: curr_second = curr_second.flatten(2).permute(1, 0)
                         if curr_second.shape[0] != combined_feat.shape[0]:
                            temp_s = curr_second.unsqueeze(0).permute(0, 2, 1)
                            temp_s = F.interpolate(temp_s.float(), size=combined_feat.shape[0], mode='linear', align_corners=False).to(combined_feat.dtype)
                            curr_second = temp_s.permute(0, 2, 1).squeeze(0)
                         combined_feat = combined_feat + curr_second
                    new_input_embeds_curr.append(combined_feat)
                else:
                    new_input_embeds_curr.append(curr_image_feat)
                cur_image_idx += 1

        new_input_embeds.append(torch.cat(new_input_embeds_curr))
        if labels is not None: new_labels.append(torch.cat(new_labels_curr))

    if labels is not None:
        return torch.stack(new_input_embeds), position_ids, attention_mask, past_key_values, None, torch.stack(new_labels)
    else:
        return torch.stack(new_input_embeds), position_ids, attention_mask, past_key_values, None, None

# ==============================================================================
# 2. 工具函数 & Dataset
# ==============================================================================
# Monkey Patch Transformers
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
    # 1. 动态应用 Monkey Patch (必须在实例化前)
    print(">>> [Patch] Applying Runtime Monkey Patch to PAVEQwen2ForCausalLM...")
    PAVEQwen2ForCausalLM.prepare_inputs_labels_for_multimodal = patched_prepare_inputs_labels_for_multimodal

    # 2. 加载模型
    print(">>> [Init] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_base"], use_fast=False)

    print(">>> [Init] Loading Model...")
    base_cfg = PAVEQwen2Config.from_pretrained(CONFIG["model_base"])
    base_cfg._attn_implementation = "eager"
    model = PAVEQwen2ForCausalLM.from_pretrained(CONFIG["model_base"], config=base_cfg, device_map="auto", torch_dtype=torch.float16)

    print(">>> [Init] Loading LoRA...")
    lora_cfg = PAVEQwen2Config.from_pretrained(CONFIG["model_path"])
    # 补全配置
    if not hasattr(lora_cfg, 'vision_tower'): lora_cfg.vision_tower = getattr(lora_cfg, 'mm_vision_tower', "./models/siglip-so400m-patch14-384")
    for k, v in {'mm_vision_select_layer': -2, 'mm_projector_type': 'mlp2x_gelu', 'pretrain_mm_mlp_adapter': None, 'mm_use_im_start_end': False}.items():
        if not hasattr(lora_cfg, k): setattr(lora_cfg, k, v)
    
    model.get_model().initialize_vision_modules(model_args=lora_cfg, fsdp=None)
    
    # 加载 Non-Lora
    non_lora_path = os.path.join(CONFIG["model_path"], "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        w = torch.load(non_lora_path, map_location="cpu")
        model.load_state_dict({k.replace("base_model.model.", "").replace("model.", ""): v for k, v in w.items()}, strict=False)

    model = PeftModel.from_pretrained(model, CONFIG["model_path"]).merge_and_unload()
    if hasattr(model, 'initialize_vision_tokenizer'): model.initialize_vision_tokenizer(lora_cfg, tokenizer=tokenizer)
    
    # 加载图像处理器
    from transformers import SiglipImageProcessor
    proc_path = "./models/siglip-so400m-patch14-384"
    if not os.path.exists(proc_path): proc_path = "google/siglip-so400m-patch14-384"
    image_processor = SiglipImageProcessor.from_pretrained(proc_path)
    model.eval()

    # 3. 准备推理
    dataset = AVSDTestDataset(CONFIG, image_processor)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    results = []

    print(f">>> Start Inference...")
    for sample in tqdm(loader):
        vid = sample['vid']
        try:
            video_tensor = sample['video'][0].half().cuda().unsqueeze(0)
            audio_feat = sample['audio_feature'].cuda().half().permute(1, 0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            feat_frame_num = audio_feat.shape[2]

            conv = conv_templates[CONFIG["conv_mode"]].copy()
            for i, turn in enumerate(sample['conversation']):
                qs = DEFAULT_IMAGE_TOKEN + "\n" + turn['question'][0] if i == 0 else turn['question'][0]
                conv.append_message(conv.roles[0], qs)
                if i < len(sample['conversation']) - 1:
                    conv.append_message(conv.roles[1], turn['answer'][0])

            prompt = conv.get_prompt() + conv.roles[1] + "\n"
            input_ids = tokenizer_vision_token(prompt, tokenizer, DEFAULT_IMAGE_TOKEN, return_tensors='pt').unsqueeze(0).cuda()

            # [终极安全操作]
            # 1. 找到所有 -200 (IMAGE_TOKEN_INDEX)
            # 2. 将它们替换为 SAFE_IMAGE_TOKEN_ID (151646)
            # 3. 这样传给 generate 的 input_ids 就是全正数，绝对不会崩溃
            # 4. 我们打过补丁的 prepare_inputs... 会认出 151646 并正确处理
            safe_input_ids = input_ids.clone()
            safe_input_ids[safe_input_ids == IMAGE_TOKEN_INDEX] = SAFE_IMAGE_TOKEN_ID
            
            # 同时确保没有其他越界值
            REAL_VOCAB_SIZE = model.config.vocab_size
            safe_input_ids = torch.clamp(safe_input_ids, max=REAL_VOCAB_SIZE - 1)

            # Pad Token
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, safe_input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    safe_input_ids,  # 传入全正数的 ID
                    video_feats=audio_feat,
                    feat_frame_nums=torch.tensor([feat_frame_num]).cuda(),
                    images=video_tensor,
                    image_sizes=[200],
                    attention_mask=safe_input_ids.ne(pad_token_id).long().cuda(),
                    pad_token_id=pad_token_id,
                    do_sample=False,
                    max_new_tokens=256,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            # 解码
            if output_ids.shape[1] > safe_input_ids.shape[1]:
                output_ids = output_ids[:, safe_input_ids.shape[1]:]
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if output_text.endswith(stop_str): output_text = output_text[:-len(stop_str)]
            results.append({"image_id": vid, "caption": output_text})

        except Exception as e:
            print(f"[ERROR] Failed ID: {vid} | {e}")
            if "CUDA" in str(e): 
                torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(CONFIG["pred_save_file"]), exist_ok=True)
    with open(CONFIG["pred_save_file"], "w") as f:
        json.dump(results, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    main()