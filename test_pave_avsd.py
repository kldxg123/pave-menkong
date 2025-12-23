import argparse
import torch
import os
import json
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, SiglipImageProcessor
from peft import PeftModel

# 引入项目依赖
from libs.conversation_lib import conv_templates, SeparatorStyle
from libs.mm_utils import tokenizer_vision_token, KeywordsStoppingCriteria
from libs.constants import DEFAULT_IMAGE_TOKEN
from libs.model.language_model.pave_qwen2 import PAVEQwen2Config, PAVEQwen2ForCausalLM

# ==========================================
# 1. Monkey Patch (必不可少)
# ==========================================
from transformers.generation import utils as generation_utils
def _validate_model_kwargs_skip(self, model_kwargs):
    pass 
generation_utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs_skip

# ==========================================
# 2. 数据集加载类
# ==========================================
class AVSDDataset(Dataset):
    def __init__(self, annotation_file, video_folder, feature_folder, image_processor, num_frames=32):
        self.video_folder = video_folder
        self.feature_folder = feature_folder
        self.image_processor = image_processor
        self.num_frames = num_frames
        
        # 加载标注文件
        print(f"Loading annotations from {annotation_file}")
        with open(annotation_file, 'r') as f:
            self.data = json.load(f).get('dialogs', [])
            
    def __len__(self):
        return len(self.data)
        
    def load_video(self, vid):
        video_path = os.path.join(self.video_folder, f"{vid}.mp4")
        if not os.path.exists(video_path):
            # 视频缺失，返回全黑帧
            return np.zeros((self.num_frames, 336, 336, 3), dtype=np.uint8)
            
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
            return frames
        except Exception as e:
            print(f"[WARN] Video load error {vid}: {e}")
            return np.zeros((self.num_frames, 336, 336, 3), dtype=np.uint8)

    def load_audio_feature(self, vid):
        feat_path = os.path.join(self.feature_folder, f"{vid}.pt")
        if os.path.exists(feat_path):
            try:
                # 假设存储的是 [T, D]
                feat = torch.load(feat_path)
                return feat
            except:
                pass
        # 默认音频特征形状 [100, 1024]
        return torch.zeros((100, 1024), dtype=torch.float16)

    def __getitem__(self, idx):
        item = self.data[idx]
        vid = item['image_id']
        dialog = item['dialog']
        
        # 1. 视频处理 -> Tensor
        video_frames = self.load_video(vid)
        video_inputs = self.image_processor.preprocess(video_frames, return_tensors="pt")
        video_tensor = video_inputs["pixel_values"][0] # [T, C, H, W]
        
        # 2. 音频处理 -> Tensor
        audio_feat = self.load_audio_feature(vid)
        
        return {
            "vid": vid,
            "video_tensor": video_tensor,
            "audio_feat": audio_feat,
            "dialog": dialog
        }

# ==========================================
# 3. 模型加载器 (带自动修复)
# ==========================================
def load_model_and_processor(args):
    print(">>> Initializing Tokenizer & Config...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    
    # 强制 Eager 模式
    base_cfg = PAVEQwen2Config.from_pretrained(args.model_base)
    base_cfg._attn_implementation = "eager"
    
    print(">>> Loading Base Model...")
    model = PAVEQwen2ForCausalLM.from_pretrained(
        args.model_base, 
        config=base_cfg, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    print(">>> Patching LoRA Config...")
    lora_cfg = PAVEQwen2Config.from_pretrained(args.model_path)
    # 补丁：修复缺失参数
    if not hasattr(lora_cfg, 'vision_tower'):
        lora_cfg.vision_tower = getattr(lora_cfg, 'mm_vision_tower', "./models/siglip-so400m-patch14-384")
    # 补全 PAVE 必要参数
    defaults = {
        'mm_vision_select_layer': -2, 
        'mm_projector_type': 'mlp2x_gelu', 
        'pretrain_mm_mlp_adapter': None, 
        'mm_use_im_start_end': False
    }
    for k, v in defaults.items():
        if not hasattr(lora_cfg, k): setattr(lora_cfg, k, v)
        
    model.get_model().initialize_vision_modules(model_args=lora_cfg, fsdp=None)
    
    # 加载 Projector 权重
    non_lora_path = os.path.join(args.model_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        print("    Loading non-lora weights...")
        w = torch.load(non_lora_path, map_location="cpu")
        clean_w = {k.replace("base_model.model.", "").replace("model.", ""): v for k, v in w.items()}
        model.load_state_dict(clean_w, strict=False)

    print(">>> Loading LoRA Adapter...")
    model = PeftModel.from_pretrained(model, args.model_path).merge_and_unload()
    
    if hasattr(model, 'initialize_vision_tokenizer'):
        model.initialize_vision_tokenizer(lora_cfg, tokenizer=tokenizer)
        
    # 加载 Image Processor
    try:
        image_processor = SiglipImageProcessor.from_pretrained(lora_cfg.vision_tower)
    except:
        image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        
    return model, tokenizer, image_processor

# ==========================================
# 4. 推理主循环
# ==========================================
def eval_model(args):
    # 1. 加载
    model, tokenizer, image_processor = load_model_and_processor(args)
    model.eval()
    
    # 获取真实词表大小 (用于安全钳制)
    REAL_VOCAB_SIZE = model.get_input_embeddings().weight.shape[0]
    print(f">>> Vocabulary Safe Limit: {REAL_VOCAB_SIZE}")

    # 2. 数据集
    dataset = AVSDDataset(
        args.annotation_file, 
        args.video_folder, 
        args.feature_folder, 
        image_processor,
        num_frames=32
    )
    # Batch size 必须为 1 (多模态输入长度不一，难以 collate)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    
    results = []
    
    print(f">>> Start Inference ({len(dataset)} samples)...")
    
    for sample in tqdm(loader):
        vid = sample['vid']
        
        # 准备数据 (Batch Dim = 1)
        # Video: [1, T, C, H, W]
        video_tensor = sample['video_tensor'].half().cuda().unsqueeze(0)
        
        # Audio: [T, D] -> [1, D, T, 1, 1] (PAVE 特殊格式)
        audio_feat = sample['audio_feat'].half().cuda()
        # 假设原始是 [100, 1024] -> permute -> [1024, 100] -> unsqueeze -> [1, 1024, 100, 1, 1]
        audio_feat = audio_feat.permute(1, 0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        feat_frame_num = audio_feat.shape[2]

        # 构建对话
        # 使用 Qwen2Tokenizer 模版避免 ChatML Bug
        conv = conv_templates["Qwen2Tokenizer"].copy()
        dialog = sample['dialog']
        
        # AVSD 任务通常是给定历史，预测最后一轮的回答
        # 或者我们需要为每一轮生成？通常 Evaluation 是生成最后一轮。
        # 这里我们按照之前的逻辑，构建完整历史，让模型预测最后一句。
        for i, turn in enumerate(dialog):
            question = turn['question'][0]
            answer = turn['answer'][0]
            
            if i == 0:
                # 第一轮加入 <image> token
                qs = DEFAULT_IMAGE_TOKEN + "\n" + question
            else:
                qs = question
            
            conv.append_message(conv.roles[0], qs)
            
            # 如果是最后一轮，不加答案，让模型预测
            if i == len(dialog) - 1:
                conv.append_message(conv.roles[1], None)
            else:
                conv.append_message(conv.roles[1], answer)
        
        prompt = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_vision_token(prompt, tokenizer, DEFAULT_IMAGE_TOKEN, return_tensors='pt').unsqueeze(0).cuda()
        
        # [安全锁] 钳制 ID
        if input_ids.max() >= REAL_VOCAB_SIZE:
            input_ids = torch.clamp(input_ids, max=REAL_VOCAB_SIZE - 1)
            
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long().cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    video_feats=audio_feat,
                    feat_frame_nums=torch.tensor([feat_frame_num]).cuda(),
                    images=video_tensor,
                    image_sizes=[200],
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=512,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )
            
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if output_text.endswith(stop_str):
                output_text = output_text[:-len(stop_str)]
                
            results.append({
                "image_id": vid,
                "caption": output_text
            })
            
        except Exception as e:
            print(f"[ERROR] Failed {vid}: {e}")
            continue

    # 保存结果
    os.makedirs(os.path.dirname(args.pred_save), exist_ok=True)
    with open(args.pred_save, 'w') as f:
        json.dump(results, f, indent=4)
    print(f">>> Evaluation Finished! Results saved to {args.pred_save}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认参数设置为你环境中的路径
    parser.add_argument("--model-path", type=str, default="./checkpoints/pave_v5_1_3_lora_avsd_7B_imagebind")
    parser.add_argument("--model-base", type=str, default="./models/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--video-folder", type=str, default="./data/video_instruction_tuning/avsd/Charades_vu17_test_480")
    parser.add_argument("--feature-folder", type=str, default="./data/video_instruction_tuning/avsd/Charades_vu17_test_audio_imagebind_feat")
    parser.add_argument("--annotation-file", type=str, default="./data/video_instruction_tuning/avsd/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json")
    parser.add_argument("--pred-save", type=str, default="./data/video_instruction_tuning/prediction/pave_prediction_result.json")
    
    args = parser.parse_args()
    eval_model(args)