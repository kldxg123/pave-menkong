import os
import json
import torch
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Monkey Patch
# ==========================================
from transformers.generation import utils as generation_utils

def _validate_model_kwargs_skip(self, model_kwargs):
    pass

generation_utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs_skip

# ==========================================
# 2. Config
# ==========================================
CONFIG = {
    "model_path": "./checkpoints/pave_v5_1_3_lora_avsd_7B_imagebind",
    "model_base": "./models/llava-onevision-qwen2-7b-ov",
    "video_folder": "./data/video_instruction_tuning/avsd/Charades_vu17_test_480",
    "feature_folder": "./data/video_instruction_tuning/avsd/Charades_vu17_test_audio_imagebind_feat",
    "annotation_file": "./data/video_instruction_tuning/avsd/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json",
    "pred_save_file": "./data/video_instruction_tuning/prediction/pave_prediction_new.json",
    "num_frames": 32,
    "batch_size": 1,
    "conv_mode": "conv_llava_ov_qwen"
}

from libs.conversation_lib import conv_templates, SeparatorStyle
from libs.mm_utils import tokenizer_vision_token, KeywordsStoppingCriteria
# 确保这里导入没问题，如果 NameError 依然存在，说明文件没覆盖对
from libs.model.language_model.pave_qwen2 import PAVEQwen2Config, PAVEQwen2ForCausalLM
from peft import PeftModel
from libs.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


# ==========================================
# 3. Utils & Dataset
# ==========================================
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vid = item['image_id']
        video_path = os.path.join(self.config["video_folder"], f"{vid}.mp4")

        video_frames = load_video_frames(video_path, self.config["num_frames"]) if os.path.exists(
            video_path) else np.zeros((32, 336, 336, 3), dtype=np.uint8)
        video_tensor = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
        video_input = [video_tensor]

        feat_path = os.path.join(self.config["feature_folder"], f"{vid}.pt")
        audio_feat = torch.load(feat_path) if os.path.exists(feat_path) else torch.zeros((100, 1024),
                                                                                         dtype=torch.float16)
        if audio_feat.requires_grad: audio_feat.requires_grad = False

        return {"vid": vid, "video": video_input, "audio_feature": audio_feat, "conversation": item['dialog']}


# ==========================================
# 4. Model Loading
# ==========================================
def load_model(config_dict):
    print(">>> 1. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config_dict["model_base"], use_fast=False)

    print(">>> 2. Loading Base Model...")
    base_cfg = PAVEQwen2Config.from_pretrained(config_dict["model_base"])
    base_cfg._attn_implementation = "eager"

    model = PAVEQwen2ForCausalLM.from_pretrained(
        config_dict["model_base"], config=base_cfg, device_map="auto", torch_dtype=torch.float16
    )

    print(">>> 3. Patching Config & Init Vision...")
    lora_cfg = PAVEQwen2Config.from_pretrained(config_dict["model_path"])

    if not hasattr(lora_cfg, 'vision_tower'):
        lora_cfg.vision_tower = getattr(lora_cfg, 'mm_vision_tower', "./models/siglip-so400m-patch14-384")
    for k, v in {'mm_vision_select_layer': -2, 'mm_projector_type': 'mlp2x_gelu', 'pretrain_mm_mlp_adapter': None,
                 'mm_use_im_start_end': False}.items():
        if not hasattr(lora_cfg, k): setattr(lora_cfg, k, v)

    model.get_model().initialize_vision_modules(model_args=lora_cfg, fsdp=None)

    non_lora_path = os.path.join(config_dict["model_path"], "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        w = torch.load(non_lora_path, map_location="cpu")
        model.load_state_dict({k.replace("base_model.model.", "").replace("model.", ""): v for k, v in w.items()},
                              strict=False)

    print(">>> 4. Loading LoRA...")
    model = PeftModel.from_pretrained(model, config_dict["model_path"]).merge_and_unload()

    if hasattr(model, 'initialize_vision_tokenizer'):
        model.initialize_vision_tokenizer(lora_cfg, tokenizer=tokenizer)

    from transformers import SiglipImageProcessor
    proc_path = "./models/siglip-so400m-patch14-384" if os.path.exists(
        "./models/siglip-so400m-patch14-384") else "google/siglip-so400m-patch14-384"
    image_processor = SiglipImageProcessor.from_pretrained(proc_path)

    return tokenizer, model, image_processor


# ==========================================
# 5. Inference
# ==========================================
def main():
    tokenizer, model, image_processor = load_model(CONFIG)
    model.eval()

    # 获取物理词表大小 (152064)
    REAL_VOCAB_SIZE = model.get_input_embeddings().weight.shape[0]
    print(f">>> Physical Vocab Size: {REAL_VOCAB_SIZE}")
    
    # 同步 Config
    model.config.vocab_size = REAL_VOCAB_SIZE
    if hasattr(model, 'generation_config'):
        model.generation_config.vocab_size = REAL_VOCAB_SIZE

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
                
                # [Fix] 修复 sequence item 11: expected str instance, NoneType found
                if i < len(sample['conversation']) - 1:
                    conv.append_message(conv.roles[1], turn['answer'][0])
                else:
                    pass

            prompt = conv.get_prompt()
            # 手动添加 Assistant 引导
            prompt += conv.roles[1] + "\n"

            input_ids = tokenizer_vision_token(prompt, tokenizer, DEFAULT_IMAGE_TOKEN, return_tensors='pt').unsqueeze(
                0).cuda()

            # [安全处理] 
            # 1. 仅 clamp 正数，保留负数用于 pave_qwen2.py 处理
            mask_positive = input_ids >= 0
            if mask_positive.any():
                clamped_vals = torch.clamp(input_ids[mask_positive], max=REAL_VOCAB_SIZE - 1)
                input_ids[mask_positive] = clamped_vals
            
            # 2. 检查 Pad Token
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None or pad_token_id >= REAL_VOCAB_SIZE:
                pad_token_id = tokenizer.eos_token_id
            if pad_token_id >= REAL_VOCAB_SIZE:
                pad_token_id = REAL_VOCAB_SIZE - 1 

            attention_mask = input_ids.ne(pad_token_id).long().cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    video_feats=audio_feat,
                    feat_frame_nums=torch.tensor([feat_frame_num]).cuda(),
                    images=video_tensor,
                    image_sizes=[200],
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    do_sample=False,
                    max_new_tokens=512,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if output_text.endswith(stop_str): output_text = output_text[:-len(stop_str)]
            results.append({"image_id": vid, "caption": output_text})

        except Exception as e:
            print(f"[ERROR] Failed to generate for {vid}")
            print(f"Error details: {e}")
            if "CUDA" in str(e):
                raise e

    os.makedirs(os.path.dirname(CONFIG["pred_save_file"]), exist_ok=True)
    with open(CONFIG["pred_save_file"], "w") as f:
        json.dump(results, f, indent=4)
    print("Done.")


if __name__ == "__main__":
    main()