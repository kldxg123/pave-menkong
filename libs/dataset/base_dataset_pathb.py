# /home/app-ahr/PAVE/libs/dataset/base_dataset_pathb.py

import os
import copy
import torch
from typing import Dict, List, Sequence
from PIL import Image

# 从原始文件中导入所有需要的函数和类
from .base_dataset import (
    LazySupervisedVideoDataset, DataCollatorForSupervisedVideoDataset,
    load_annotation_and_filter, preprocess_video_multimodal, preprocess,
    load_audio_feature, process_video_with_decord, get_transforms_video
)
from ..constants import IGNORE_INDEX
from ..utils.train_utils import DataArguments


class LazySupervisedVideoDatasetPathB(LazySupervisedVideoDataset):
    """
    Path B 专用数据集：
    - 慢路径: 加载原始视频帧
    - 快路径: 加载预计算的音频特征
    """

    def __init__(self, anno_path: str, tokenizer, data_args: DataArguments):
        # 强制设置标志位，绕过父类的检查
        data_args.use_slow = True
        data_args.use_slow_feat = False
        data_args.use_fast = False
        data_args.use_fast_feat = True
        data_args.fast_feat_type = 'audio'  # 明确指定快路径是音频

        # 调用父类初始化
        super().__init__(
            anno_path=anno_path,
            fast_path_mapping_path=data_args.fast_path_mapping_path,  # 音频特征映射
            tokenizer=tokenizer,
            data_args=data_args
        )
        print(f"[PathB Dataset] Initialized with {len(self.list_data_dict)} samples.")
        print(f"[PathB Dataset] Slow video root: {data_args.slow_path_data_root}")
        print(f"[PathB Dataset] Fast audio root: {data_args.data_root}")

    def _get_item(self, i: int) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        # 1. 加载慢路径：原始视频帧
        video_file_path = sources[0]['video_path']
        print(f"[_get_item] Loading raw video frames from: {video_file_path}")
        try:
            # 使用decord加载视频帧
            video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file_path,
                                                                                            self.data_args)
            processor = self.data_args.image_processor
            image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
            image = [(image, video[0].size, "video")]
        except Exception as e:
            print(f"Error processing video {video_file_path}: {e}")
            # 如果失败，返回下一个样本，这是一种健壮性处理
            return self._get_item((i + 1) % len(self.list_data_dict))

        # 2. 加载快路径：预计算的音频特征
        audio_feat_file_path = sources[0]['feat_path']
        print(f"[_get_item] Loading audio feature from: {audio_feat_file_path}")
        # 使用现有的load_audio_feature函数
        # 它会返回 [C, T, 1, 1] 格式的特征
        audio_feat, audio_feat_fps, feat_frame_num = load_audio_feature(audio_feat_file_path)

        # 3. 处理文本
        sources = preprocess_video_multimodal(
            copy.deepcopy([e["conversations"] for e in sources]),
            self.data_args
        )

        old_data_dict = preprocess(
            sources,
            self.tokenizer,
            has_vision=('video' in self.list_data_dict[i]),
            for_video=True,
            prepare_qid=self.prepare_qid
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=old_data_dict["input_ids"][0],
                             labels=old_data_dict["labels"][0])

        # 4. 将所有信息打包
        data_dict['image'] = image  # 原始视频帧
        data_dict['video_feat'] = audio_feat  # 音频特征
        data_dict['video_feat_fps'] = audio_feat_fps
        data_dict['feat_frame_num'] = feat_frame_num
        data_dict['video_meta'] = self.list_data_dict[i]

        return data_dict


def make_video_supervised_data_module_pathb(tokenizer, data_args: DataArguments) -> Dict:
    """Make dataset and collator for PathB supervised fine-tuning."""
    train_dataset = LazySupervisedVideoDatasetPathB(
        tokenizer=tokenizer,
        anno_path=data_args.annotation_path,
        data_args=data_args
    )
    # DataCollatorForSupervisedVideoDataset 应该可以直接复用
    # 因为它已经处理了 'image' 和 'video_feat' 的padding
    data_collator = DataCollatorForSupervisedVideoDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

