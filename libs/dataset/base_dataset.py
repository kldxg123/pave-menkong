# This script contains all the implementation for the training dataset.

import os
import copy
from dataclasses import dataclass, field
import json
import time
import signal  # === ADDED: For timeout handling ===
from typing import Dict, List, Optional, Tuple, Union, Sequence
from PIL import Image
import numpy as np
import math
import random
from copy import deepcopy

from einops import rearrange, repeat
import torch
from torch.utils.data import Dataset
import transformers

from ..constants import IGNORE_INDEX, VID_EXTENSIONS, IMG_EXTENSIONS
from ..utils.train_utils import DataArguments, rank0_print
from .image_dataset import preprocess_multimodal, preprocess, preprocess_qwen, preprocess_video_multimodal
from .video_loading_utils import read_video, temporal_random_crop, get_transforms_video, fps_base_temporal_sampling, \
    frame_base_temporal_sampling, process_video_with_decord, process_video_with_tv
from .dataset_utils import load_data


# === ADDED: Timeout Handler ===
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Video loading took too long!")


def load_images_from_folder(folder_path):
    # try loading the npy files
    npy_file_path = os.path.join(folder_path, 'stacked_images.npy')
    if os.path.exists(npy_file_path):
        images_array = np.load(npy_file_path)
        return images_array

    # if the npy files not exist, list all the files
    all_files_under_folder = os.listdir(folder_path)

    # filter and sort the file along the temporal
    jpg_files = sorted(
        [f for f in all_files_under_folder if f.endswith('.jpg')],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    # load the image
    images = []
    for filename in jpg_files:
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("RGB")  # Ensure all images are RGB
        img_array = np.array(img)
        images.append(img_array)

    # handle the special case
    if len(images) > 32:
        # print(folder_path, len(images))
        images = images[:32]
    elif len(images) < 32:
        gap = 32 - len(images)
        # print(folder_path, len(images))
        images = images + [images[-1]] * gap

    # stack the image
    if images:
        # Convert list of images to a 4D NumPy array
        images_array = np.stack(images, axis=0)  # Shape: (number_of_images, H, W, C)
        # save the npy file
        save_file_name = os.path.join(folder_path, "stacked_images.npy")
        # print('stacked numpy array is saved to:', save_file_name)
        np.save(save_file_name, images_array)
        return images_array
    else:
        print("No JPG images found in the specified folder.")
        return None


def load_dense_frame_feature(video_feat_file_path, exclude_languagebind_cls_token):
    # load the feature
    video_feat = torch.load(video_feat_file_path, map_location=torch.device('cpu'))
    # exclude the cls tokens
    if exclude_languagebind_cls_token:
        video_feat = video_feat[:, 1:,]
    # reshape
    S = video_feat.shape[1]
    assert int(math.sqrt(S)) ** 2 == S  # assert is a square
    W = H = int(math.sqrt(S))
    video_feat = rearrange(video_feat, 't (h w) c -> c t h w', h=H)
    video_feat_fps = 2
    feat_frame_num = video_feat.shape[1]

    return video_feat, video_feat_fps, feat_frame_num


def load_exo_feature(fast_feat_type, video_feat_file_path):
    video_folder = video_feat_file_path
    all_files = os.listdir(video_folder)

    if fast_feat_type == 'exo_random':
        random_num = random.randint(1, len(all_files))
        all_files = random.sample(all_files, random_num)

    video_feat = []
    for curr_pt_file in all_files:
        curr_feature = torch.load(os.path.join(video_folder, curr_pt_file)).unsqueeze(dim=0)
        video_feat.append(curr_feature)

    video_feat = torch.cat(video_feat, dim=0)
    V, T, S, D = video_feat.shape
    video_feat = video_feat.permute([1, 0, 2, 3]).reshape(-1, S, D)
    video_feat = video_feat.view(-1, 14, 14, D).permute([3, 0, 1, 2])

    video_feat_fps = 1
    feat_frame_num = video_feat.shape[1]

    return video_feat, video_feat_fps, feat_frame_num


def load_audio_feature(video_feat_file_path):
    video_feat = torch.load(video_feat_file_path, map_location=torch.device('cpu'))
    if len(video_feat.shape) == 3:
        feat_dim = video_feat.shape[-1]
        video_feat = video_feat.view(-1, feat_dim)
    if video_feat.requires_grad:
        video_feat.requires_grad = False

    video_feat = video_feat.permute([1, 0])
    video_feat = video_feat.unsqueeze(dim=-1).unsqueeze(dim=-1)
    video_feat_fps = 100 / 16
    feat_frame_num = video_feat.shape[1]

    return video_feat, video_feat_fps, feat_frame_num


class LazySupervisedVideoDataset(Dataset):
    """Dataset for supervised Video fine-tuning."""

    def __init__(self,
                 anno_path: str,
                 fast_path_mapping_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedVideoDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_root = data_args.data_root
        self.data_args = data_args
        self.prepare_qid = data_args.prepare_qid
        self.data_sample_ratio = data_args.data_sample_ratio

        self.use_fast = data_args.use_fast
        self.use_fast_feat = data_args.use_fast_feat

        self.use_slow = data_args.use_slow
        self.use_slow_feat = data_args.use_slow_feat
        slow_path_mapping_path = data_args.slow_path_mapping_path
        self.slow_path_data_root = data_args.slow_path_data_root

        self.fast_feat_type = data_args.fast_feat_type
        self.exclude_languagebind_cls_token = data_args.exclude_languagebind_cls_token

        self.video_loading_backbone = data_args.video_loading_backbone

        self.use_second_sides = data_args.use_second_sides
        self.second_sides_type = data_args.second_sides_type
        self.second_sides_data_root = data_args.second_sides_data_root

        assert (self.data_args.use_fast == True and self.data_args.use_fast_feat == False) or \
               (self.data_args.use_fast == False and self.data_args.use_fast_feat == True) or \
               (self.data_args.use_fast == False and self.data_args.use_fast_feat == False)
        assert (self.data_args.use_slow == True and self.data_args.use_slow_feat == False) or \
               (self.data_args.use_slow == False and self.data_args.use_slow_feat == True) or \
               (self.data_args.use_slow == False and self.data_args.use_slow_feat == False)

        if self.data_sample_ratio is not None:
            self.data_sample_ratio = self.data_sample_ratio.split(',')
            self.data_sample_ratio = [float(ele) for ele in self.data_sample_ratio]
            assert len(self.data_sample_ratio) == len(slow_path_mapping_path)

        prefilter_video_ids = ['009151_009200/1057949419', '077251_077300/14911927', '00013654']

        if isinstance(anno_path, list):
            assert isinstance(fast_path_mapping_path, list)
            assert isinstance(self.data_root, list)

            if self.use_slow or self.use_slow_feat:
                assert isinstance(slow_path_mapping_path, list)
                assert isinstance(self.slow_path_data_root, list)
                assert len(anno_path) == len(fast_path_mapping_path) == len(self.data_root) == len(
                    slow_path_mapping_path) == len(self.slow_path_data_root)

                all_filtered_anno = []
                for i, (curr_anno_path, curr_fast_path_mapping_path, curr_data_root, curr_slow_path_mapping_path,
                        curr_slow_path_data_root) in \
                        enumerate(zip(anno_path, fast_path_mapping_path, self.data_root, slow_path_mapping_path,
                                      self.slow_path_data_root)):
                    curr_anno = load_annotation_and_filter(curr_anno_path, curr_fast_path_mapping_path, curr_data_root,
                                                           prefilter_video_ids=prefilter_video_ids,
                                                           slow_path_mapping_path=curr_slow_path_mapping_path,
                                                           slow_path_data_root=curr_slow_path_data_root,
                                                           second_side_channels_root=self.second_sides_data_root if self.use_second_sides else None)
                    if self.data_sample_ratio is not None:
                        curr_ratio = self.data_sample_ratio[i]
                        curr_selected_len = int(curr_ratio * len(curr_anno))
                        curr_anno = curr_anno[:curr_selected_len]
                        print(curr_anno_path, 'sample ratio:', curr_ratio, 'remaining len:', len(curr_anno))
                    all_filtered_anno += curr_anno
            else:
                assert len(anno_path) == len(fast_path_mapping_path) == len(self.data_root)

                all_filtered_anno = []
                for i, (curr_anno_path, curr_fast_path_mapping_path, curr_data_root) in enumerate(
                        zip(anno_path, fast_path_mapping_path, self.data_root)):
                    curr_anno = load_annotation_and_filter(curr_anno_path, curr_fast_path_mapping_path, curr_data_root,
                                                           prefilter_video_ids=prefilter_video_ids)
                    if self.data_sample_ratio is not None:
                        curr_ratio = self.data_sample_ratio[i]
                        curr_selected_len = int(curr_ratio * len(curr_anno))
                        curr_anno = curr_anno[:curr_selected_len]
                        print(curr_anno_path, 'sample ratio:', curr_ratio, 'remaining len:', len(curr_anno))
                    all_filtered_anno += curr_anno
        else:
            all_filtered_anno = load_annotation_and_filter(anno_path, fast_path_mapping_path, self.data_root,
                                                           prefilter_video_ids=prefilter_video_ids,
                                                           slow_path_mapping_path=slow_path_mapping_path,
                                                           slow_path_data_root=self.slow_path_data_root,
                                                           second_side_channels_root=self.second_sides_data_root if self.use_second_sides else None)

        self.list_data_dict = all_filtered_anno

        self.transforms = {
            "video": get_transforms_video(self.data_args.transform_name, self.data_args.image_size),
        }

        if self.use_fast_feat:
            self.original_feat_fps = self.data_args.original_feat_fps
            self.training_feat_fps = self.data_args.training_feat_fps

    def __len__(self):
        return len(self.list_data_dict)

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3

        # === DEBUG: Print first 10 items or every 500th ===
        if i < 10 or i % 500 == 0:
            print(f"[DEBUG-Dataset] Fetching index {i}...", flush=True)

        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except TimeoutException as te:
                print(f"[TIMEOUT] Reading data at index {i} took too long (>60s). Skipping...", flush=True)
                # Fallback to random sample to keep batch size
                # In distributed training, this might cause data mismatch if seed not synced, but better than hang.
                # Ideally we skip, but collator expects data. Let's try next index.
                i = (i + 1) % len(self.list_data_dict)
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception: {e}", flush=True)
                time.sleep(1)

        # Fatal if retry failed
        print(f"[FATAL] Could not fetch sample {i}. Returning next item...", flush=True)
        return self.__getitem__((i + 1) % len(self.list_data_dict))

    def _get_item(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if 'video' in sources[0]:
            if self.use_fast:
                raise NotImplementedError
            elif self.use_fast_feat:
                video_feat_file_path = self.list_data_dict[i]['feat_path']
                if self.fast_feat_type == 'languagebind' or self.fast_feat_type == 'languagebind_14x14' or self.fast_feat_type == 'internvideo2' or self.fast_feat_type == 'siglip':
                    video_feat, video_feat_fps, feat_frame_num = load_dense_frame_feature(video_feat_file_path,
                                                                                          self.exclude_languagebind_cls_token)
                elif self.fast_feat_type == 'audio':
                    video_feat, video_feat_fps, feat_frame_num = load_audio_feature(video_feat_file_path)
                elif self.fast_feat_type == '3d_feature':
                    video_feat = torch.load(video_feat_file_path, map_location=torch.device('cpu'))
                    B, _, D = video_feat.shape
                    V, H, W = 32, 24, 24
                    video_feat = video_feat.view(B, V, H, W, D).squeeze(dim=0)
                    video_feat = video_feat.permute([3, 0, 1, 2])
                    video_feat_fps = 1
                    feat_frame_num = video_feat.shape[1]
                elif self.fast_feat_type == 'exo' or self.fast_feat_type == 'exo_random':
                    video_feat, video_feat_fps, feat_frame_num = load_exo_feature(self.fast_feat_type,
                                                                                  video_feat_file_path)
                else:
                    raise NotImplementedError

            if self.use_slow:
                video_file_path = self.list_data_dict[i]['video_path']

                # === WATCHDOG: Set 60s timeout for video reading ===
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)  # 60 seconds limit

                try:
                    # Log potentially problematic files
                    # print(f"[DEBUG-IO] Reading video: {video_file_path}", flush=True)

                    if os.path.isdir(video_file_path):
                        video = load_images_from_folder(video_file_path)
                        # Fix attribute error
                        if hasattr(self.data_args, 'image_processor'):
                            processor = self.data_args.image_processor
                        else:
                            # Attempt to grab it globally if missing (Hack fix)
                            # Assuming CLIP/SigLip processor
                            from transformers import CLIPImageProcessor
                            processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

                        image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                        image = [(image, 100, "video")]
                    else:
                        if self.video_loading_backbone == 'decord':
                            video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(
                                video_file_path, self.data_args)
                        else:
                            video, video_time, frame_time, num_frames_to_sample = process_video_with_tv(video_file_path,
                                                                                                        self.data_args)

                        if hasattr(self.data_args, 'image_processor'):
                            processor = self.data_args.image_processor
                        else:
                            # Fallback
                            from transformers import SiglipImageProcessor
                            processor = SiglipImageProcessor.from_pretrained("./models/siglip-so400m-patch14-384")

                        image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                        image = [(image, video[0].size, "video")]

                except TimeoutException:
                    print(f"\n[TIMEOUT CRITICAL] Video file STUCK: {video_file_path}\n", flush=True)
                    raise
                except Exception as e:
                    print(f"\n[IO ERROR] Failed reading: {video_file_path} Error: {e}\n", flush=True)
                    raise
                finally:
                    signal.alarm(0)  # Disable alarm

            elif self.use_slow_feat:
                video_file_path = self.list_data_dict[i]['video_path']
                image = torch.load(video_file_path)
                image = [(image, -1, "video")]

            sources = preprocess_video_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        old_data_dict = preprocess(
            sources,
            self.tokenizer,
            has_vision=('video' in self.list_data_dict[i]),
            for_video=True,
            prepare_qid=self.prepare_qid)
        if isinstance(i, int):
            data_dict = dict(input_ids=old_data_dict["input_ids"][0],
                             labels=old_data_dict["labels"][0])
            if 'question_ids' in old_data_dict:
                data_dict['question_ids'] = old_data_dict["question_ids"][0]
                data_dict['question_len'] = old_data_dict["question_len"]

        if 'video' in self.list_data_dict[i]:
            if self.use_fast or self.use_fast_feat:
                data_dict['video_feat'] = video_feat
                data_dict['video_feat_fps'] = video_feat_fps
                data_dict['feat_frame_num'] = feat_frame_num

            if self.use_slow or self.use_slow_feat:
                data_dict["image"] = image

            data_dict['video_meta'] = self.list_data_dict[i]

        elif self.data_args.is_multimodal:
            raise NotImplementedError

        if self.use_second_sides:
            assert 'second_side_file_path' in self.list_data_dict[i]
            second_side_file_path = self.list_data_dict[i]['second_side_file_path']
            if self.second_sides_type == 'audio':
                second_feat, second_feat_fps, second_feat_frame_num = load_audio_feature(second_side_file_path)
            elif self.second_sides_type == 'exo' or self.second_sides_type == 'exo_random':
                second_feat, second_feat_fps, second_feat_frame_num = load_exo_feature(self.second_sides_type,
                                                                                       second_side_file_path)
            else:
                raise NotImplementedError

            data_dict['second_feat'] = second_feat
            data_dict['second_feat_fps'] = second_feat_fps
            data_dict['second_feat_frame_num'] = second_feat_frame_num
        return data_dict


@dataclass
class DataCollatorForSupervisedVideoDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'video_feat_fps' in instances[0]:
            batch['video_feat_fps'] = torch.tensor([ele['video_feat_fps'] for ele in instances])

        if 'feat_frame_num' in instances[0]:
            batch['feat_frame_nums'] = torch.tensor([ele['feat_frame_num'] for ele in instances])

        if 'video_feat' in instances[0]:
            video_feats = [instance['video_feat'] for instance in instances]
            if all(x is not None and x.shape == video_feats[0].shape for x in video_feats):
                batch['video_feats'] = torch.stack(video_feats)
            else:
                all_lens = [ele['feat_frame_num'] for ele in instances]
                max_len = max(all_lens)
                C, T, H, W = instances[0]['video_feat'].shape
                padded_tensor = torch.zeros([len(instances), C, max_len, H, W])
                for i, (v, v_len) in enumerate(zip(video_feats, all_lens)):
                    padded_tensor[i][:, :v_len] = v
                batch['video_feats'] = padded_tensor

        if 'second_feat_fps' in instances[0]:
            batch['second_feat_fps'] = torch.tensor([ele['second_feat_fps'] for ele in instances])

        if 'second_feat_frame_num' in instances[0]:
            batch['second_feat_frame_nums'] = torch.tensor([ele['second_feat_frame_num'] for ele in instances])

        if 'second_feat' in instances[0]:
            second_feats = [instance['second_feat'] for instance in instances]
            if all(x is not None and x.shape == second_feats[0].shape for x in second_feats):
                batch['second_feats'] = torch.stack(second_feats)
            else:
                all_lens = [ele['second_feat_frame_num'] for ele in instances]
                max_len = max(all_lens)
                C, T, H, W = instances[0]['second_feat'].shape
                padded_tensor = torch.zeros([len(instances), C, max_len, H, W])
                for i, (v, v_len) in enumerate(zip(second_feats, all_lens)):
                    padded_tensor[i][:, :v_len] = v
                batch['second_feats'] = padded_tensor

        if 'question_ids' in instances[0]:
            # Ensure we are handling 1D or 2D tensors correctly
            question_ids = [
                instance['question_ids'].squeeze(0) if instance['question_ids'].ndim > 1 else instance['question_ids']
                for instance in instances]

            question_ids = torch.nn.utils.rnn.pad_sequence(
                question_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            question_ids = question_ids[:, :self.tokenizer.model_max_length]
            batch['question_ids'] = question_ids
            batch['question_lens'] = torch.tensor([instance['question_len'] for instance in instances])

        batch['video_metas'] = [ele['video_meta'] for ele in instances]

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            batch["images"] = images
        return batch


def make_video_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = LazySupervisedVideoDataset(tokenizer=tokenizer,
                                               anno_path=data_args.annotation_path,
                                               fast_path_mapping_path=data_args.fast_path_mapping_path,
                                               data_args=data_args)
    data_collator = DataCollatorForSupervisedVideoDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def load_annotation_and_filter(anno_path, fast_path_mapping_path, data_root,
                               prefilter_video_ids=None,
                               slow_path_mapping_path=None,
                               slow_path_data_root=None,
                               second_side_channels_root=None):
    list_data_dict = load_data(anno_path)
    feat_path_mapping = json.load(open(fast_path_mapping_path))

    if slow_path_mapping_path is not None:
        video_path_mapping = json.load(open(slow_path_mapping_path))
    else:
        video_path_mapping = None

    filtered_anno_list = []
    remaining_video = []
    video_not_exist = []

    for ele in list_data_dict:
        vid = ele['video']
        if '/' in vid:
            vid = vid.split('/')[-1]
        if 'Scene' in vid:
            vid = vid[:13]
        if '.' in vid and not vid.endswith('.mp4'):
            vid = vid.split('.')[0]
        if prefilter_video_ids is not None and vid in prefilter_video_ids:
            # print(vid, 'in prefilter list, filtered.')
            video_not_exist.append(vid)
            continue
        if vid in feat_path_mapping:
            feat_file_path = os.path.join(data_root, feat_path_mapping[vid])
            if not os.path.exists(feat_file_path):
                video_not_exist.append(vid)
                continue
            ele['feat_path'] = feat_file_path

            if video_path_mapping is not None:
                video_filename = video_path_mapping[vid]
                video_file_path = os.path.join(slow_path_data_root, video_filename)
                if not os.path.exists(video_file_path):
                    video_not_exist.append(vid)
                    continue
                ele['video_path'] = video_file_path

            if second_side_channels_root is not None:
                second_side_file_path = os.path.join(second_side_channels_root, feat_path_mapping[vid])
                if 'ego' in second_side_file_path:
                    second_side_file_path = '/'.join(second_side_file_path.split('/')[:-1])

                if not os.path.exists(second_side_file_path):
                    video_not_exist.append(vid)
                    continue
                ele['second_side_file_path'] = second_side_file_path

            filtered_anno_list.append(ele)
            remaining_video.append(vid)
        else:
            video_not_exist.append(vid)

    print('dataset:', anno_path,
          'total annotation:', len(list_data_dict),
          'remaining anno:', len(filtered_anno_list),
          'existing video:', len(set(remaining_video)),
          'video_not_exist:', len(set(video_not_exist)))

    return filtered_anno_list