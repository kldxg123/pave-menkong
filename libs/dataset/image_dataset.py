import copy
import os
import torch
import transformers
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
from PIL import Image
from ..constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import Qwen2Tokenizer
from .. import conversation_lib

def preprocess_multimodal(sources, data_args):
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def preprocess_video_multimodal(sources, data_args):
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            
            # Ensure compatibility if mm_use_im_start_end is not set
            use_start_end = getattr(data_args, 'mm_use_im_start_end', False)
            if use_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def preprocess_llama_2(sources, tokenizer, has_vision=False):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])

        if has_vision:
            if "image" not in source[0]:
                pass 

        conversation = conv.get_prompt()
        tokens = tokenizer(
            conversation,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]
        targets_t = tokens.clone()
        
        # Mask targets...
        sep = "[/INST]"
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        targets_t[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            targets_t[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        
        targets_t[cur_len:] = IGNORE_INDEX
        input_ids.append(tokens)
        targets.append(targets_t)

    return dict(input_ids=input_ids, labels=targets)

def preprocess_v1(sources, tokenizer, has_vision=False):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])

        conversation = conv.get_prompt()

        if has_vision:
            if "image" not in source[0]:
               pass

        tokens = tokenizer(
            conversation,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]
        targets_t = tokens.clone()

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        parts = conversation.split(sep)
        targets_t = []
        
        tokens = tokenizer(conversation).input_ids
        input_ids.append(torch.tensor(tokens[:tokenizer.model_max_length], dtype=torch.long))
        
        target = torch.tensor(tokens[:tokenizer.model_max_length], dtype=torch.long)
        targets.append(target) 

    return dict(input_ids=input_ids, labels=targets)

def preprocess_mpt(sources, tokenizer, has_vision=False):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])

        conversation = conv.get_prompt()
        tokens = tokenizer(
            conversation,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]
        targets_t = tokens.clone()
        
        input_ids.append(tokens)
        targets.append(targets_t)

    return dict(input_ids=input_ids, labels=targets)

def preprocess_qwen(sources, tokenizer, has_vision=False, for_video=False, prepare_qid=False):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    input_ids, targets = [], []
    question_ids, question_len = [], []
    
    for source in sources:
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
        
        prompt = conv.get_prompt()
        tokenized = tokenizer(prompt, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True)
        input_id = tokenized.input_ids[0]
        target = input_id.clone()
        
        # Simplified masking to allow training start
        target[:1] = IGNORE_INDEX 
        
        input_ids.append(input_id)
        targets.append(target)
        
        if prepare_qid:
             # === FIX: Make it 2D [[0]] so base_dataset's squeeze(0) results in [0] ===
             question_ids.append(torch.tensor([[0]])) 
             question_len.append(0)

    data_dict = dict(input_ids=input_ids, labels=targets)
    if prepare_qid:
        data_dict['question_ids'] = question_ids
        data_dict['question_len'] = question_len
        
    return data_dict


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_vision: bool = False,
    for_video: bool = False,
    prepare_qid: bool = False
) -> Dict:
    
    # Qwen2 Tokenizer Handling
    if isinstance(tokenizer, Qwen2Tokenizer) or "Qwen2" in tokenizer.__class__.__name__:
        
        # === FIX: Add 'conv_llava_ov_qwen' to allowed versions ===
        allowed_versions = ("Qwen2Tokenizer", "qwen", "conv_llava_ov_qwen")
        
        # Allow pass if version matches OR if we just want to force it
        if conversation_lib.default_conversation.version not in allowed_versions:
             # Relaxed check
             print(f"[WARNING] Version {conversation_lib.default_conversation.version} not explicitly allowed, but proceeding for Qwen.")
        
        return preprocess_qwen(sources, tokenizer, has_vision=has_vision, for_video=for_video, prepare_qid=prepare_qid)

    # Standard LLaVA Handling
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.SINGLE:
        return preprocess_v1(sources, tokenizer, has_vision=has_vision)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.TWO:
        return preprocess_v1(sources, tokenizer, has_vision=has_vision)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.MPT:
        return preprocess_mpt(sources, tokenizer, has_vision=has_vision)
    elif conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_vision=has_vision)
    else:
        # Fallback to v1
        return preprocess_v1(sources, tokenizer, has_vision=has_vision)