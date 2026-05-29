"""
RealWorldQA Dataset Utilities

Data loading and processing utilities, fully independent of VLMEvalKit.
"""

import os
import pandas as pd
import numpy as np
import string
from typing import Dict, Any, List
from PIL import Image
from common_utils import download_file, md5, toliststr, decode_base64_to_image_file

# RealWorldQA dataset URL and MD5
REALWORLDQA_DATASET_URL = 'https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv'
REALWORLDQA_DATASET_MD5 = '92321028d2bc29040284b6674721e48f'

def load_dataset(dataset_name='RealWorldQA'):
    """
    Load RealWorldQA dataset.
    
    Args:
        dataset_name: Dataset name (default: 'RealWorldQA')
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if 'LMUData' not in os.environ:
        raise ValueError("Please set LMUData environment variable or use --data-dir argument")
    data_root = os.path.join(os.environ['LMUData'])
    os.makedirs(data_root, exist_ok=True)
    
    file_name = f"{dataset_name}.tsv"
    data_path = os.path.join(data_root, file_name)
    
    # Download dataset if not exists or MD5 mismatch
    if not os.path.exists(data_path) or md5(data_path) != REALWORLDQA_DATASET_MD5:
        print(f"Downloading {dataset_name} dataset...")
        download_file(REALWORLDQA_DATASET_URL, data_path)
    
    # Load dataset
    data = pd.read_csv(data_path, sep='\t')
    
    # Process dataset
    data['index'] = [str(x) for x in data['index']]
    
    # Process image data (base64 encoded or referenced)
    if 'image' in data:
        data['image'] = [str(x) for x in data['image']]
        image_map = {x: y for x, y in zip(data['index'], data['image'])}
        
        # Process image references (some images may reference other indices)
        for k in image_map:
            if len(image_map[k]) <= 64:
                idx = image_map[k]
                assert idx in image_map and len(image_map[idx]) > 64
                image_map[k] = image_map[idx]

        images = [toliststr(image_map[k]) for k in data['index']]
        data['image'] = [x[0] if len(x) == 1 else x for x in images]
    
    # Process image paths
    if 'image_path' in data:
        paths = [toliststr(x) for x in data['image_path']]
        data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]
    
    # Convert index to integer if possible
    if np.all([isinstance(x, int) or (isinstance(x, str) and x.isdigit()) for x in data['index']]):
        data['index'] = [int(x) for x in data['index']]
    
    return data

def dump_image(line, img_root):
    """
    Save image data to disk and return path.
    
    Args:
        line: Data row containing image data
        img_root: Image save root directory
    
    Returns:
        list: List of image paths
    """
    os.makedirs(img_root, exist_ok=True)
    
    if 'image' in line:
        if isinstance(line['image'], list):
            tgt_path = []
            assert 'image_path' in line
            for img, im_name in zip(line['image'], line['image_path']):
                path = os.path.join(img_root, im_name)
                if not os.path.exists(path):
                    decode_base64_to_image_file(img, path)
                tgt_path.append(path)
        else:
            tgt_path = os.path.join(img_root, f"{line['index']}.jpg")
            if not os.path.exists(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path = [tgt_path]
    else:
        assert 'image_path' in line
        tgt_path = toliststr(line['image_path'])
    
    return tgt_path

def build_realworldqa_prompt(line, dump_image_func, min_pixels, max_pixels):
    """
    Build RealWorldQA dataset prompt.
    
    Args:
        line: Data row
        dump_image_func: Image save function
        min_pixels: Minimum pixels
        max_pixels: Maximum pixels
    
    Returns:
        list: List of messages in standard conversation format
    """
    # Save and get image path
    tgt_path = dump_image_func(line)
    
    # Build question text
    question = line['question']
    
    # Build options
    options = {
        cand: line[cand]
        for cand in string.ascii_uppercase
        if cand in line and not pd.isna(line[cand])
    }
    
    options_prompt = 'Options:\n'
    for key, item in options.items():
        options_prompt += f'{key}. {item}\n'
    
    # Process hint if exists
    hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
    
    # Build complete prompt
    prompt = ''
    if hint is not None:
        prompt += f'Hint: {hint}\n'
    prompt += f'Question: {question}\n'
    if len(options):
        prompt += options_prompt
        prompt += 'Please select the correct answer from the options above. \n'
    
    # Build messages in standard conversation format
    content = []
    
    # Add images (using file:// prefix for consistency)
    if isinstance(tgt_path, list):
        for p in tgt_path:
            content.append({
                "type": "image",
                "image": f"file://{p}",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels
            })
    else:
        content.append({
            "type": "image", 
            "image": f"file://{tgt_path}",
            "min_pixels": min_pixels,
            "max_pixels": max_pixels
        })
    
    # Add text
    content.append({"type": "text", "text": prompt})
    
    # Return messages in standard conversation format
    messages = [{
        "role": "user",
        "content": content
    }]
    
    return messages

