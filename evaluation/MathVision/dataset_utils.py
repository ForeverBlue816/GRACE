import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from common_utils import download_file, md5, toliststr, decode_base64_to_image_file

# MathVision dataset URLs and MD5
MATHVISION_DATASET_URL = {
    'MathVision': 'https://opencompass.openxlab.space/utils/VLMEval/MathVision.tsv',
    'MathVision_MINI': 'https://opencompass.openxlab.space/utils/VLMEval/MathVision_MINI.tsv'
}

MATHVISION_DATASET_MD5 = {
    'MathVision': '93f6de14f7916e598aa1b7165589831e',
    'MathVision_MINI': '060fe4fa5d868987ce179307bd5f8a33'
}

def load_dataset(dataset_name='MathVision'):
    """Load the MathVision dataset."""
    data_root = os.path.join(os.environ['LMUData'])
    os.makedirs(data_root, exist_ok=True)
    
    file_name = f"{dataset_name}.tsv"
    data_path = os.path.join(data_root, file_name)
    
    # Download if not exists or MD5 doesn't match
    if dataset_name in MATHVISION_DATASET_MD5:
        expected_md5 = MATHVISION_DATASET_MD5[dataset_name]
        if not os.path.exists(data_path) or md5(data_path) != expected_md5:
            print(f"Downloading {dataset_name} dataset...")
            download_file(MATHVISION_DATASET_URL[dataset_name], data_path)
    
    # Load the dataset
    data = pd.read_csv(data_path, sep='\t')
    
    # Process the dataset
    data['index'] = [str(x) for x in data['index']]
    
    # Handle image data
    if 'image' in data:
        data['image'] = [str(x) for x in data['image']]
        image_map = {x: y for x, y in zip(data['index'], data['image'])}
        for k in image_map:
            if len(image_map[k]) <= 64:
                idx = image_map[k]
                assert idx in image_map and len(image_map[idx]) > 64
                image_map[k] = image_map[idx]

        images = [toliststr(image_map[k]) for k in data['index']]
        data['image'] = [x[0] if len(x) == 1 else x for x in images]
    
    # Handle image paths
    if 'image_path' in data:
        paths = [toliststr(x) for x in data['image_path']]
        data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]
    
    # Convert index to int if possible
    if np.all([isinstance(x, int) or x.isdigit() for x in data['index']]):
        data['index'] = [int(x) for x in data['index']]
    
    return data

def dump_image(line, img_root):
    """Save image data to disk and return the path."""
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
