import torch.nn.functional as F
from PIL import Image
import json
from tqdm import tqdm
from tools.utils import get_mean
import numpy as np
import pathlib
from torchvision import transforms
import torch
from scipy.stats import entropy
from tools.utils import ImagePathDataset
from tools.fid_util import save_fid_stats, calculate_fid_given_paths


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                        'tif', 'tiff', 'webp'}

def cal_IS(model, path, dims=1000, batch_size=1, splits=5):
    path = pathlib.Path(path)
    files = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in path.glob('*.{}'.format(ext))])
    N = len(files)
    if batch_size > N:
        print(('Warning: batch size is bigger than the data size. '
                'Setting batch size to data size'))
        batch_size = N
    dataset = ImagePathDataset(files,
                               transforms=transforms.Compose([
                                    transforms.Resize(299),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=8)
    device = next(model.parameters()).device
    # Get predictions
    preds = np.zeros((N, dims))
    start_idx = 0
    # for loop
    for batch in tqdm(dataloader, desc="Processing:"):
        batch = batch.to(device)
        pred = model(batch)
        pred = F.softmax(pred, dim=-1).data.cpu().numpy()
        preds[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]
    # Now compute the mean kl-div
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# cal FID
def cal_FID(model, path_list, batch_size=1, dims=2048, num_workers=8, save_stats=False):
    device = next(model.parameters()).device

    if save_stats:
        save_fid_stats(path_list, batch_size, device, dims, num_workers)
        return

    fid_value = calculate_fid_given_paths(
        path_list, batch_size, device, dims, num_workers
    )
    
    return fid_value

# CLIP-T
def cal_CLIP_T(model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            prompt = entry['prompt']
            img_feats = model(img)
            text_feats = model(prompt)
            clip_score = F.cosine_similarity(img_feats, text_feats)
            metric_val_list.append(clip_score) 
    avg_val = get_mean(metric_val_list)

    return avg_val

# CLIP-I
def cal_CLIP_I(model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img1 = Image.open(entry['img_path'])
            img2 = Image.open(entry['img_path2'])
            img_feats = model(img1)
            text_feats = model(img2)
            clip_score = F.cosine_similarity(img_feats, text_feats)
            metric_val_list.append(clip_score) 
    avg_val = get_mean(metric_val_list)

    return avg_val

# DINO
def cal_DINOv2(model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img1 = Image.open(entry['img_path'])
            img2 = Image.open(entry['img_path2'])
            img_feats = model(img1)
            text_feats = model(img2)
            clip_score = F.cosine_similarity(img_feats, text_feats)
            metric_val_list.append(clip_score) 
    avg_val = get_mean(metric_val_list)

    return avg_val
 
# DreamSim
def cal_DreamSim(metric_model, jsonl_path, device="cuda"):
    metric_val_list = []
    preprocess, model_func = metric_model[0], metric_model[1]

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img1 = Image.open(entry['img_path'])
            img2 = Image.open(entry['img_path2'])
            img1 = preprocess(img1).to(device)
            img2 = preprocess(img2).to(device)
            distance = model_func(img1, img2)
            metric_val_list.append(distance)
    avg_val = get_mean(metric_val_list)

    return avg_val

# LPIPS
def cal_LPIPS(metric_model, jsonl_path):
    metric_val_list = []
    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img1 = Image.open(entry['img_path'])
            img2 = Image.open(entry['img_path2'])
            val = metric_model(img1, img2)
            metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)
                
    return avg_val

# LAION_Aes
def cal_LAION_Aes(metric_model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            val = metric_model(img)
            metric_val_list.append(val)        
    avg_val = get_mean(metric_val_list)

    return avg_val

# Q-Align
def cal_Q_Align(metric_model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            val = metric_model(img, task_='aesthetic')
            metric_val_list.append(val)    
    avg_val = get_mean(metric_val_list)

    return avg_val

# Q-Align-IQ
def cal_Q_Align_IQ(metric_model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            val = metric_model(img, task_='quality')
            metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)

    return avg_val

# CLIP-IQA
def cal_CLIP_IQA(metric_model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            val = metric_model(img)
            metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)
    
    return avg_val

# TOPIQ-NR
def cal_TOPIQ_NR(metric_model, jsonl_path):
    metric_val_list = []
    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            val = metric_model(img)
            metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)

    return avg_val

# MANIQA
def cal_MANIQA(metric_model, jsonl_path):
    metric_val_list = []
    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            val = metric_model(img)
            metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)

    return avg_val

# HYPERIQA
def cal_HYPERIQA(metric_model, jsonl_path):
    metric_val_list = []

    with open(jsonl_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file):
            entry = json.loads(line.strip())
            img = Image.open(entry['img_path'])
            val = metric_model(img)
            metric_val_list.append(val)
    avg_val = get_mean(metric_val_list)

    return avg_val


# **************************************************************************


METRIC_CAL_FUNC = {
    "CLIP-I": cal_CLIP_I,
    "CLIP-T": cal_CLIP_T,
    "DINO": cal_DINOv2,
    "DreamSim": cal_DreamSim,
    "LPIPS":  cal_LPIPS,
    "LAION-Aes":  cal_LAION_Aes,
    "Q-Align": cal_Q_Align,
    "Q-Align-IQ": cal_Q_Align_IQ,
    "CLIP-IQA": cal_CLIP_IQA,
    "TOPIQ-NR": cal_TOPIQ_NR,
    "MANIQA": cal_MANIQA,
    "HYPERIQA": cal_HYPERIQA,
}