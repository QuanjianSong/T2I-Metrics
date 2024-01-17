"""Calculates the CLIP Scores

The CLIP model is a contrasitively learned language-image model. There is
an image encoder and a text encoder. It is believed that the CLIP model could 
measure the similarity of cross modalities. Please find more information from 
https://github.com/openai/CLIP.

The CLIP Score measures the Cosine Similarity between two embedded features.
This repository utilizes the pretrained CLIP Model to calculate 
the mean average of cosine similarities. 

See --help to see further details.

Code apapted from https://github.com/mseitzer/pytorch-fid and https://github.com/openai/CLIP.

Copyright 2023 The Hong Kong Polytechnic University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import os.path as osp
from parser_config.get_parser import get_basic_parser, get_clip_score_parser, merged_parser

import clip
import torch
from torch.utils.data import DataLoader


try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from utils.metrics_utils import DummyDataset


@torch.no_grad()
def calculate_clip_score(dataloader, model, real_flag, fake_flag):
    score_acc = 0.
    sample_num = 0.
    logit_scale = model.logit_scale.exp()
    for batch_data in tqdm(dataloader, desc="Processing:"):
        real = batch_data['real']
        real_features = forward_modality(model, real, real_flag)
        fake = batch_data['fake']
        fake_features = forward_modality(model, fake, fake_flag)
        
        # normalize features
        real_features = real_features / real_features.norm(dim=1, keepdim=True).to(torch.float32)
        fake_features = fake_features / fake_features.norm(dim=1, keepdim=True).to(torch.float32)
        
        # calculate scores
        # score = logit_scale * real_features @ fake_features.t()
        # score_acc += torch.diag(score).sum()
        score = logit_scale * (fake_features * real_features).sum()
        score_acc += score
        sample_num += real.shape[0]
    
    return score_acc / sample_num

def forward_modality(model, data, flag):
    device = next(model.parameters()).device
    if flag == 'img':
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    return features

def main(args):
    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    clip_score = cal_clip_score(clip_model=args.clip_model, batch_size=args.batch_size, device=device,
                    num_workers=num_workers, real_path=args.real_path, fake_path=args.fake_path, 
                    real_flag=args.real_flag, fake_flag=args.fake_flag)

    return clip_score

def cal_clip_score(clip_model, batch_size, device, num_workers, real_path, fake_path,
                    real_flag, fake_flag):
    print('Loading CLIP model: {}'.format(clip_model))
    model, preprocess = clip.load(clip_model, device=device)
    dataset = DummyDataset(real_path, fake_path,
                            real_flag, fake_flag,
                            transform=preprocess, tokenizer=clip.tokenize)
    dataloader = DataLoader(dataset, batch_size, 
                            num_workers=num_workers, pin_memory=True)
    clip_score = calculate_clip_score(dataloader, model,
                                        real_flag, fake_flag)
    clip_score = clip_score.cpu().item()
    
    return clip_score


if __name__ == '__main__':
    basic_parser = get_basic_parser()
    clip_score_parser = get_clip_score_parser()
    args = merged_parser(*[basic_parser, clip_score_parser])

    main(args)


