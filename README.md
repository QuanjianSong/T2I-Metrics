<div align="center">
<h1>
T2I-Metrics: A Pipeline for Metrics in Text-to-Image
<br>
[Official Code of PyTorch]
</h1>
</div>


## 🎉 News
<pre>
• <strong>2025.08</strong>: 🔥 The repository has been reorganized and now includes various AIGC metrics: FID, IS, CLIP, DINO, and DreamSim etc.
</pre>

    
## 🎬 Overview
In recent years, the development of diffusion models is very rapid, but I found that the current evaluation metrics on diffusion models are not well integrated. Therefore, I refer to the market for some of the more standard code for calculating diffusion metrics, and built a pipeline code base for integrating several evaluation metrics of diffusion models. Welcome to star + fork.
We will also update some other metrics, and tensorflow integration pipeline, may also add T2V series, please look forward to!

## 🔧 Environment
```
# Git clone the repo
git clone https://github.com/QuanjianSong/T2I-Metrics.git

# Installation with the requirement.txt
conda create -n t2i-metrics python=3.8
conda activate t2i-metrics
pip install -r requirements.txt
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116  # Choose a version that suits your GPU

# Or installation with the environment.yaml
conda env create -f environment.yaml
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116  # Choose a version that suits your GPU
```

## 🤗 Checkpoint

If you need the `inception_v3_google.pth`, `pt_inception.pth`, and `ViT-B-32.pt` weight files, you can download them from the this [link](https://pan.baidu.com/s/1nGPq5y2OfCumMQkY6ROKGA?)(extraction code: fpfp).

## 📖 Dataset
Before starting the evaluation, you need to prepare the corresponding jsonl files in advance. Different evaluation metrics require reading different types of jsonl files. These generally fall into three categories: image-prompt pairs, image-image pairs, and single images. Each line in the jsonl file should include the appropriate file paths. We provide example files in the `./examples` directory to help you construct your own.


## 🚀 Start
We provide simple examples in `main.py` to quickly get started with metric evaluation.
```
python main.py
```
Different evaluation metrics require reading different jsonl files, and we have provided corresponding examples in the previous step.


## 🎓 Bibtex
[IS Value reference link](https://github.com/sbarratt/inception-score-pytorch/tree/master)

[FID Value reference link](https://github.com/mseitzer/pytorch-fid)

