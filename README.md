# T2I-Metrics--This is a Pytorch-integrated pipeline codebase for Metrics in Text-to-Image. [To refer to the Chinese introduction, please click on this link.](https://github.com/QuanjianSong/T2I-Metrics/blob/main/README_cn.md)

# News
- 🔥 We’ve added new evaluation metrics in [AutoMetrics](https://github.com/QuanjianSong/AutoMetrics), including DINO score, DreaSim score, and aesthetic metrics. 

## 0. Projects Introduction

In recent years, the development of diffusion models is very rapid, but I found that the current evaluation metrics on diffusion models are not well integrated. Therefore, I refer to the market for some of the more standard code for calculating diffusion metrics, and built a pipeline code base for integrating several evaluation metrics of diffusion models. Welcome to star + fork.

We will also update some other metrics, and tensorflow integration pipeline, may also add T2V series, please look forward to!

## 1. Environment Configuration

#### 1.1 Installation with the requirement.txt

```
pip install -r requirements.txt
```

#### 1.2 Installation with environment.yaml

```
conda env create -f environment.yaml
```

#### 1.3 Installation with the pip command

- Install PyTorch:

```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116  # Choose a version that suits your GPU
```

- Install Scipy

```
pip install scipy
```

- Install CLIP:

```
pip install git+https://github.com/openai/CLIP.git
```

## 2. Model Weights Download

You need to download the inception_v3_google.pth, pt_inception.pth, and ViT-B-32.pt weights files and place them in the checkpoints folder. We have integrated them into the following links for your convenience.

[Baidu cloud disk link, extraction code: fpfp](https://pan.baidu.com/s/1nGPq5y2OfCumMQkY6ROKGA?)

## 3. Data Preparation

- About the data format of IS Value

```
├── path/to/image
│   ├── cat.png
│   ├── dog.png
│   └── bird.jpg
```

- About the data format of FID Value

```
├── path/to/image
│   ├── cat1.png
│   ├── dog1.png
│   └── bird1.jpg
├── path/to/image
│   ├── cat2.png
│   ├── dog2.png
│   └── bird2.jpg
```

- About the CLIP Score data format

```
├── path/to/image
│   ├── cat.png
│   ├── dog.png
│   └── bird.jpg
└── path/to/text
    ├── cat.txt
    ├── dog.txt
    └── bird.txt
```

OR

```
├── path/to/jsonl
│   ├── {"real_path": cat.png, "fake_path": cat.txt or prompt}
│   ├── {"real_path": dog.png, "fake_path": dog.txt or prompt}
│   └── {"real_path": bird.png, "fake_path": bird.txt or prompt}
```

## 4. Quick Start

We provide a simple script for quickly computing an integrated pipeline on several metrics of diffusion models.

```
bash scripts/start.sh
```

You can also run the following command directly from the command line to calculate metrics.

```
# for img-txt
python ./cal_diffusion_metric.py  --cal_IS --cal_FID --cal_CLIP \
    --path1 ./examples/imgs1 --path2 ./examples/imgs2 \
    --real_path ./examples/imgs1 --fake_path ./examples/prompt
# for jsonl
python ./cal_diffusion_metric.py  --cal_IS --cal_FID --cal_CLIP \
    --path1 ./examples/imgs1 --path2 ./examples/imgs2 \
    --jsonl_path .examples/img-txt.jsonl # for img-txt
```

where --cal_IS indicates whether to calculate IS, --cal_FID indicates whether to calculate FID, and --cal_CLIP indicates whether to calculate CLIP.

Where --path1 denotes the path of the generated image when calculating FID, and --path2 denotes the path of the real image when calculating FID. Calculate IS will use --path1 by default.

Where --real_path denotes the path to the real image used to compute the clip score, and --fake_path denotes the path to the text used to compute the clip score. Passing in a single --jsonl_path is also supported, with the jsonl format taking precedence.

## 5. Reference Source

[IS Value reference link](https://github.com/sbarratt/inception-score-pytorch/tree/master)

[FID Value reference link](https://github.com/mseitzer/pytorch-fid)

[CLIP Score Reference Link](https://github.com/Taited/clip-score)
