# T2I-Metrics--这是一个关于Text-to-Image中Metrics的Pytorch集成pipeline代码库
## 0. 项目介绍

近年来diffusion models的发展十分迅速，但本人发现目前关于diffusion models的评价指标并没有良好的集成.因此本人参考了市面上一些比较标准的计算diffusion metrics的代码, 自己搭建了一个关于diffusion models若干评价指标的集成pipeline代码库。欢迎各位star + fork。

后续还会更新其他的一些指标，以及tensorflow的集成pipeline，可能也会增加T2V系列, 敬请期待！！！

## 1. 环境配置
#### 1.1 利用requirement.txt文件进行安装
```
pip install -r requirements.txt
```
#### 1.2 利用environment.yaml文件进行安装
```
conda env create -f environment.yaml
```
#### 1.3 利用pip命令安装
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

## 2. 数据准备
- 关于IS Value的数据格式
```
├── path/to/image
│   ├── cat.png
│   ├── dog.png
│   └── bird.jpg
```
- 关于FID Value的数据格式
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
- 关于CLIP Score的数据格式
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

## 3. 快速开始
我们提供了一个简单的脚本，用于快速计算关于diffusion models若干指标的集成pipeline。

```
bash scripts/start.sh
```

您也可以直接在命令行运行如下的命令进行metrics的计算

```
# for img-txt
python ./cal_diffusion_metric.py  --cal_IS True --cal_FID True --cal_CLIP True \
    --path1 ./examples/imgs1 --path2 ./examples/imgs2 \
    --real_path ./examples/imgs1 --fake_path ./examples/prompt
# for jsonl
python ./cal_diffusion_metric.py  --cal_IS True --cal_FID True --cal_CLIP True \
    --path1 ./examples/imgs1 --path2 ./examples/imgs2 \
    --jsonl_path .examples/img-txt.jsonl # for img-txt
```

其中，--cal_IS 表示是否计算IS， 默认为True。 --cal_FID 表示是否计算FID， 默认为True。 --cal_CLIP 表示是否计算CLIP， 默认为True。

其中，--path1 表示计算FID时的生成图像的路径，--path2 表示计算FID时的真实图像的路径。计算IS会默认采用--path1。

其中，--real_path 表示计算clip score时使用的真实图像的路径, --fake_path 表示计算clip score时使用的文字的路径。 也支持传入单个--jsonl_path, jsonl格式具有优先级。
## 4. 参考来源

[IS Value参考链接](https://github.com/sbarratt/inception-score-pytorch/tree/master)

[FID Value参考链接](https://github.com/mseitzer/pytorch-fid)

[CLIP Score参考链接](https://github.com/Taited/clip-score)