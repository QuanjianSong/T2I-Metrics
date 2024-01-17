# T2I-Metrics--这是一个关于Text-to-Image中Metrics的Pytorch集成pipeline代码库
## 0. 项目介绍

近年来diffusion models的发展十分迅速，但本人发现目前关于diffusion models的评价指标并没有良好的集成.因此本人参考了市面上一些比较标准的计算diffusion metrics的代码, 自己搭建了一个关于diffusion models若干评价指标的集成pipeline代码库。欢迎各位star + fork。

后续还会更新其他的一些指标，以及tensorflow的集成pipeline，可能也会增加T2V系列, 敬请期待！！！

## 1. 环境配置
- Install PyTorch:
```
pip install torch  # Choose a version that suits your GPU
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
## 3. 快速开始
我们提供了一个简单的脚本，用于快速计算关于diffusion models若干指标的集成pipeline。

```
bash scripts/start.sh
```

其中，--cal_IS 表示是否计算IS， 默认为True。 --cal_FID 表示是否计算FID， 默认为True。 --cal_CLIP 表示是否计算CLIP， 默认为True。

其中，--path1 表示计算FID时的生成图像的路径，--path2 表示计算FID时的真实图像的路径。计算IS会默认采用--path1。

其中，--real_path 表示计算clip score时使用的真实图像的路径, --fake_path 表示计算clip score时使用的文字的路径。
## 4. 参考来源
[IS Value参考链接](https://github.com/sbarratt/inception-score-pytorch/tree/master)

[FID Value参考链接](https://github.com/mseitzer/pytorch-fid)

[CLIP Score参考链接](https://github.com/Taited/clip-score)