# T2I-Metrics ----- 这是一个关于Text-to-Image中Metrics的Pytorch代码库
## 0. 项目介绍

近年来diffusion models的发展十分迅速，但本人发现目前关于diffusion models的评价指标并没有良好的集成，因此本人参考了时间,搭建了一个关于diffusion models若干评价指标的pipeline代码库。欢迎各位star + fork。

## 1. 环境配置


## 2. 快速开始
```
bash scripts/start.sh
```

其中，--cal_IS 表示是否计算IS， 默认为True。 -- --cal_FID 表示是否计算FID， 默认为True。 --cal_CLIP 表示是否计算CLIP， 默认为True。  
--path1 表示计算FID时的生成图像的路径，--path2 表示计算FID时的真实图像的路径。计算IS会默认采用--path1。
--real_path 表示计算clip score时使用的真实图像的路径, --fake_path 表示计算clip score时使用的文字的路径。

## 3. 参考来源
[IS Value参考链接](https://github.com/sbarratt/inception-score-pytorch/tree/master)
[FID Value参考链接](https://github.com/mseitzer/pytorch-fid)
[CLIP Score参考链接](https://github.com/Taited/clip-score)