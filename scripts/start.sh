#! /bin/bash

export http_proxy=http://100.68.161.73:3128 
export https_proxy=http://100.68.161.73:3128

python ./cal_diffusion_metric.py  --cal_IS True --cal_FID True --cal_CLIP True \
    --path1 ./examples/imgs1 --path2 ./examples/imgs2 \
    --real_path ./examples/imgs1 --fake_path ./examples/prompt