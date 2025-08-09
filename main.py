from models.get_models import init_metric_model
from tools.metrics import cal_CLIP_I, cal_CLIP_T, cal_DINOv2, cal_DreamSim, cal_LPIPS, cal_LAION_Aes, \
                    cal_Q_Align, cal_Q_Align_IQ, cal_CLIP_IQA, cal_TOPIQ_NR, cal_MANIQA, cal_HYPERIQA, \
                    cal_IS, cal_FID

METRIC_CAL_FUNC = {
    "IS": cal_IS,
    "FID": cal_FID,
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


if __name__ == '__main__':
    # choose your metrics to load
    metric_list = [
        "IS",
        "FID",
        "CLIP-T",
        "DINO",
        "LAION-Aes",
    ]
    models = init_metric_model(metric_list, "cuda")
    # ------------------------------------------------------------------
    # cal IS
    avg_mean, avg_std = cal_IS(models['IS'], path='examples/imgs1', dims=1000, batch_size=1, splits=5)
    print(f"avg_mean:{avg_mean}, avg_std:{avg_std}")
    # breakpoint()
    # cal FID
    avg_val = cal_FID(models['FID'], path_list=['examples/imgs1', 'examples/imgs2'])
    print(f"avg:{avg_val}")
    # breakpoint()
    # ------------------------------------------------------------------
    # Specify the corresponding jsonl file to compute the related metrics.
    avg_val = cal_CLIP_T(models['CLIP-T'], jsonl_path='examples/img_txt.jsonl') # for prompt-img pair
    print(f"avg:{avg_val}")
    # breakpoint()
    # ------------------------------------------------------------------
    avg_val = cal_DINOv2(models['DINO'], jsonl_path='examples/img_img.jsonl') # for img-img pair
    print(f"avg:{avg_val}")
    # breakpoint()
    # ------------------------------------------------------------------
    avg_val = cal_LAION_Aes(models['LAION-Aes'], jsonl_path='examples/img.jsonl') # for img
    print(f"avg:{avg_val}")
    # breakpoint()
