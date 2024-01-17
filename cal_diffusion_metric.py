import os
import torch

from src.is_score import cal_is_value
from src.fid_score import save_fid_stats, cal_fid_value
from src.clip_score import cal_clip_score
from parser_config.get_parser import get_basic_parser, get_is_value_parser, get_fid_value_parser, get_clip_score_parser, merged_parser

def print_metrics(metrics):
    print("------------------T2I-Metrics-----------------")
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                print('{}: {:.8f}'.format(sub_k, sub_v), end='  ')
            print()
        else:
            print('{}: {:.8f}'.format(k, v))

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

    if args.save_stats:
        save_fid_stats([args.path1, args.path2], args.batch_size, device, args.dims, num_workers)
        return
    
    # metrics_dicts
    metrics = {}

    if args.cal_IS:
        assert args.path1 is not None, 'path1 is necessary for calculating IS value.'
        print('********************Calculate IS Value*********************')
        is_value, is_std = cal_is_value(args.path1, args.batch_size, args.is_dims,
                                device, num_workers, splits=5)
        
        metrics['IS_Value'] = {"IS_Value": is_value, "IS_std": is_std}
        print('**************************End******************************')

    if args.cal_FID:
        assert args.path1 is not None and args.path2 is not None, 'path1 and path2 is necessary for calculating FID value.'
        print('********************Calculate FID Value*********************')
        fid_value = cal_fid_value([args.path1, args.path2],
                                        args.batch_size,
                                        device,
                                        args.fid_dims,
                                        num_workers)
        metrics['FID_Value'] = fid_value
        print('**************************End******************************')

    if args.cal_CLIP:
        assert args.jsonl_path is not None or args.real_path is not None and args.fake_path is not None, 'jsonl_path or real_path and fake_path is necessary for calculating CLIP score.'
        print('********************Calculate CLIP Score*********************')
        clip_score = cal_clip_score(clip_model=args.clip_model, batch_size=args.batch_size, device=device,
                    num_workers=num_workers, real_path=args.real_path, fake_path=args.fake_path, jsonl_path=args.jsonl_path,
                    real_flag=args.real_flag, fake_flag=args.fake_flag)
        metrics['CLIP_Score'] = clip_score
        print('**************************End******************************')

    print_metrics(metrics)


if __name__ == '__main__':
    get_basic_parser = get_basic_parser()
    is_value_parser = get_is_value_parser()
    fid_value_parser = get_fid_value_parser()
    clip_score_parser = get_clip_score_parser()

    args = merged_parser(*[get_basic_parser, is_value_parser,
                        fid_value_parser, clip_score_parser])

    main(args)