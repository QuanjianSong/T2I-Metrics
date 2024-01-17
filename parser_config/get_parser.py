from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from utils.inception import InceptionV3

def get_basic_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int,
                        help=('Number of processes to use for data loading. '
                            'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')

    return parser

def get_is_value_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    parser.add_argument('--cal_IS', type=str, default=True,
                        help='whether or not to calculate the IS.')
    parser.add_argument('--is_dims', type=int, default=1000,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                            'By default, uses pool3 features'))
    parser.add_argument('--save-stats', action='store_true',
                        help=('Generate an npz archive from a directory of samples. '
                            'The first path is used as input and the second as output.'))
    parser.add_argument('--path1', type=str, 
                        help=('Paths to the generated images or '
                            'to .npz statistic files'))
    
    return parser

def get_fid_value_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    parser.add_argument('--cal_FID', type=str, default=True,
                        help='whether or not to calculate the FID.')
    parser.add_argument('--fid_dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                            'By default, uses pool3 features'))
    parser.add_argument('--save_stats', type=bool, default=False,
                        help=('Dimensionality of Inception features to use. '
                            'By default, uses pool3 features'))
    parser.add_argument('--path2', type=str, 
                        help=('Paths to the generated images or '
                            'to .npz statistic files'))

    return parser

def get_clip_score_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    parser.add_argument('--cal_CLIP', type=str, default=True,
                        help='whether or not to calculate the FID.')
    parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                        help='CLIP model to use')
    parser.add_argument('--real_flag', type=str, default='img',
                        help=('The modality of real path. '
                            'Default to img'))
    parser.add_argument('--fake_flag', type=str, default='txt',
                        help=('The modality of real path. '
                            'Default to txt'))
    parser.add_argument('--real_path', type=str, default='/alg_vepfs/private/panfayu/sqj/my_code/T2I-Metrics/examples/imgs1', 
                        help=('Paths to the generated images or '
                            'to .npz statistic files'))
    parser.add_argument('--fake_path', type=str, default='/alg_vepfs/private/panfayu/sqj/my_code/T2I-Metrics/examples/prompt',
                        help=('Paths to the generated images or '
                            'to .npz statistic files'))

    return parser

def merged_parser(*parser_list):
    merged_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                                parents=parser_list)
    args = merged_parser.parse_args()
    return args