import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.join(__file__), '../../'))
LOG_ROOT = os.path.join(ROOT, 'exp_data/logs')
DATA_ROOT = os.path.join(ROOT, 'data')
OUTPUT_ROOT = os.path.join(ROOT, 'output')
FIG_ROOT = os.path.join(ROOT, 'figures')

os.makedirs(LOG_ROOT, exist_ok=True)


def get_parser():
    parser = argparse.ArgumentParser('Argument Parser for GCDR')
    parser.add_argument('--dataset', '-d', type=str,
                        choices=('kuairec', 'movielens', 'yelp', 'books'), required=True)
    parser.add_argument('--device', type=str, choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seed', '-seed', type=int, default=2024)

    # Base Setting
    parser.add_argument('--max_history_length', '-max_hl', type=int, default=5)
    parser.add_argument('--min_history_length', '-min_hl', type=int, default=1)
    parser.add_argument('--n_hidden', '-dim', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=4, help='n_head for Transformer')
    parser.add_argument('--n_layer', type=int, default=4, help='n_layer for Transformer')

    # Training Setting
    parser.add_argument('--ne', '-ne', type=int, default=1000, help='n_epoch')
    parser.add_argument('--bs', '-bs', type=int, default=2048, help='batch_size')
    parser.add_argument('--test_bs', '-tbs', type=int, default=8192, help='test_batch_size')
    parser.add_argument('--ri', '-ri', type=int, default=50, help='report interval')
    parser.add_argument('--patience', '-p', type=int, default=5)
    parser.add_argument('--warmup', '-warmup', type=int, default=0, help='warmup')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--use_final', action='store_true')

    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='learning_rate')
    parser.add_argument('--wd', '-wd', type=float, default=0, help='weight_decay')
    parser.add_argument('--norm', '-nm', '-norm', action='store_true')

    # Parameters for Diffusion
    parser.add_argument('--scale', type=float, default=1000)
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--beta_base', type=float, default=0.01)  
    parser.add_argument('--diffusion_steps', '-ds', type=int, default=32)
    parser.add_argument('--skip_step', '-ss', type=int, default=1)

    # Sensitive Model Setting
    parser.add_argument('--uncondition_rate', '-us', type=float, default=0.2)
    parser.add_argument('--category_uncondition_rate', '-uc', type=float, default=0.2)
    parser.add_argument('--dropout_g_c', '-dgc', type=float, default=0.3)

    # Loss Setting
    parser.add_argument('--n_negative', '-ng', type=int, default=100)
    parser.add_argument('--tau', '-tau', type=int, default=8)
    parser.add_argument('--delta', '-delta', type=float, default=0.001)
    parser.add_argument('--gamma', '-gamma', type=float, default=0)

    parser.add_argument('--lambda_rec_loss', '-lrec', type=float, default=1.0)
    parser.add_argument('--lambda_mse_loss', '-lmse', type=float, default=1.0)
    parser.add_argument('--lambda_user_loss', '-luser', type=float, default=0.001)

    # Multiple Distribution Setting
    parser.add_argument('--mc', '-mc', action='store_true', help='Multiple Distributions')
    parser.add_argument('--nmc', '-nmc', type=int, default=5, help='L: number of cases for mc')
    parser.add_argument('--mc_func', '-mc_func', type=str, default='max', help='avg or max')

    # Ablation Study
    parser.add_argument('--g_c_l', '-g_c_l', type=float, default=1.0)
    parser.add_argument('--g_g_s', '-g_g_s', type=float, default=1.0)
    parser.add_argument('--g_g_c', '-g_g_c', type=float, default=1.0)

    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args, parser


class Logger:
    def __init__(self) -> None:
        self.filename = None
        self.writer = None
        self.start = time.time()

    def get_log_filename(self, args, parser):
        full_filename = '|'.join(map(lambda x: '='.join(
            map(str, x)), args.__dict__.items())) + '.log'
        require_keys = {'dataset'}
        pairs = []
        for k, v in args.__dict__.items():
            if k not in require_keys and v == parser.get_default(k):
                continue
            pairs.append((k, str(v)))
        filename = '|'.join(map('='.join, pairs)) + '.log'
        return full_filename, filename

    def set_log_file(self, args, parser):
        full_filename, filename = self.get_log_filename(args, parser)
        self.filename = filename
        print(f'[Log File] {filename}\n')
        self.writer = open(os.path.join(LOG_ROOT, filename), 'w')
        self.writer.write(' '.join(sys.argv) + '\n\n')
        self.writer.write('[Full Filename] ' + full_filename + '\n\n')

    def print(self, *values, sep=" ", end="\n", **kwargs):
        print(*values, sep=sep, end=end, **kwargs)
        if self.writer is not None:
            self.writer.write(sep.join(map(str, values)) + end)

    def __del__(self):
        end = time.time()
        self.print(f'\n[Time Cost] {end - self.start}s')
        if self.writer is not None:
            self.writer.close()
            print(f'\n[Log File] {self.filename}')


logger = Logger()


def main():
    args, parser = get_args()
    _, filename = logger.get_log_filename(args, parser)
    print(filename)


if __name__ == '__main__':
    main()
