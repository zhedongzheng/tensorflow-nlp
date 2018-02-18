import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--embed_dim', type=int, default=80)
parser.add_argument('--hidden_size', type=int, default=80)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--n_hops', type=int, default=1)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--display_step', type=int, default=50)
parser.add_argument('--add_gradient_noise', action='store_true')

args = parser.parse_args()