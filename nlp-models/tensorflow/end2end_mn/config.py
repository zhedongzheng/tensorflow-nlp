import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dropout_rate', type=float, default=0.3)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--display_step', type=int, default=50)

args = parser.parse_args()