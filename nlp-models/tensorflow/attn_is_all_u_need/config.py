import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--source_max_len', type=int, default=10)
parser.add_argument('--target_max_len', type=int, default=20)
parser.add_argument('--min_freq', type=int, default=50)
parser.add_argument('--hidden_units', type=int, default=128)
parser.add_argument('--num_blocks', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--dropout_rate', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--positional_encoding', type=str, default='sinusoidal')
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--tied_proj_weight', action='store_false')
parser.add_argument('--tied_embedding', action='store_true')
parser.add_argument('--label_smoothing', action='store_true')
parser.add_argument('--lr_decay_strategy', type=str, default='exp')
parser.add_argument('--warmup_steps', type=int, default=4000,
    help="this will be used when '--lr_decay_strategy=noam'")
parser.add_argument('--model_dir', type=str, default='./saved')

parser.set_defaults(tied_proj_weight=True)

args = parser.parse_args()
