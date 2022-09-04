import argparse


def pargs():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Weibo')
    parser.add_argument('--vector_size', type=int, help='word embedding size', default=200)
    parser.add_argument('--unsup_train_size', type=int, help='word embedding unlabel data train size', default=20000)
    parser.add_argument('--runs', type=int, default=8)
    parser.add_argument('--ft_runs', type=int, default=5)

    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--unsup_bs_ratio', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--separate_encoder', type=str2bool, default=True)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--ft_lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ft_epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--gamma ', type=float, default=1)
    parser.add_argument('--lamda', type=float, default=0.001)

    parser.add_argument('--k', type=int, default=10000)

    args = parser.parse_args()
    return args
