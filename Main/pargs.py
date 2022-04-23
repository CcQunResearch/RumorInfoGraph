import argparse


def pargs():
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument('--lr', dest='lr', type=float, default=0.001,)
    parser.add_argument('--lamda', dest='lamda', type=float, default=0.001)
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=0)
    parser.add_argument('--dim', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=500)


    parser.add_argument('--no_use-unsup-loss', dest='use_unsup_loss', action='store_false')
    parser.add_argument('--no_separate-encoder', dest='separate_encoder', action='store_false')


    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    return args
