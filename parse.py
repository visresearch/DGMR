import argparse

def get_args():
    parser = argparse.ArgumentParser("Distill for pruned model")
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--frame", type=str)
    parser.add_argument("--print_freq", type=int)
    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--base_lr", type=float)

    parser.set_defaults(
        config_file = "./configs/distill_clip.yaml",
        frame = 'eva_clip',
        print_freq = 10,
        save_freq = 1,
        exclude_file_list = ['__pycache__', '.vscode', 'log', 'ckpt', '.git', 'out', 'dataset', 'weight', 'third_party'],
        resume = None,
        base_lr=None
    )

    args = parser.parse_args()

    return args

def get_args_classification():
    parser = argparse.ArgumentParser("Distill for pruned model")
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--frame", type=str)

    parser.set_defaults(
        config_file = "./configs/distill_clip.yaml",
        frame = 'eva_clip',
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args() 
    print(args)