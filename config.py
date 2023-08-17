import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Dataset
data_arg = add_argument_group('Dataset')
data_arg.add_argument('--dataset', type=str, default='VimeoSepTuplet')
data_arg.add_argument('--data_root', type=str, default='/mnt/sdb5/vimeo_septuplet/')

# Model
model_arg = add_argument_group('Model')
model_arg.add_argument('--model',  type=str, default="SCubA_RP_2")
model_arg.add_argument('--n_inputs' , type=int , default=4)
model_arg.add_argument('--n_outputs' , type=int , default=1)
model_arg.add_argument('--emb_dim', type=int, default=32)
model_arg.add_argument('--patch_size', type=tuple, default=(3,4,4))
model_arg.add_argument('--cube_size', type=tuple, default=(2,8,8))
model_arg.add_argument('--stage', type=int, default=2)
model_arg.add_argument('--num_scale', type=int, default=3)
model_arg.add_argument('--joinType' , choices=["concat" , "add" , "none"], default="concat")
model_arg.add_argument('--num_channels' , type=int , default=3)

# Training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--loss', type=str, default='1*L1')
learn_arg.add_argument('--lr', type=float, default=2e-4)
learn_arg.add_argument('--beta1', type=float, default=0.9)
learn_arg.add_argument('--beta2', type=float, default=0.99)
learn_arg.add_argument('--batch_size', type=int, default=4)
learn_arg.add_argument('--test_batch_size', type=int, default=4)
learn_arg.add_argument('--start_epoch', type=int, default=0)
learn_arg.add_argument('--max_epoch', type=int, default=100)
learn_arg.add_argument('--checkpoint_dir', type=str ,default='.')
# learn_arg.add_argument("--checkpoint_path"  ,type=str , default='checkpoints/VimeoSepTuplet/SCubA_Ntd/checkpoint.pth')
learn_arg.add_argument("--checkpoint_path"  ,type=str , default=None)



learn_arg.add_argument("--pretrained" , type=str, default=None, help="Load from a pretrained model.")

learn_arg.add_argument("--best_model_path" , type=str,default=None, help="path for model_best")

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_iter', type=int, default=60)
misc_arg.add_argument('--num_gpu', type=int, default=2)
misc_arg.add_argument('--random_seed', type=int, default=12345)
misc_arg.add_argument('--num_workers', type=int, default=16)
misc_arg.add_argument('--use_tensorboard', action='store_true')
misc_arg.add_argument('--val_freq', type=int, default=1)
misc_arg.add_argument('--main_rank', type=int, default=0)
misc_arg.add_argument('--snufilm_mode',  type=str, default="easy")
def get_args():
    """Parses all of the arguments above
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed
