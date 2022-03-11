from ntpath import join
import os

def get_log_folder(args):
    folder = get_ckpt_folder(args)
    folder = folder.replace("checkpoints",'logs')
    return folder


def get_ckpt_folder(args):
    folder = f'ckpt/{args.embedding_model}_{args.metric}_{args.loss}_{args.n_keyword}/checkpoints/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def get_ckpt_path(args):
    path = os.path.join(get_ckpt_folder(args),'model.ckpt')
    return path
def get_visualize_folder(args):
    folder = f'visualize/{args.embedding_model}_{args.metric}_{args.loss}_{args.n_keyword}'
    sub_folder = ['train','test']
    for sub in sub_folder:
        subfolder = os.path.join(folder,sub)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
    return folder 

