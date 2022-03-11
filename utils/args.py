import argparse
import torch 

def get_args():
    parser = argparse.ArgumentParser(description="Get the argument")

    #prepare_data
    parser.add_argument("--data_dir",type=str,default="data")
    parser.add_argument("--extract_data",action="store_false")
    parser.add_argument("--extract_rir_noise",action="store_false")

    parser.add_argument("--scenario",type=str,default="prepare_data")


    #build json
    parser.add_argument("--n_keyword",type=int,default=12)

    #train
    parser.add_argument("--embedding_model",type=str,default='xvector')
    parser.add_argument("--metric",type=str,default="softmax")
    parser.add_argument("--loss",type=str,default="ce")
    parser.add_argument("--no_shuffle",action="store_false")
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--no_pin_memory",action="store_false")
    parser.add_argument("--num_worker",type=int,default=2)
    parser.add_argument("--n_mels",type=int,default=40)
    parser.add_argument("--cnn_channels",type=str,default='512,512,512,512,1500')
    parser.add_argument("--cnn_kernel",type=str,default='5,3,3,1,1')
    parser.add_argument("--cnn_dilation",type=str,default="1,2,3,1,1")
    parser.add_argument("--n_embed",type=int,default=512)
    parser.add_argument("--n_heads",type=int,default=2)
    parser.add_argument('--optimizer',type=str,default='adam')
    parser.add_argument("--learning_rate",type=float,default=0.001)
    parser.add_argument("--scheduler",type=str,default='plateau')
    parser.add_argument("--clip_grad_norm",type=float,default=5.0)
    parser.add_argument("--limit_train_batch_hook",type=int,default=-1)
    parser.add_argument("--limit_val_batch_hook",type=int,default=-1)
    parser.add_argument("--num_epoch",type=int,default=3)
    parser.add_argument("--no_evaluate",action='store_false')
    parser.add_argument("--log_iter",type=int,default=10)
    parser.add_argument("--clear_cache",action='store_true')
    parser.add_argument("--s",type=float,default=64)
    parser.add_argument("--m",type=float,default=0.5)
    parser.add_argument("--n_fft",type=int,default=400)









    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args