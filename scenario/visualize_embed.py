import numpy as np
import torch
import tqdm
import os

import util
import nnet

def visualize_step(model, batch, batch_idx, args):
    # extract model from class
    model = model.embedding
    # evaluate
    model.eval()
    with torch.no_grad():
        # extract data
        x, lens, y = batch
        x = x.to(args.device)
        lens = lens.to(args.device)
        y = y.to(args.device)
        # compute embedding
        x = model(util.MelSpectrogram(args).transform(x, lens))
        x, y = x.cpu().detach().numpy(), y.cpu().detach().numpy()
    return x, y

def visualize_embed(args):
    # get model
    path = util.get_checkpoint_path(args)
    if not torch.cuda.is_available():
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)
    model = nnet.get_model(args)
    model.load_state_dict(checkpoint['model_state_dict'])

    # get loader
    train_loader, val_loader, test_loader, _, _ = util.get_loader(args)
    vocab = train_loader.dataset.vocab

    # visualize train dataset
    X_train, y_train = [], []
    for batch_idx, batch in enumerate(tqdm.tqdm(train_loader)):
        x, y = visualize_step(model, batch, batch_idx, args)
        X_train.append(x)
        y_train.append(y)
        if args.limit_train_batch > 0:
            if batch_idx > args.limit_train_batch:
                break
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    # visualize test dataset
    X_test, y_test = [], []
    for batch_idx, batch in enumerate(tqdm.tqdm(test_loader)):
        x, y = visualize_step(model, batch, batch_idx, args)
        X_test.append(x)
        y_test.append(y)
        if args.limit_val_batch > 0:
            if batch_idx > args.limit_val_batch:
                break
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    # save vector and metadata
    folder = util.get_visualize_folder(args)
    np.savetxt(os.path.join(folder, 'train', 'vectors.tsv'), X_train, delimiter='\t')
    with open(os.path.join(folder, 'train', 'metadata.tsv'), 'w', encoding='utf-8') as f:
        [f.write(str(vocab[int(_)]) + '\n') for _ in y_train]
    
    np.savetxt(os.path.join(folder, 'test', 'vectors.tsv'), X_test, delimiter='\t')
    with open(os.path.join(folder, 'test', 'metadata.tsv'), 'w', encoding='utf-8') as f:
        [f.write(str(vocab[int(_)]) + '\n') for _ in y_test]