import json
import torch
import os
from tqdm import tqdm
from resnet import ResNet1d
from dataloader import BatchDataloader
import torch.optim as optim
import numpy as np


device = torch.device('cpu')

def compute_loss(ages, pred_ages, weights):
    diff = ages.flatten() - pred_ages.flatten()
    loss = torch.sum(weights.flatten() * diff * diff)
    return loss


def compute_weights(ages, max_weight=np.inf):
    _, inverse, counts = np.unique(ages, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    normalized_weights = weights / sum(weights)
    w = len(ages) * normalized_weights
    # Truncate weights to a maximum
    if max_weight < np.inf:
        w = np.minimum(w, max_weight)
        w = len(ages) * w / sum(w)
    return w


def train(model, optimizer, ep, dataload):
    model.train()
    total_loss = 0
    n_entries = 0
    train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
    train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                     desc=train_desc.format(ep, 0, 0), position=0)
    for traces, ages, weights in dataload:
        traces = traces.transpose(1, 2)
        traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)
        # Reinitialize grad
        model.zero_grad()
        # Send to device
        # Forward pass
        pred_ages = model(traces)
        loss = compute_loss(ages, pred_ages, weights)
        # Backward pass
        loss.backward()
        # Optimize

        optimizer.step()
        # Update
        bs = len(traces)
        total_loss += loss.detach().cpu().numpy()
        n_entries += bs
        # Update train bar
        train_bar.desc = train_desc.format(ep, total_loss / n_entries)
        train_bar.update(1)
    train_bar.close()
    return total_loss / n_entries


def eval(model, ep, dataload):
    model.eval()
    total_loss = 0
    n_entries = 0
    eval_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    eval_bar = tqdm(initial=0, leave=True, total=len(dataload),
                    desc=eval_desc.format(ep, 0, 0), position=0)
    for traces, ages, weights in dataload:
        traces = traces.transpose(1, 2)
        traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)
        with torch.no_grad():
            # Forward pass
            pred_ages = model(traces)
            loss = compute_loss(ages, pred_ages, weights)
            # Update outputs
            bs = len(traces)
            # Update ids
            total_loss += loss.detach().cpu().numpy()
            n_entries += bs
            # Print result
            eval_bar.desc = eval_desc.format(ep, total_loss / n_entries)
            eval_bar.update(1)
    eval_bar.close()
    return total_loss / n_entries


def train_model(arguments):
    import h5py
    import pandas as pd
    import argparse
    from warnings import warn

    torch.manual_seed(arguments['seed'])
    # Set device
    # device = torch.device('cuda:0' if args.cuda else 'cpu')
    device = torch.device('cpu')
    folder = arguments['folder']

    # Generate output folder if needed
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Save config file
    # with open(os.path.join(folder, 'args.json'), 'w') as f:
    #     json.dump(vars(arguments.keys), f, indent='\t')

    tqdm.write("Building data loaders...")
    # Get csv data
    df = pd.read_csv(arguments['path_to_csv'], index_col=arguments['ids_col'])
    ages = df[arguments['age_col']]
    # Get h5 data
    f = h5py.File(arguments['path_to_traces'], 'r')
    traces = f[arguments['traces_dset']]
    if arguments['ids_dset']:
        h5ids = f[arguments['ids_dset']]
        df = df.reindex(h5ids, fill_value=False, copy=True)
    # Train/ val split
    valid_mask = np.arange(len(df)) <= arguments['n_valid']
    train_mask = ~valid_mask
    # weights
    weights = compute_weights(ages)
    # Dataloader
    train_loader = BatchDataloader(traces, ages, weights, bs=arguments['batch_size'], mask=train_mask)
    valid_loader = BatchDataloader(traces, ages, weights, bs=arguments['batch_size'], mask=valid_mask)
    tqdm.write("Done!")

    tqdm.write("Define model...")
    N_LEADS = 12  # the 12 leads
    N_CLASSES = 1  # just the age
    model = ResNet1d(input_dim=(N_LEADS, arguments['seq_length']),
                     blocks_dim=list(zip(arguments['net_filter_size'], arguments['net_seq_length'])),
                     n_classes=N_CLASSES,
                     kernel_size=arguments['kernel_size'],
                     dropout_rate=arguments['dropout_rate'])
    model.to(device=device)
    print("model")
    print (model)
    tqdm.write("Done!")

    tqdm.write("Define optimizer...")
    optimizer = optim.Adam(model.parameters(), arguments['lr'])
    tqdm.write("Done!")

    tqdm.write("Define scheduler...")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=arguments['patience'],
                                                     min_lr=arguments['lr_factor'] * arguments['min_lr'],
                                                     factor=arguments['lr_factor'])
    tqdm.write("Done!")

    tqdm.write("Training...")
    start_epoch = 0
    best_loss = np.inf
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                    'weighted_rmse', 'weighted_mae', 'rmse', 'mse'])
    for ep in range(start_epoch, 1):
        print (type(train_loader))
        print (train_loader)
        train_loss = train(model=model, optimizer=optimizer, ep=ep, dataload=train_loader)
        valid_loss = eval(model, ep, valid_loader)
        # Save best model
        if valid_loss < best_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'valid_loss': valid_loss,
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'model.pth'))
            # Update best validation loss
            best_loss = valid_loss
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Interrupt for minimum learning rate
        if learning_rate < arguments['min_lr']:
            break
        # Print message
        tqdm.write('Epoch {:2d}: \tTrain Loss {:.6f} '
                   '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t'
                   .format(ep, train_loss, valid_loss, learning_rate))
        # Save history
        history = history.append({"epoch": ep, "train_loss": train_loss,
                                  "valid_loss": valid_loss, "lr": learning_rate}, ignore_index=True)
        history.to_csv(os.path.join(folder, 'history.csv'), index=False)
        # Update learning rate
        scheduler.step(valid_loss)
    tqdm.write("Done!")


