
import torch

import hybrid_mlp as H
import QViT as M  # auto patch
import data_satellite as DS
import train as T

import os


def run(c):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists(c['dir_path']):
        os.makedirs(c['dir_path'])

    with open(os.path.join(c['dir_path'], "config.txt"), 'w') as f:
        print(c, file=f)

    if c['model'] == 'qvt':
        model = M.QuantumVisionTransformer(
            embed_dim=c['embed_dim'],
            hidden_dim=c['hidden_dim'],
            num_channels=c['num_channels'],
            num_heads=c['num_heads'],
            num_layers=c['num_layers'],
            num_classes=c['num_classes'],
            patch_size=c['patch_size'],
            dropout=c['dropout'],
            vec_loader=c['vec_loader'],
            matrix_mul=c['ort_layer']
        ).to(device)

    elif c['model'] == 'hmlp':
        model = H.HybridMLP(c['embed_dim'], c['num_layers'], c['num_classes'], c['hidden_dim'], c['vec_loader'], c['ort_layer']).to(device)
    else:
        raise ValueError("Unknown model type: {}".format(c['model']))

    optimizer = torch.optim.Adam(model.parameters(), lr=c['lr'])

    train_loader, test_loader = DS.get_satellite_loaders(
        batch_size=c['batch_size'],
        subset_classes=c['classes']
    )

    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    T.train(model, c['epochs'], train_loader, test_loader, optimizer, criterion, c['dir_path'], device)

    return
