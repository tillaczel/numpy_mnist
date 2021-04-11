import random

import numpy as np
import yaml

from mnist_dataset import MnistDataset
from nn import Model
from nn.activations import ReLu, LeakyReLu, SoftMax
from nn.layers import Linear, BatchNorm, DropOut
from nn.optimizers import SGD, Momentum
import augmentations

def set_seed(seed=43):
    np.random.seed(seed)
    random.seed(seed)


def load_data(data_config):
    f = open(data_config['image_path'], 'r')
    a = np.fromfile(f, dtype='>i4', count=4)  # data type is signed integer big-endian
    images = np.fromfile(f, dtype=np.uint8)
    images = images.reshape(a[1:])

    f = open(data_config['label_path'], 'r')
    t = np.fromfile(f, count=2, dtype='>i4')  # data type is signed integer big-endian
    labels = np.fromfile(f, dtype=np.uint8)

    return images, labels


def get_datasets(config):
    # DATA
    images, labels = load_data(config['data']['dev'])

    # Training- and Validation-split
    n_all_ims = images.shape[0]
    n_train_ims = round(n_all_ims * config['data']['dev']['train_fraction'])

    idx_train = np.random.choice(range(n_all_ims), n_train_ims, replace=False)
    idx_val = list(set(range(n_all_ims)) - set(idx_train))

    X_train = images[idx_train, :]
    y_train = labels[idx_train]
    train_dataset = MnistDataset(X_train, y_train, batch_size=config['train']['batch_size'])

    X_val = images[idx_val, :]
    y_val = labels[idx_val]
    val_dataset = MnistDataset(X_val, y_val, batch_size=config['eval']['batch_size'], transforms=get_transforms(config["data"]["augmentations"]))

    return train_dataset, val_dataset


def get_optimizer(optimizer_config, model, loss):
    name = optimizer_config['name']
    lr = optimizer_config['lr']
    if name == 'SGD':
        return SGD(model, loss, lr=lr)
    elif name == 'momentum':
        return Momentum(model, loss, lr=lr, beta=optimizer_config['beta'])
    else:
        raise ValueError(f'Invalid optimizer: {name}')


def get_activation_f(name):
    if name == 'ReLu':
        return ReLu
    elif name == 'LeakyReLu':
        return LeakyReLu
    else:
        raise ValueError(f'Invalid activation: {name}')


def get_model(model_config):
    input_dim = model_config['input_dim']
    fc_layer_dims = model_config['fc_layer_dims']
    activation_f = get_activation_f(model_config['activation'])
    dropout_p = model_config['dropout_p']
    batchnorm = model_config['batchnorm']

    layers = list()
    for layer_dim in fc_layer_dims[:-1]:
        layers.append(Linear(input_dim, layer_dim))
        if batchnorm:
            layers.append(BatchNorm(layer_dim, gamma=batchnorm))
        if dropout_p:
            layers.append(DropOut(dropout_p))
        layers.append(activation_f())
        input_dim = layer_dim
    layers.append(Linear(input_dim, fc_layer_dims[-1]))
    layers.append(SoftMax())
    return Model(layers)

def get_transforms(augmentation_config):
    aug_funcs = []
    all_funcs_dict = augmentations.__dict__
    for aug_identifier in augmentation_config:
        if aug_identifier in all_funcs_dict:
            aug_funcs.append(all_funcs_dict[aug_identifier])

    return aug_funcs

def get_config():
    config_path = 'config.yaml'
    with open(config_path) as fd:
        config = yaml.load(fd, yaml.FullLoader)
    return config