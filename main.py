import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

from nn.optimizers import SGD, Momentum
from nn.losses import CrossEntropy
from nn.activations import ReLu, LeakyReLu, SoftMax
from nn.layers import Linear, DropOut, BatchNorm
from nn import Model
from mnist_dataset import MnistDataset


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
    images, labels = load_data(config['data'])

    # Training- and Validation-split
    n_all_ims = images.shape[0]
    n_train_ims = round(n_all_ims * config['data']['train_fraction'])

    idx_train = np.random.choice(range(n_all_ims), n_train_ims, replace=False)
    idx_val = list(set(range(n_all_ims)) - set(idx_train))

    X_train = images[idx_train, :]
    y_train = labels[idx_train]
    train_dataset = MnistDataset(X_train, y_train, batch_size=config['train']['batch_size'])

    X_val = images[idx_val, :]
    y_val = labels[idx_val]
    val_dataset = MnistDataset(X_val, y_val, batch_size=config['train']['batch_size'])

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

    layers = list()
    for layer_dim in fc_layer_dims[:-1]:
        layers.append(Linear(input_dim, layer_dim))
        if dropout_p:
            layers.append(DropOut(dropout_p))
        layers.append(activation_f())
        input_dim = layer_dim
    layers.append(Linear(input_dim, fc_layer_dims[-1]))
    layers.append(SoftMax())
    return Model(layers)


if __name__ == "__main__":
    config_path = 'config.yaml'
    with open(config_path) as fd:
        config = yaml.load(fd, yaml.FullLoader)

    # Setting seed for reproducability
    set_seed()

    # Get data
    train_dataset, val_dataset = get_datasets(config)

    # MODEL
    model = get_model(config['model'])
    # Define loss
    loss = CrossEntropy()
    # Define optimizer
    optimizer = get_optimizer(config['train']['optimizer'], model, loss)

    # Main loop
    train_loss_hist, val_loss_hist, val_acc_hist = list(), list(), list()
    pbar = tqdm(range(config['train']['epochs']))
    for i in pbar:
        # TRAINING
        model.train()
        for j in range(len(train_dataset)):
            x, y = train_dataset[j]
            train_loss = optimizer.step(x, y)
            train_loss_hist.append((i + j / len(train_dataset), train_loss))

        # VALIDATION
        model.eval()
        y_hat_hist, y_hist = list(), list()
        for j in range(len(val_dataset)):
            x_val, y_val = val_dataset[j]
            y_hat = model.forward(x_val)
            y_hat_hist.append(y_hat), y_hist.append(y_val)

        y_hat = np.vstack(y_hat_hist)
        y = np.vstack(y_hist)
        val_loss = loss.forward(y, y_hat)
        val_loss_hist.append((i+1, val_loss))

        preds = np.argmax(y_hat, axis=1)
        real = np.argmax(y, axis=1)
        val_acc = (preds == real).mean()
        val_acc_hist.append(val_acc)
        pbar.set_postfix({'val_acc': val_acc})

    plt.plot(np.array(train_loss_hist)[:, 0], np.array(train_loss_hist)[:, 1], label='train', alpha=0.3)
    plt.plot(np.array(val_loss_hist)[:, 0], np.array(val_loss_hist)[:, 1], label='valid')
    plt.legend()
    plt.show()

    plt.plot(np.array(val_acc_hist), label='val acc')
    plt.legend()
    plt.show()






