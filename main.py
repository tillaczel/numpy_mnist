import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from nn.optimizers import SGD, Momentum
from nn.losses import CrossEntropy
from nn.activations import ReLu, LeakyReLu, SoftMax
from nn.layers import Linear, DropOut
from nn import Model
from mnist_dataset import MnistDataset


def set_seed(seed=43):
    np.random.seed(seed)
    random.seed(seed)


def img_reshape(imgs): # Simpel reshaping
    if len(imgs.shape) == 3:
        num_imgs = imgs.shape[0]
        return imgs.reshape((num_imgs, -1))
    elif len(imgs.shape) == 2:
        return imgs.reshape(-1)
    else:
        print("Input needs to be array with shape of length 2 or 3")


if __name__ == "__main__":
    # Setting seed for reproducability
    set_seed()

    # DATA
    f = open('data/train-images-idx3-ubyte', 'r')
    a = np.fromfile(f, dtype='>i4', count=4)  # data type is signed integer big-endian
    images = np.fromfile(f, dtype=np.uint8)
    images = images.reshape(a[1:])

    f = open('data/train-labels-idx1-ubyte', 'r')
    t = np.fromfile(f, count=2, dtype='>i4')  # data type is signed integer big-endian
    labels = np.fromfile(f, dtype=np.uint8)

    # Training- and Validation-split
    n_all_ims = images.shape[0]
    train_fraction = 5 / 6
    n_train_ims = round(n_all_ims * train_fraction)

    idx_train = np.random.choice(range(n_all_ims), n_train_ims, replace=False)
    idx_val = list(set(range(n_all_ims)) - set(idx_train))

    X_train = images[idx_train, :]
    y_train = labels[idx_train]
    train_dataset = MnistDataset(X_train, y_train, batch_size=64)

    X_val = images[idx_val, :]
    y_val = labels[idx_val]
    val_dataset = MnistDataset(X_val, y_val, batch_size=64)

    # MODEL
    layers = [Linear(784, 32), DropOut(0.2), ReLu(), Linear(32, 10), SoftMax()]
    model = Model(layers)
    # Define loss
    loss = CrossEntropy()
    # Define optimizer
    optimizer = SGD(model, loss, lr=0.1)

    EPOCHS = 100

    # Main loop
    train_loss_hist, val_loss_hist, val_acc_hist = list(), list(), list()
    pbar = tqdm(range(EPOCHS))
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

    plt.plot(np.array(train_loss_hist)[:, 0], np.array(train_loss_hist)[:, 1], label='train')
    plt.plot(np.array(val_loss_hist)[:, 0], np.array(val_loss_hist)[:, 1], label='valid')
    plt.legend()
    plt.show()






