import numpy as np
import random
import matplotlib.pyplot as plt
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
    train_dataset = MnistDataset(X_train, y_train, batch_size=X_train.shape[0])

    X_val = images[idx_val, :]
    y_val = labels[idx_val]
    val_dataset = MnistDataset(X_val, y_val, batch_size=X_val.shape[0])

    # MODEL
    layers = [Linear(784, 32), DropOut(0.2), ReLu(), Linear(32, 10), SoftMax()]
    model = Model(layers)
    # Define loss
    loss = CrossEntropy()
    # Define optimizer
    optimizer = SGD(model, loss, lr=0.1)

    # TRAINING
    model.train()
    loss_hist = list()
    for i in range(100):
        for j in range(len(train_dataset)):
            x, y = train_dataset[j]
            loss_hist.append(optimizer.step(x, y))
    plt.plot(loss_hist)
    plt.show()

    # VALIDATION
    model.eval()
    correct_imgs = 0
    total_imgs = 0
    for i in range(len(val_dataset)):
        x_val, y_val = val_dataset[i]
        preds = np.argmax(model.forward(x_val), axis=1)
        total_imgs += x_val.shape[0]
        real = np.argmax(y_val, axis=1)
        correct_imgs += (preds == real).sum()
    print(f"Validation accuracy: {correct_imgs/total_imgs}")




