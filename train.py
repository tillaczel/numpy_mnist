import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from nn.losses import CrossEntropy
from utils import set_seed, get_datasets, get_optimizer, get_model, get_config

if __name__ == "__main__":
    config = get_config()

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

    # Plots
    plt.plot(np.array(train_loss_hist)[:, 0], np.array(train_loss_hist)[:, 1], label='train')
    plt.plot(np.array(val_loss_hist)[:, 0], np.array(val_loss_hist)[:, 1], label='valid')
    plt.legend()
    plt.show()

    plt.plot(np.array(val_acc_hist), label='val acc')
    plt.legend()
    plt.show()

    model.save(config['experiment']['model_path'])





