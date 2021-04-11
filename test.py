import numpy as np

from utils import set_seed, load_data, get_config, get_datasets, get_optimizer, get_model
from mnist_dataset import MnistDataset


if __name__ == "__main__":
    config = get_config()

    images, labels = load_data(config['data']['test'])

    test_dataset = MnistDataset(images, labels, batch_size=config['eval']['batch_size'])

    # MODEL
    model = get_model(config['model'])
    model.load(config['experiment']['model_path'])

    # VALIDATION
    model.eval()
    y_hat_hist, y_hist = list(), list()
    for j in range(len(test_dataset)):
        x_val, y_val = test_dataset[j]
        y_hat = model.forward(x_val)
        y_hat_hist.append(y_hat), y_hist.append(y_val)

    y_hat = np.vstack(y_hat_hist)
    y = np.vstack(y_hist)

    preds = np.argmax(y_hat, axis=1)
    real = np.argmax(y, axis=1)
    test_acc = (preds == real).mean()
    print(f'Test accuracy: {test_acc}')

