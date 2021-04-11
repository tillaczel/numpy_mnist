import numpy as np


class MnistDataset:
    def __init__(self, images, labels, batch_size, transforms=None):
        self.batch_size = batch_size
        self.transforms = transforms
        # Load images
        self.images = images

        # Load labels
        self.labels = labels

        self.length = int(np.ceil(self.images.shape[0] / self.batch_size))

    def img_to_vector(self, imgs):  # Simpel reshaping
        if len(imgs.shape) == 3:
            num_imgs = imgs.shape[0]
            return imgs.reshape((num_imgs, -1))
        elif len(imgs.shape) == 2:
            return imgs.reshape(-1)
        else:
            print("Input needs to be array with shape of length 2 or 3")

    def norm_imgs(self, imgs):
        return imgs/255

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index > self.length:
            return None
        if index == self.length:
            images = self.images[index * self.batch_size:-1]
            labels = self.labels[index * self.batch_size:-1]
        else:
            images = self.images[index * self.batch_size:index * self.batch_size + self.batch_size]
            labels = self.labels[index * self.batch_size:index * self.batch_size + self.batch_size]
        # ToDo: Augmentation
        # ToDo: Normalize
        images = self.norm_imgs(images)
        images_as_vec = self.img_to_vector(images)
        labels = np.eye(10)[labels]

        return images_as_vec, labels




