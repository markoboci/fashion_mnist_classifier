import tensorflow as tf
import numpy as np
import os
import gzip
import numpy as np

# mnist data loader
def load_mnist(path, kind='train'):

    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


class DataInput:

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    batch_size = 4
    step_num = 0
    shuffle_indexes = []
    epoch_num = 0

    def __init__(self, data_path, batch_size, val_set_size=4000, augment_prob=0.15,
                 add_noise_aug=True, adjust_brightness_aug=True, horizontal_flip_aug=True,
                 erase_patch_aug=True, crop_aug=True):

        images, labels = load_mnist(data_path)
        print(type(images))
        n = len(images)
        val_indexes = self.get_val_indexes(labels, n, val_set_size)

        self.val_images = images[val_indexes]
        self.val_labels = labels[val_indexes]
        self.train_images = np.array([images[i] for i in range(n) if i not in val_indexes])
        self.train_labels = np.array([labels[i] for i in range(n) if i not in val_indexes])

        self.shuffle_indexes = np.arange(len(self.train_images))
        np.random.shuffle(self.shuffle_indexes)
        self.batch_size = batch_size

        self.add_noise_aug = add_noise_aug
        self.adjust_brightness_aug = adjust_brightness_aug
        self.horizontal_flip_aug = horizontal_flip_aug
        self.erase_patch_aug = erase_patch_aug
        self.crop_aug= crop_aug

        self.augment_prob = augment_prob


    def get_batch(self):

        np.random.seed()
        self.step_num += 1
        if self.step_num * self.batch_size > len(self.train_images):

            self.shuffle_indexes = np.arange(len(self.train_images))
            np.random.shuffle(self.shuffle_indexes)
            self.step_num = 1

            self.epoch_num += 1
            print("Completed epochs: " + str(self.epoch_num))

        idx = self.shuffle_indexes[(self.step_num - 1)*self.batch_size : self.step_num*self.batch_size]
        labels = self.train_labels[idx]
        # reshape and normalize images
        images = np.array([np.expand_dims(img.reshape(28, 28) / 255, axis = 2) for img in self.train_images[idx]])

        # augmentation
        if self.add_noise_aug:
            images = self.add_noise(images)
        if self.adjust_brightness_aug:
            images = self.adjust_brightness(images)
        if self.horizontal_flip_aug:
            images = self.horizontal_flip(images, labels)
        if self.erase_patch_aug:
            images = self.erase_patch(images)
        if self.crop_aug:
            images = self.crop(images)

        return images, labels

    def get_val_set(self):
        labels = self.val_labels
        # reshape and normalize images
        images = np.array([np.expand_dims(img.reshape(28, 28) / 255, axis=2) for img in self.val_images])
        return images, labels

    def get_val_indexes(self, labels, n, val_set_size):
        np.random.seed(1)
        val_indexes = np.array([], dtype = np.int64)

        # get indexes for validation set - each class is processed separately in order to
        # preserve class distribution the same in both training and validation sets - stratified sampling
        for cl in range(10):
            class_indexes = np.where(labels == cl)[0]
            assert(len(class_indexes) == 6000)
            class_indexes_val = np.random.choice(n // 10, val_set_size // 10, replace=False)
            val_indexes = np.concatenate([val_indexes, class_indexes[class_indexes_val]])

        return val_indexes


    # --- Data Augmentation options ---

    def add_noise(self, images, sigma=0.05, non_zero_pixels=False):
        for i, img in enumerate(images):
            if np.random.uniform() < self.augment_prob:
                gauss_noise = np.random.normal(0, sigma, img.shape)
                # if noise should be added only to non-zero pixels
                if non_zero_pixels:
                    gauss_noise[img == 0] = 0
                img = img + gauss_noise
                img[img > 1] = 1
                img[img < 0] = 0
                images[i] = img
            else:
                images[i] = img
        return images

    def adjust_brightness(self, images, min_adj=0.9, max_adj=1.10):
        # brightness adjustment augmentation method will be applied only if relative change in brightness ih bigger than 5%
        assert(min_adj < 0.98)
        assert(max_adj > 1.02)

        for i, img in enumerate(images):
            adj = 1
            while adj > 0.98 and adj < 1.02:
                adj = np.random.uniform(min_adj, max_adj)
            #print("Adjust brightness: " + str(adj))
            if np.random.uniform() < self.augment_prob:
                img = img * adj
                img[img > 1] = 1
                images[i] = img
            else:
                images[i] = img
        return images

    def horizontal_flip(self, images, labels):
        for i, img in enumerate(images):
            # we don't want to flip sandals, sneakers and ankle boots since they oriented in the same direction
            if labels[i] not in [5, 7, 9] and np.random.uniform() < self.augment_prob:
                images[i] = np.fliplr(img)
            else:
                images[i] = img
        return images

    # augmentation method described here: +
    def erase_patch(self, images):
        sl, sh = (0.06, 0.09)
        r1, r2 = (0.5, 2)
        img_w, img_h = images.shape[2], images.shape[1]

        for i, img in enumerate(images):
            if np.random.uniform() < self.augment_prob:
                Se = np.random.uniform(sl, sh) * img_h * img_w
                re = np.random.uniform(r1, r2)
                He = int(np.sqrt(Se * re))
                We = int(np.sqrt(Se / re))
                while True:
                    xe, ye = (np.random.choice(img_w), np.random.choice(img_h))
                    if xe + We >= img_w or ye + He >= img_h:
                        continue
                    else:
                        break
                gauss_noise = np.random.uniform(0, 1, (He, We, 1))
                img[ye:(ye+He),xe:(xe+We),:] = gauss_noise
                images[i] = img
        return images

    def crop(self, images):
        img_w, img_h = images.shape[2], images.shape[1]
        for i, img in enumerate(images):
            if np.random.uniform() < self.augment_prob:
                img_padded = np.zeros((img_h+8, img_w+8, 1))
                img_padded[4:(img_h+4), 4:(img_w+4), :] = img
                xr = np.random.randint(9)
                yr = np.random.randint(9)
                images[i] = img_padded[yr:(img_h+yr), xr:(img_w+xr), :]

        return images