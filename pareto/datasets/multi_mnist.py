from pathlib import Path
import codecs
import gzip
import urllib
import random

import numpy as np
from scipy import ndimage

from PIL import Image

import torch


class MultiMNIST(torch.utils.data.Dataset):
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pth'
    test_file = 'test.pth'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if train:
            self.data, self.labels_l, self.labels_r = torch.load(
                self.root / self.processed_folder /self.training_file)
        else:
            self.data, self.labels_l, self.labels_r = torch.load(
                self.root / self.processed_folder / self.test_file)

        if transform is not None:
            self.data = [self.transform(Image.fromarray(
                img.numpy().astype(np.uint8), mode='L')) for img in self.data]

    def __getitem__(self, index):
        img, target_l, target_r = self.data[index], self.labels_l[index], self.labels_r[index]

        return img, torch.stack([target_l, target_r])

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (self.root / self.processed_folder / self.training_file).is_file() and \
            (self.root / self.processed_folder / self.test_file).is_file()

    def download(self):
        if self._check_exists():
            return

        # download files
        (self.root / self.raw_folder).mkdir(parents=True, exist_ok=True)
        (self.root / self.processed_folder).mkdir(parents=True, exist_ok=True)

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = self.root / self.raw_folder / filename
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(self.root / self.raw_folder / '.'.join(filename.split('.')[:-1]), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            file_path.unlink()

        # process and save as torch files
        print('Processing...')
        multi_mnist_ims, extension = self.read_image_file(
            self.root / self.raw_folder / 'train-images-idx3-ubyte', shift_pix=4, rand_shift=True)
        multi_mnist_labels_l, multi_mnist_labels_r = self.read_label_file(
            self.root / self.raw_folder / 'train-labels-idx1-ubyte', extension)

        tmulti_mnist_ims, textension = self.read_image_file(
            self.root / self.raw_folder / 't10k-images-idx3-ubyte', shift_pix=4, rand_shift=True)
        tmulti_mnist_labels_l, tmulti_mnist_labels_r = self.read_label_file(
            self.root / self.raw_folder / 't10k-labels-idx1-ubyte', textension)

        multi_mnist_training_set = (multi_mnist_ims, multi_mnist_labels_l, multi_mnist_labels_r)
        multi_mnist_test_set = (tmulti_mnist_ims, tmulti_mnist_labels_l, tmulti_mnist_labels_r)

        with open(self.root / self.processed_folder / self.training_file, 'wb') as f:
            torch.save(multi_mnist_training_set, f)
        with open(self.root / self.processed_folder / self.test_file, 'wb') as f:
            torch.save(multi_mnist_test_set, f)
        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def get_int(b):
        return int(codecs.encode(b, 'hex'), 16)

    @staticmethod
    def read_label_file(path, extension):
        with open(path, 'rb') as f:
            data_1 = f.read()
            assert MultiMNIST.get_int(data_1[:4]) == 2049
        with open(path, 'rb') as f:
            data_2 = f.read()
            assert MultiMNIST.get_int(data_2[:4]) == 2049
        length = MultiMNIST.get_int(data_1[4:8])
        parsed_1 = np.frombuffer(data_1, dtype=np.uint8, offset=8)
        parsed_2 = np.frombuffer(data_2, dtype=np.uint8, offset=8)
        multi_labels_l = np.zeros(length, dtype=np.long)
        multi_labels_r = np.zeros(length, dtype=np.long)
        for im_id in range(length):
            multi_labels_l[im_id] = parsed_1[im_id]
            multi_labels_r[im_id] = parsed_2[extension[im_id]]
        return (torch.from_numpy(multi_labels_l).view(-1).long(),
                torch.from_numpy(multi_labels_r).view(-1).long())

    @staticmethod
    def read_image_file(path, shift_pix=4, rand_shift=True, rot_range=(0, 0), corot=True):
        with open(path, 'rb') as f:
            data_1 = f.read()
            assert MultiMNIST.get_int(data_1[:4]) == 2051
        with open(path, 'rb') as f:
            data_2 = f.read()
            assert MultiMNIST.get_int(data_2[:4]) == 2051
        length = MultiMNIST.get_int(data_1[4:8])
        num_rows = MultiMNIST.get_int(data_1[8:12])
        num_cols = MultiMNIST.get_int(data_1[12:16])
        parsed_1 = np.frombuffer(data_1, dtype=np.uint8, offset=16)
        pv_1 = parsed_1.reshape(length, num_rows, num_cols)
        parsed_2 = np.frombuffer(data_2, dtype=np.uint8, offset=16)
        pv_2 = parsed_2.reshape(length, num_rows, num_cols)
        multi_data = np.zeros((length, num_rows, num_cols))
        extension = np.zeros(length, dtype=np.int32)
        rights = np.random.permutation(length)
        for left in range(length):
            extension[left] = rights[left]
            lim = pv_1[left, :, :]
            rim = pv_2[rights[left], :, :]
            if not rot_range[0] == rot_range[1] == 0:
                if corot:
                    rot_deg = random.randint(rot_range[0], rot_range[1])
                    lim = ndimage.rotate(lim, rot_deg, reshape=False)
                    rim = ndimage.rotate(rim, rot_deg, reshape=False)
                else:
                    rot_deg = random.randint(rot_range[0], rot_range[1])
                    lim = ndimage.rotate(lim, rot_deg, reshape=False)
                    rot_deg = random.randint(rot_range[0], rot_range[1])
                    rim = ndimage.rotate(rim, rot_deg, reshape=False)
            # in case of 100% overlapping
            shift_pix1 = shift_pix2 = 0
            if rand_shift:
                if random.choice([True, False]):
                    shift_pix1 = random.randint(0, shift_pix - 1)
                    shift_pix2 = random.randint(0, shift_pix)
                else:
                    shift_pix1 = random.randint(0, shift_pix)
                    shift_pix2 = random.randint(1, shift_pix)
            new_im = np.zeros((36, 36))
            new_im[shift_pix1:shift_pix1 + 28, shift_pix1:shift_pix1 + 28] += lim
            new_im[shift_pix2 + 4:shift_pix2 + 4 + 28, shift_pix2 + 4:shift_pix2 + 4 + 28] += rim
            new_im = np.clip(new_im, 0, 255)
            multi_data_im = np.array(Image.fromarray(new_im).resize((28, 28), resample=Image.NEAREST))
            multi_data[left, :, :] = multi_data_im
        return torch.from_numpy(multi_data).view(length, num_rows, num_cols), extension
