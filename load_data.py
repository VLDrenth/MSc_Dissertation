import ddu_dirty_mnist
import os
import torch

def download_dirty_mnist():
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    dirty_mnist_train = ddu_dirty_mnist.DirtyMNIST(root=data_dir, train=True, download=True, device='cpu')
    dirty_mnist_test = ddu_dirty_mnist.DirtyMNIST(root=data_dir, train=False, download=True, device='cpu')
    return data_dir, dirty_mnist_train, dirty_mnist_test

def create_data_loader(data, shuffle, batch_size=32):
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=128,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )
    return loader

def load_dirty_mnist():
    data_dir, dirty_mnist_train, dirty_mnist_test = download_dirty_mnist()
    train_loader = create_data_loader(dirty_mnist_train, shuffle=True)
    test_loader = create_data_loader(dirty_mnist_test, shuffle=False)
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = load_dirty_mnist()
    print('Data loaded successfully!')