from batchbald_redux import repeated_mnist, active_learning
from torch.utils import data
import torch
from torchvision import datasets, transforms

def create_MNIST_dataloaders(config, **kwargs):
    # loading data
    train_dataset, test_dataset = repeated_mnist.create_MNIST_dataset()

    # Create data loaders
    train_loader, test_loader, pool_loader, active_learning_data = create_dataloaders_AL(train_dataset, test_dataset, config, kwargs)

    return train_loader, test_loader, pool_loader, active_learning_data


def create_repeated_MNIST_dataloaders(config, **kwargs):
    # Set up transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load and prepare train dataset
    train_dataset = datasets.MNIST("data", train=True, download=False, transform=transform)
    
    train_dataset_indices = active_learning.get_balanced_sample_indices(
        repeated_mnist.get_targets(train_dataset),
        num_classes=config.num_classes,
        n_per_digit=config.samples_per_digit
    )
    
    train_dataset = data.Subset(train_dataset, train_dataset_indices)
    train_dataset = data.ConcatDataset([train_dataset] * config.num_repeats)
    
    # Add noise to the dataset
    dataset_noise = torch.empty((len(train_dataset), 28, 28), dtype=torch.float32).normal_(0.0, 0.1)
    
    def apply_noise(idx, sample):
        data, target = sample
        return data + dataset_noise[idx], target
    
    train_dataset = repeated_mnist.TransformedDataset(train_dataset, transformer=apply_noise)

    # Load test dataset
    test_dataset = datasets.MNIST("data", train=False, download=False, transform=transform)

    # Create data loaders
    train_loader, test_loader, pool_loader, active_learning_data = create_dataloaders_AL(train_dataset, test_dataset, config)
    
    return train_loader, test_loader, pool_loader, active_learning_data

def create_dataloaders_AL(train_dataset, test_dataset, config, **kwargs):
    # Get indices of initial samples
    initial_samples = active_learning.get_balanced_sample_indices(
        repeated_mnist.get_targets(train_dataset),
        num_classes=config.num_classes,
        n_per_digit=config.num_initial_samples / config.num_classes
    )

    # Create data loaders
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        **kwargs
    )

    active_learning_data = active_learning.ActiveLearningData(train_dataset)
    active_learning_data.acquire(initial_samples)
    active_learning_data.extract_dataset_from_pool(config.extract_pool)

    train_loader = torch.utils.data.DataLoader(
        active_learning_data.training_dataset,
        sampler=active_learning.RandomFixedLengthSampler(
            active_learning_data.training_dataset,
            config.training_iterations
        ),
        batch_size=config.train_batch_size,
        **kwargs
    )

    pool_loader = torch.utils.data.DataLoader(
        active_learning_data.pool_dataset,
        batch_size=config.scoring_batch_size,
        shuffle=False,
        **kwargs
    )

    return train_loader, test_loader, pool_loader, active_learning_data