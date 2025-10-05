import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

def create_non_iid_splits(dataset, num_clients, alpha=0.5, seed=42):
    """Dirichlet non-IID partitioning."""
    np.random.seed(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))

    label_dist = np.random.dirichlet([alpha]*num_classes, num_clients)
    class_idxs = [np.where(labels == i)[0] for i in range(num_classes)]
    client_idxs = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        np.random.shuffle(class_idxs[c])
        proportions = label_dist[:, c]
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(class_idxs[c])).astype(int)
        splits = np.concatenate([[0], splits])
        for i in range(num_clients):
            client_idxs[i].extend(class_idxs[c][splits[i]:splits[i+1]].tolist())

    for i in range(num_clients):
        np.random.shuffle(client_idxs[i])
    return client_idxs

def get_cifar10_loaders(root="./data", batch_size_test=128):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    test  = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    test_loader = DataLoader(test, batch_size=batch_size_test, shuffle=False)
    return train, test_loader

def make_client_subsets(train_dataset, client_indices):
    from .fl_server import Client  # local import to avoid cycle (only for typing)
    subsets = [Subset(train_dataset, idxs) for idxs in client_indices]
    return subsets
