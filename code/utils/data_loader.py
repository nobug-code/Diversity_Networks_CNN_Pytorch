import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

def get_data_loader(args):

    dataset = args.dataset
    
    if(dataset == 'mnist'):
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(([0.5]),([0.5])),
            ])
        train_dataset = dset.MNIST(root='/home/nkim/data', train=True, download=args.download, transform=trans)
        test_dataset = dset.MNIST(root='/home/nkim/data', train=False, download=args.download, transform=trans)

    elif(dataset == 'cifar10'):
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])
        tests = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

        ])

        train_dataset = dset.CIFAR10(root='/home/nkim/data', train=True, download=args.download, transform=trans)
        test_dataset = dset.CIFAR10(root='/home/nkim/data', train=False, download=args.download, transform=tests)

    assert train_dataset
    assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_dataloader, test_dataloader
