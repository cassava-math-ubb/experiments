import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler, DistributedSampler, SequentialSampler

from utils.leaf_dataset import CassavaLeafDataset

from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

splits = KFold(n_splits = 5, shuffle = True, random_state = 42)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    else:
        full_dataset = CassavaLeafDataset(root_dir=f"{args.dataset_path}/train", annotation_file=args.train_annotations, transform=transform_train)
        train_idx = None
        test_idx = None
        for fold, (train_ix, valid_ix) in enumerate(splits.split(full_dataset)):
            train_idx = train_ix
            test_idx = valid_ix

    if args.local_rank == 0:
        torch.distributed.barrier()
    
    if args.split_data:
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        train_loader = torch.utils.data.DataLoader(full_dataset,   
                                                batch_size=args.train_batch_size, 
                                                sampler=train_sampler, 
                                                num_workers=4,
                                                pin_memory=True)
        test_loader = torch.utils.data.DataLoader(full_dataset, 
                                                batch_size=args.eval_batch_size, 
                                                sampler=test_sampler, 
                                                num_workers=4,
                                                pin_memory=True)
    else:
        train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
        test_sampler = SequentialSampler(testset)
        train_loader = DataLoader(trainset,
                                sampler=train_sampler,
                                batch_size=args.train_batch_size,
                                num_workers=4,
                                pin_memory=True)
        test_loader = DataLoader(testset,
                                sampler=test_sampler,
                                batch_size=args.eval_batch_size,
                                num_workers=4,
                                pin_memory=True) if testset is not None else None

    return train_loader, test_loader
