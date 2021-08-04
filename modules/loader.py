import torch, os
from torchvision import datasets, transforms
from data.mvtec import MVTecDataset
from data.cloth import ClothDataset
from data.other import OtherDataset

def get_loader(config='mvtec', class_name='bottle', img_size=256, is_train=True, batch_size=32, val_rate=0.1):
    if config == 'mvtec':
        dataset = MVTecDataset(class_name=class_name, is_train=is_train, resize=img_size)
    elif config == 'cloth':
        dataset = ClothDataset(class_name=class_name, is_train=is_train, resize=img_size)
    elif config == 'other':
        dataset = OtherDataset(class_name=class_name, is_train=is_train, resize=img_size)
    else:
        print('dataset error')

    if is_train:
        n_samples = len(dataset)
        val_size = int(len(dataset) * val_rate)
        train_size = n_samples - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=is_train,
            drop_last=is_train,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
        print('Train: {} ({}), Val: {} ({})'.format(train_size, len(train_loader), val_size, len(val_loader)))
        return train_loader, val_loader
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
        print('Test: {} ({})'.format(len(dataset), len(loader)))
        return loader
