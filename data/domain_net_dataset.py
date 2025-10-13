from configs.default import domain_net_path
import os
import torch
from data.meta_dataset import MetaDataset, GetDataLoaderDict
from torchvision import transforms
import numpy as np
import random
domain_net_name_dict = {
    'c': 'clipart',
    'i': 'infograph',
    'p': 'painting',
    'q': 'quickdraw',
    'r': 'real',
    's': 'sketch',    
}

transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale( 0.1),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

transform_test = transforms.Compose(
            [transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

class domain_net_SingleDomain():
    def __init__(self, root_path=domain_net_path, domain_name='100', split='train', train_transform=None, seed=0):
        self.domain_name = domain_name
        assert domain_name in domain_net_name_dict.keys(), 'domain_name must be in {}'.format(domain_net_name_dict.keys())
        self.root_path = root_path
        self.domain = domain_net_name_dict[domain_name]
        self.domain_label = list(domain_net_name_dict.keys()).index(domain_name)
        self.txt_path = os.path.join(root_path, '{}_img_label_list.txt'.format(self.domain))
        
        self.split = split
        assert self.split in ['train', 'val', 'test'] , 'split must be train, val or test'
        
        if train_transform is not None:
            self.transform = train_transform
        else:
            self.transform = transform_test
        self.seed = seed
        
        self.imgs, self.labels = domain_net_SingleDomain.read_txt(self.txt_path)
        
        if self.split == 'train' or self.split == 'val':
            random.seed(self.seed)
            train_img, val_img = domain_net_SingleDomain.split_list(self.imgs, 0.9)
            random.seed(self.seed)
            train_label, val_label = domain_net_SingleDomain.split_list(self.labels, 0.9)
            if self.split == 'train':
                self.imgs, self.labels = train_img, train_label
            elif self.split == 'val':
                self.imgs, self.labels = val_img, val_label
                
        self.dataset = MetaDataset(self.imgs, self.labels, self.domain_label, self.transform) # get数据集
    
    @staticmethod
    def split_list(l, ratio):
        assert ratio > 0 and ratio < 1
        random.shuffle(l)
        train_size = int(len(l)*ratio)
        train_l = l[:train_size]
        val_l = l[train_size:]
        return train_l, val_l
        
    @staticmethod
    def read_txt(txt_path):
        imgs = []
        labels = []
        with open(txt_path, 'r') as f:
            contents = f.readlines()
            
        for line_txt in contents:
            line_txt = line_txt.replace('\n', '')
            line_txt_list = line_txt.split(' ')
            imgs.append(line_txt_list[0])
            labels.append(int(line_txt_list[1]))
            
        return imgs, labels
    
class domain_net_FedDG():
    def __init__(self, test_domain='100', batch_size=64, test_batch_size=8, seed=0):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.domain_list = list(domain_net_name_dict.keys())
        self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain)  
        self.seed = seed
        
        self.site_dataset_dict = {}
        self.site_dataloader_dict = {}
        for domain_name in self.domain_list:
            self.site_dataloader_dict[domain_name], self.site_dataset_dict[domain_name] = domain_net_FedDG.SingleSite(
                domain_name, 
                self.batch_size, 
                self.test_batch_size if domain_name == self.test_domain else self.batch_size,
                self.seed)
            
        
        self.test_dataset = self.site_dataset_dict[self.test_domain]['test']
        self.test_dataloader = self.site_dataloader_dict[self.test_domain]['test']
        
          
    @staticmethod
    def SingleSite(domain_name, batch_size=64, test_batch_size=8, seed=0):
        dataset_dict = {
            'train': domain_net_SingleDomain(domain_name=domain_name, split='train', train_transform=transform_train, seed=seed).dataset,
            'val': domain_net_SingleDomain(domain_name=domain_name, split='val', seed=seed).dataset,
            'test': domain_net_SingleDomain(domain_name=domain_name, split='test', seed=seed).dataset,
        }
        dataloader_dict = {
            'train': torch.utils.data.DataLoader(dataset_dict['train'], batch_size=batch_size, shuffle=True),
            'val': torch.utils.data.DataLoader(dataset_dict['val'], batch_size=batch_size, shuffle=False),
            'test': torch.utils.data.DataLoader(dataset_dict['test'], batch_size=test_batch_size, shuffle=False)
        }
        return dataloader_dict, dataset_dict
        
    def GetData(self):
        return self.site_dataloader_dict, self.site_dataset_dict

def GenFileList():
    '''
    get path label list on all domains
    '''
    total_class_list = None
    for domain_name in domain_net_name_dict.keys():
        domain = domain_net_name_dict[domain_name]
        domain_path = os.path.join(domain_net_path, domain)
        class_list = os.listdir(domain_path)
        class_list.sort()
        if total_class_list is None:
            total_class_list = class_list
        
        assert total_class_list == class_list, 'class_list must be same'
    
    domain_file_dict = {}
    for domain_name in domain_net_name_dict.keys():
        domain_file_dict[domain_name] = []
        domain = domain_net_name_dict[domain_name]
        domain_path = os.path.join(domain_net_path, domain)
        for label_idx, class_name in enumerate(total_class_list):
            class_path = os.path.join(domain_path, class_name)
            file_list = os.listdir(class_path)
            
            for file_name in file_list:
                file_path = os.path.join(class_path, file_name)
                domain_file_dict[domain_name].append(file_path + ' ' + str(label_idx) + '\n')
        
    return domain_file_dict

