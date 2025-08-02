import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random


class CustomDataset(Dataset):
    def __init__(
            self,
            seqs_labels_path_pair
    ):
        super(CustomDataset, self).__init__()
        self.seqs_labels_path_pair = seqs_labels_path_pair

    def __len__(self):
        return len((self.seqs_labels_path_pair))

    def __getitem__(self, idx):
        seq_path = self.seqs_labels_path_pair[idx][0]
        label_path = self.seqs_labels_path_pair[idx][1]
        subject_id = self.seqs_labels_path_pair[idx][2]
        #print(seq_path)
        seq_eeg = np.load(seq_path)[:, :1, :]
        #print(seq_eeg.shape)
        seq_eog = np.load(seq_path)[:, 1:2, :]
        #print(seq_eog.shape)
        seq = np.concatenate((seq_eeg, seq_eog), axis=1)
        label = np.load(label_path)
        return seq, label, subject_id

    def collate(self, batch):
        x_seq = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        z_label = np.array([x[2] for x in batch]) # subject id
        return to_tensor(x_seq), to_tensor(y_label).long(), to_tensor(z_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets = {
            'sleep-edfx': 0,
            'HMC': 1,
            'ISRUC': 2,
            'SHHS1': 3,
            'P2018': 4,
        }
        # self.datasets = {
        #     'HMC': 0,
        #     'sleep-edfx': 1,
        #     'ISRUC': 2,
        #     'SHHS1': 3,
        #     'P2018': 4,
        # }
        # self.datasets = {
        #     'SHHS1': 0,
        #     'HMC': 1,
        #     'ISRUC': 2,
        #     'sleep-edfx': 3,
        #     'P2018': 4,
        # }
        self.targets_dirs = [f'{self.params.datasets_dir}/{item}' for item in self.datasets.keys() if item in self.params.target_domains]
        self.source_dirs = [f'{self.params.datasets_dir}/{item}' for item in self.datasets.keys() if item not in self.params.target_domains]
        print(self.targets_dirs)
        print(self.source_dirs)

    def get_data_loader(self):
        source_domains, subject_id = self.load_path(self.source_dirs, 0)
        target_domains, _ = self.load_path(self.targets_dirs, subject_id)
        # print(len(target_domains), len(source_domains))
        train_pairs, val_pairs = self.split_dataset(source_domains)
        print(len(train_pairs), len(val_pairs), len(target_domains))
        train_set = CustomDataset(train_pairs)
        val_set = CustomDataset(val_pairs)
        test_set = CustomDataset(target_domains) # target domains is test set
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
                num_workers=self.params.num_workers,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=True,
                num_workers=self.params.num_workers,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=True,
                num_workers=self.params.num_workers,
            ),
        }
        return data_loader, subject_id

    def load_path(self, domains_dirs, subject_id):
        domains = []
        for dataset in domains_dirs:
            seq_dirs = os.listdir(f'{dataset}/seq')
            labels_dirs = os.listdir(f'{dataset}/labels')
            for seq_dir, labels_dir in zip(seq_dirs, labels_dirs):
                seq_names = os.listdir(os.path.join(dataset, 'seq', seq_dir))
                labels_names = os.listdir(os.path.join(dataset, 'labels', labels_dir))
                for seq_name, labels_name in zip(seq_names, labels_names):
                    domains.append((os.path.join(dataset, 'seq', seq_dir, seq_name), os.path.join(dataset, 'labels', labels_dir, labels_name), subject_id))
            subject_id += 1
        return domains, subject_id

    def split_dataset(self, source_domains):
        random.shuffle(source_domains)
        split_num = int(len(source_domains) * 0.8)
        train_pairs = source_domains[:split_num]
        val_pairs = source_domains[split_num:]
        return train_pairs, val_pairs

