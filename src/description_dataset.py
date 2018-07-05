import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pydash as _

from data_transformers import page_to_desc_encoder_input

def build_datasets(data_splits, entity_lookup, embedding_lookup, batch_size):
  datasets = {}
  for dataset_name, data in data_splits.items():
    dataset = DescriptionDataset(data,
                                 entity_lookup,
                                 embedding_lookup)
    shuffle = dataset_name == 'train'
    datasets[dataset_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  return datasets

def get_data_splits(cursors, num_samples_per_dataset):
  split_index = 0
  raw_datasets = {}
  for dataset_name, num_samples in num_samples_per_dataset.items():
    cursor = cursors[dataset_name]
    raw_datasets[dataset_name] = _get_data(cursor, split_index, num_samples)
    split_index += num_samples
  return raw_datasets

def _get_data(cursor, start_id, max_num_samples):
  cursor.execute('select * from pages where is_seed_page = 1 limit %s, %s',
                 (start_id, max_num_samples))
  return cursor.fetchall()

class DescriptionDataset(Dataset):
  def __init__(self, data, entity_lookup, embedding_lookup, transform=None):
    self.entity_lookup = entity_lookup
    self.embedding_lookup = embedding_lookup
    self.data = data
    self.num_samples = len(self.data)
    self.transform = transform
    self.entity_id_lookup = {}
    self.num_classes = 0

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    description, db_id = page_to_desc_encoder_input(self.entity_lookup, self.embedding_lookup, self.data[idx])
    if db_id not in self.entity_id_lookup:
      self.entity_id_lookup[db_id] = self.num_classes
      self.num_classes += 1
    sample = {'description': description, 'label': self.entity_id_lookup[db_id]}
    if self.transform:
      sample = self.transform(sample)
    return sample
