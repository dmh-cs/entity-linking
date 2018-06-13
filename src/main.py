import torch
import torch.nn as nn
from trainer import Trainer
from description_encoder_model import DescriptionEncoder
from data_fetchers import get_connection, get_train_validation_test_cursors, close_cursors, get_entity_lookup, get_embedding_lookup
from description_dataset import build_datasets, get_data_splits
import pydash as _
import math

def drop_unneeded_entities(data_splits, entity_lookup):
  titles = set(sum([[page['title'] for page in raw_dataset] for raw_dataset in data_splits.values()],
                []))
  to_drop = []
  for entity_name in entity_lookup.keys():
    if entity_name not in titles:
      to_drop.append(entity_name)
  for entity_name in to_drop:
    entity_lookup.pop(entity_name)

def main():
  try:
    batch_size = 1000
    db_connection = get_connection()
    cursors = get_train_validation_test_cursors(db_connection)
    print('Fetching datasets')
    data_splits = get_data_splits(cursors, {'train': 2})
    print('Creating entity lookup')
    entity_lookup = get_entity_lookup()
    # drop_unneeded_entities(data_splits, entity_lookup)
    num_entities = len(entity_lookup)
    embed_len = 100
    entity_embed_weights = nn.Parameter(torch.Tensor(num_entities, embed_len))
    entity_embed_weights.data.normal_(0, 1.0/math.sqrt(embed_len))
    entity_embeds = nn.Embedding(num_entities, embed_len, _weight=entity_embed_weights)
    print('Creating word embedding lookup')
    embedding_lookup = get_embedding_lookup('./glove.6B.100d.txt')
    print('Building datasets')
    datasets = build_datasets(data_splits,
                              entity_lookup,
                              embedding_lookup,
                              batch_size)
    desc_encoder = DescriptionEncoder(embed_len, entity_embeds)
    print('Training')
    trainer = Trainer(model=desc_encoder, datasets=datasets)
    trainer.train()

  finally:
    close_cursors(cursors)
    db_connection.close()


if __name__ == "__main__": main()
