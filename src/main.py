import torch
import torch.nn as nn
from trainer import Trainer
from description_encoder_model import DescriptionEncoder
from data_fetchers import get_connection, get_train_validation_test_cursors, get_raw_datasets, close_cursors, get_entity_lookup, get_embedding_lookup
from data_transformers import transform_raw_datasets
import pickle

def main():
  try:
    db_connection = get_connection()
    cursors = get_train_validation_test_cursors(db_connection)
    print('Fetching raw datasets')
    raw_datasets = get_raw_datasets(cursors, {'train': 10000, 'validation': 1000, 'test': 1000})
    print('Creating entity lookup')
    entity_lookup = get_entity_lookup()
    embed_len = 200
    num_embeddings = len(entity_lookup)
    embed_weights = nn.Parameter(torch.Tensor(num_embeddings, embed_len))
    embed_weights.data.normal_(0, 1/embed_len)
    entity_embeds = nn.Embedding(num_embeddings, embed_len, _weight=embed_weights)
    print('Creating embedding lookup')
    embedding_lookup = get_embedding_lookup('./glove.6B.100d.txt')
    print('Transforming raw datasets')
    datasets = transform_raw_datasets(entity_lookup, embedding_lookup, raw_datasets)
    desc_encoder = DescriptionEncoder(embed_len, entity_embeds)
    print('Training')
    trainer = Trainer(model=desc_encoder, datasets=datasets)
    trainer.train()

  finally:
    close_cursors(cursors)
    db_connection.close()


if __name__ == "__main__": main()
