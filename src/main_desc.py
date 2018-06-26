import torch
import torch.nn as nn
from desc_trainer import DescTrainer
from description_encoder_model import DescriptionEncoder
from data_fetchers import get_connection, get_train_validation_test_cursors, close_cursors, get_entity_lookup, get_embedding_lookup
from description_dataset import build_datasets, get_data_splits
import math

def main():
  try:
    batch_size = 1000
    num_epochs = 1000
    db_connection = get_connection()
    cursors = get_train_validation_test_cursors(db_connection)
    print('Fetching datasets')
    data_splits = get_data_splits(cursors, {'train': 1000})
    print('Creating entity lookup')
    entity_lookup = get_entity_lookup()
    num_entities = len(entity_lookup)
    embed_len = 100
    entity_embed_weights = nn.Parameter(torch.Tensor(num_entities, embed_len))
    entity_embed_weights.data.normal_(0, 1.0/math.sqrt(embed_len))
    entity_embeds = nn.Embedding(num_entities, embed_len, _weight=entity_embed_weights)
    print('Creating word embedding lookup')
    embedding_lookup = get_embedding_lookup('./glove.6B.100d.txt')
    word_embed_len = 100
    print('Building datasets')
    datasets = build_datasets(data_splits,
                              entity_lookup,
                              embedding_lookup,
                              batch_size)
    desc_encoder = DescriptionEncoder(word_embed_len, entity_embeds)
    print('Training')
    trainer = DescTrainer(model=desc_encoder,
                          datasets=datasets,
                          num_epochs=num_epochs)
    trainer.train(batch_size)

  finally:
    close_cursors(cursors)
    db_connection.close()


if __name__ == "__main__": main()
