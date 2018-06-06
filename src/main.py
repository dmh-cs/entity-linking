from trainer import Trainer
from description_encoder_model import DescriptionEncoder
from data_fetchers import get_connection, get_train_validation_test_cursors, get_raw_datasets, close_cursors

def main():
  try:
    db_connection = get_connection()
    cursors = get_train_validation_test_cursors(db_connection)
    raw_datasets = get_raw_datasets(cursors, {'train': 10000, 'validation': 1000, 'test': 1000})
    desc_encoder = DescriptionEncoder()
    trainer = Trainer(model=desc_encoder, raw_datasets=raw_datasets)
    trainer.train()

  finally:
    close_cursors(cursors)
    db_connection.close()


if __name__ == "__main__": main()
