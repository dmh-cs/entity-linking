from itertools import count

import torch
from torch import nn
from progressbar import progressbar

def eval_model(test_dataloader, model):
  model.eval()
  ctr = count()
  num_correct = 0
  criteria = nn.BCEWithLogitsLoss()
  loss = 0
  for batch in progressbar(test_dataloader):
    (candidate_ids, features), target_rankings = batch
    target = [ranking[0] for ranking in target_rankings]
    candidate_scores = model(features)
    loss += criteria(torch.tensor(candidate_scores), torch.tensor(target))
    top_1 = []
    offset = 0
    for ids in candidate_ids:
      ranking_size = len(ids)
      top_1.append(ids[torch.argmax(candidate_scores[offset : offset + ranking_size]).item()])
      offset += ranking_size
    for guess, label, idx in zip(top_1, target, ctr): # pylint: disable=unused-variable
      if guess == label: num_correct += 1
  return num_correct / next(ctr), loss.item()
