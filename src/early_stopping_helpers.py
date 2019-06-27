from itertools import count

import torch
from torch import nn

def eval_model(test_dataloader, model, device):
  model.eval()
  with torch.no_grad():
    ctr = count()
    num_correct = 0
    criteria = nn.BCEWithLogitsLoss()
    loss = 0
    for batch in test_dataloader:
      (candidate_ids, features), target_rankings = batch
      features = features.to(device)
      target = [ranking[0] for ranking in target_rankings]
      candidate_scores = model(features)
      label = torch.ones(len(features), device=device)
      ctr2 = count()
      for ids, true_id in zip(candidate_ids, target):
        for entity_id, idx in zip(ids, ctr2):
          if entity_id == true_id: label[idx] = 1
      loss += criteria(candidate_scores, label) / len(candidate_ids)
      top_1 = []
      offset = 0
      for ids in candidate_ids:
        ranking_size = len(ids)
        top_1.append(ids[torch.argmax(candidate_scores[offset : offset + ranking_size]).item()])
        offset += ranking_size
      for guess, label, idx in zip(top_1, target, ctr): # pylint: disable=unused-variable
        if guess == label: num_correct += 1
    return {'acc': num_correct / next(ctr), 'bce_loss': loss.item()}
