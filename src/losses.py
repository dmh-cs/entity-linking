import torch
import torch.nn.functional as F

def hinge_loss(scores, labels_for_batch):
  score_1, score_2 = [], []
  for batch, label in zip(scores, labels_for_batch):
    score_1.append(batch[label].repeat(len(batch) - 1))
    score_2.append(batch[[idx for idx in range(len(batch)) if idx != label]])
  score_1 = torch.cat(score_1)
  score_2 = torch.cat(score_2)
  return F.margin_ranking_loss(score_1, score_2, torch.ones_like(score_1), margin=1.0)
