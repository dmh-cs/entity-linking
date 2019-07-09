"""Assuming tree with root at idx 0 and leaves at idxs > 0"""

from collections import OrderedDict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from utils import elim_dim


def compatibility_from_ids(entity_id_to_row, compats, candidate_ids):
  dim = OrderedDict([('root_mention_dim' , 0),
                     ('leaf_mention_dim' , 1),
                     ('leaf_option_dim'  , 2),
                     ('root_option_dim'  , 3)])
  num_mentions = len(candidate_ids)
  num_options = max(map(len, candidate_ids))
  cand_compats = torch.zeros(num_mentions, num_mentions, num_options, num_options)
  row_nums = [entity_id_to_row.get(cand_id)
              for ids in candidate_ids
              for cand_id in ids]
  for root_mention_idx, idxs in enumerate(row_nums):
    for leaf_mention_idx, idxs in enumerate(row_nums):
      for

def mp_doc(emissions, compatibilities, emission_weight):
  dim = OrderedDict([('root_mention_dim' , 0),
                     ('leaf_mention_dim' , 1),
                     ('leaf_option_dim'  , 2),
                     ('root_option_dim'  , 3)])
  num_mentions, num_options = emissions.shape
  shaped_emissions = emissions.reshape(1, num_mentions, num_options, 1)
  option_contributions = compatibilities + shaped_emissions
  max_contribution = np.max(option_contributions, dim['leaf_option_dim'])
  dim = elim_dim(dim, 'leaf_option_dim')
  contributions = np.sum(max_contribution, dim['leaf_mention_dim'])
  dim = elim_dim(dim, 'leaf_mention_dim')
  root_scores = (1-emission_weight) * contributions + emission_weight * emissions
  return np.argmax(root_scores, dim['root_option_dim'])
