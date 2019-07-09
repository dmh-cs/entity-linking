from collections import OrderedDict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def emissions_from_flat_scores(lens, flat_scores):
  scores = []
  offset = 0
  for length in lens:
    scores.append(torch.log(torch.softmax(torch.tensor(flat_scores[offset : offset + length]),
                                          0)))
    offset += length
  return pad_sequence(scores,
                      batch_first=True,
                      padding_value=-np.inf).numpy()

def compatibilities_from_ids(entity_id_to_row, compats, candidate_ids):
  row_nums = [[entity_id_to_row.get(cand_id) for cand_id in ids]
              for ids in candidate_ids]
  compat_with_roots = []
  for root_idx, root_options in enumerate(row_nums):
    factors = []
    for leaf_idx, leaf_options in enumerate(row_nums):
      if root_idx == leaf_idx: continue
      edge_factor = []
      for leaf in leaf_options:
        if leaf is None:
          row = [0.0 for __ in range(len(root_options))]
        else:
          row = []
          for root in root_options:
            if root is None:
              row.append(0.0)
            else:
              row.append(compats[root, leaf])
        edge_factor.append(row)
      result = np.array(edge_factor)
      factors.append(np.log(result / result.sum()))
    compat_with_roots.append(factors)
  return compat_with_roots

def mp_tree_depth_1(root_emission, leaf_emissions, compat_with_root) -> int:
  def _message(emission, compatibility):
    option_dim = 0
    best_compatibility = np.max(compatibility, option_dim)
    return emission + best_compatibility
  messages = np.stack([_message(emission, compatibility)
                       for emission, compatibility in zip(leaf_emissions,
                                                          compat_with_root)])
  leaf_mention_dim = 0
  root_scores = root_emission + np.sum(messages, leaf_mention_dim)
  return np.argmax(root_scores)

def mp_shallow_tree_doc(emissions, compatibilities):
  results = []
  gen = enumerate(zip(emissions, compatibilities))
  for root_idx, (root_emission, compat_with_root) in gen:
    leaf_emissions = np.delete(emissions, root_idx)
    results.append(mp_tree_depth_1(root_emission,
                                   leaf_emissions,
                                   compat_with_root))
  return results
