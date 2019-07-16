from collections import OrderedDict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def emissions_from_flat_scores(lens, flat_scores):
  scores = []
  offset = 0
  for length in lens:
    section = torch.tensor(flat_scores[offset : offset + length])
    section_min = section.min()
    scale = section.max() - section_min
    if scale == 0.0:
      scores.append(torch.log(torch.ones_like(section)))
    else:
      scores.append(torch.log(torch.softmax((section - section_min) / scale,
                                            0)))
    offset += length
  return [score.numpy() for score in scores]

def compatibilities_from_ids(entity_id_to_row, desc_vs, norm, candidate_ids):
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
              prod = desc_vs[root].multiply(desc_vs[leaf]).sum()
              div = norm[root].item() * norm[leaf].item()
              if div > 0:
                compat = (prod / div)
              else:
                compat = 0.0
              row.append(compat)
        edge_factor.append(row)
      result = np.array(edge_factor)
      factor_normalization = result.sum()
      if factor_normalization == 0.0:
        factors.append(np.log(np.ones_like(result) / result.size))
      else:
        factors.append(np.log(result / result.sum()))
    compat_with_roots.append(factors)
  return compat_with_roots

def mp_tree_depth_1(root_emission, leaf_emissions, compat_with_root) -> int:
  def _message(emission, compatibility):
    leaf_option_dim = 0
    root_option_dim = 1
    return np.max(emission.reshape(-1, 1) + compatibility, leaf_option_dim)
  if len(leaf_emissions) == 0: return np.argmax(root_emission)
  to_stack = [torch.tensor(_message(emission, compatibility))
              for emission, compatibility in zip(leaf_emissions,
                                                 compat_with_root)]
  messages = np.stack(pad_sequence(to_stack, batch_first=True, padding_value=-np.inf))
  leaf_mention_dim = 0
  root_scores = root_emission + np.sum(messages, leaf_mention_dim)
  return np.argmax(root_scores)

def mp_shallow_tree_doc(emissions, compatibilities):
  results = []
  gen = enumerate(zip(emissions, compatibilities))
  for root_idx, (root_emission, compat_with_root) in gen:
    leaf_emissions = np.delete(emissions, root_idx, 0)
    results.append(mp_tree_depth_1(root_emission,
                                   leaf_emissions,
                                   compat_with_root))
  return results
