import numpy as np


def mp_tree_depth_1(calc_emissions, calc_compatibilities) -> int:
  """Assuming tree with root at idx 0 and leaves at idxs > 0"""
  def _calc_message(calc_emission, calc_compatibility):
    emission = calc_emission()
    compatibility = calc_compatibility()
    best_compatibility = np.max(compatibility, 0)
    return emission + best_compatibility
  messages = np.stack([_calc_message(calc_emission, calc_compatibility)
                       for calc_emission, calc_compatibility in zip(calc_emissions[1:],
                                                                    calc_compatibilities)])
  root_scores = calc_emissions[0]() + np.sum(messages, 0)
  return np.argmax(root_scores)
