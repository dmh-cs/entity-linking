import warnings
import numpy as np
import torch
from scipy.sparse import dok_matrix

from numpy.testing import assert_equal

from max_product import (mp_tree_depth_1,
                         mp_shallow_tree_doc,
                         emissions_from_flat_scores,
                         compatibilities_from_ids)

def test_mp_tree_depth_1_1():
  root_emission = np.array([1.0, 0.0])
  leaf_emissions = np.array([[1.0, 0.0]])
  compat_with_root = np.array([[[1.0, 0.0], [1.0, 0.0]]])
  assert mp_tree_depth_1(root_emission, leaf_emissions, compat_with_root) == 0

def test_mp_tree_depth_1_2():
  root_emission = np.array([0.5, 0.5])
  leaf_emissions = np.array([[0.5, 0.5]])
  compat_with_root = np.array([[[1.0, 0.0], [1.0, 0.0]]])
  assert mp_tree_depth_1(root_emission, leaf_emissions, compat_with_root) == 0

def test_mp_tree_depth_1_3():
  root_emission = np.array([0.5, 0.5])
  leaf_emissions = np.array([[0.5, 0.5]])
  compat_with_root = np.array([[[0.0, 1.0], [0.9, 0.1]]])
  assert mp_tree_depth_1(root_emission, leaf_emissions, compat_with_root) == 1

def test_mp_tree_depth_1_4():
  root_emission = np.array([0.3, 0.7])
  leaf_emissions = np.array([[0.2, 0.8],
                             [1.0, 0.0]])
  compat_with_root = np.array([[[0.9, 0.1], [0.6, 0.4]],
                               [[0.99, 0.01], [0.0, 0.0]]])
  assert mp_tree_depth_1(root_emission, leaf_emissions, compat_with_root) == 0

def test_mp_shallow_tree_doc():
  emissions = np.array([[0.3, 0.7],
                        [0.2, 0.8],
                        [1.0, 0.0]])
  compatibilities = np.array([[[[0.9, 0.1], [0.6, 0.4]],
                               [[0.0, 1.0], [0.0, 0.0]]],
                              [[[0.9, 0.1], [0.6, 0.4]],
                               [[0.0, 1.0], [0.0, 0.0]]],
                              [[[0.0, 1.0], [0.0, 0.0]],
                               [[0.0, 1.0], [0.0, 0.0]]]])
  assert mp_shallow_tree_doc(emissions, compatibilities) == [1, 1, 1]

def test_emissions_from_flat_scores():
  flat_scores = torch.tensor([1, 2, 3, 5, 4]).float()
  lens = [2, 3]
  expected = [torch.log(torch.softmax(torch.tensor(row).float(), 0)).tolist()
              for row in [[1, 2, -np.inf], [3, 5, 4]]]
  assert emissions_from_flat_scores(lens, flat_scores).tolist() == expected

def test_compatibilities_from_ids_1():
  def normalize(coll): return np.array(coll) / np.array(coll).sum()
  entity_id_to_row = dict(zip(range(10, 20), range(0, 10)))
  compats = dok_matrix((len(entity_id_to_row), len(entity_id_to_row)))
  c_12_14 = 20
  c_12_19 = 30
  c_14_19 = 50
  c_13_14 = 0
  c_13_19 = 0
  c_14_20 = 0
  c_12_20 = 0
  c_13_20 = 0
  compats[2, 4] = c_12_14
  compats[4, 2] = c_12_14
  compats[2, 9] = c_12_19
  compats[9, 2] = c_12_19
  compats[4, 9] = c_14_19
  compats[9, 4] = c_14_19
  candidate_ids = [[12, 13], [14], [19, 20]]
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    compatibilities = compatibilities_from_ids(entity_id_to_row,
                                               compats,
                                               candidate_ids)
    mention_0 = [np.log(normalize([[c_12_14, c_13_14]])),
                 np.log(normalize([[c_12_19, c_13_19],
                                   [c_12_20, c_13_20]]))]
    mention_1 = [np.log(normalize([[c_12_14],
                                   [c_13_14]])),
                 np.log(normalize([[c_14_19],
                                   [c_14_20]]))]
    mention_2 = [np.log(normalize([[c_12_19, c_12_20],
                                   [c_13_19, c_13_20]])),
                 np.log(normalize([[c_14_19, c_14_20]]))]
  expected = [mention_0, mention_1, mention_2]
  assert_equal(compatibilities, expected)

def test_compatibilities_from_ids_2():
  def normalize(coll): return np.array(coll) / np.array(coll).sum()
  entity_id_to_row = dict(zip(range(10, 21), range(0, 11)))
  compats = dok_matrix((len(entity_id_to_row), len(entity_id_to_row)))
  c_12_14 = 20
  c_12_19 = 30
  c_14_19 = 50
  c_13_14 = 40
  c_13_19 = 70
  c_14_20 = 30
  c_12_20 = 40
  c_13_20 = 20
  compats[2, 4] = c_12_14
  compats[4, 2] = c_12_14
  compats[2, 9] = c_12_19
  compats[9, 2] = c_12_19
  compats[4, 9] = c_14_19
  compats[9, 4] = c_14_19
  compats[3, 4] = c_13_14
  compats[4, 3] = c_13_14
  compats[3, 9] = c_13_19
  compats[9, 3] = c_13_19
  compats[4, 10] = c_14_20
  compats[10, 4] = c_14_20
  compats[2, 10] = c_12_20
  compats[10, 2] = c_12_20
  compats[3, 10] = c_13_20
  compats[10, 3] = c_13_20
  candidate_ids = [[12, 13], [14], [19, 20]]
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    compatibilities = compatibilities_from_ids(entity_id_to_row,
                                               compats,
                                               candidate_ids)
    mention_0 = [np.log(normalize([[c_12_14, c_13_14]])),
                 np.log(normalize([[c_12_19, c_13_19],
                                   [c_12_20, c_13_20]]))]
    mention_1 = [np.log(normalize([[c_12_14],
                                   [c_13_14]])),
                 np.log(normalize([[c_14_19],
                                   [c_14_20]]))]
    mention_2 = [np.log(normalize([[c_12_19, c_12_20],
                                   [c_13_19, c_13_20]])),
                 np.log(normalize([[c_14_19, c_14_20]]))]
  expected = [mention_0, mention_1, mention_2]
  assert_equal(compatibilities, expected)
