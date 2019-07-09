import numpy as np

from max_product_shallow_tree import mp_doc

def test_mp_doc_1():
  a_emissions = [1.0, 0.0]
  b_emissions = [1.0, 0.0]
  emissions = np.array([a_emissions,
                        b_emissions])
  compat_with_a = [[[0.0, 0.0], [0.0, 0.0]],
                   [[1.0, 0.0], [1.0, 0.0]]]
  compat_with_b = [[[1.0, 0.0], [1.0, 0.0]],
                   [[0.0, 0.0], [0.0, 0.0]]]
  compatibilities = np.array([compat_with_a,
                              compat_with_b])
  assert mp_doc(emissions, compatibilities, 0.5).tolist() == [0, 0]

def test_mp_doc_2():
  a_emissions = [0.5, 0.5]
  b_emissions = [0.5, 0.5]
  emissions = np.array([a_emissions,
                        b_emissions])
  compat_with_a = [[[0.0, 0.0], [0.0, 0.0]],
                   [[0.0, 1.0], [0.0, 1.0]]]
  compat_with_b = [[[0.0, 1.0], [0.0, 1.0]],
                   [[0.0, 0.0], [0.0, 0.0]]]
  compatibilities = np.array([compat_with_a,
                              compat_with_b])
  assert mp_doc(emissions, compatibilities, 0.5).tolist() == [1, 1]

def test_mp_doc_3():
  a_emissions = [1.0, 0.0]
  b_emissions = [0.4, 0.6]
  emissions = np.array([a_emissions,
                        b_emissions])
  compat_with_a = [[[0.0, 0.0], [0.0, 0.0]],
                   [[0.0, 1.0], [0.0, 1.0]]]
  compat_with_b = [[[0.0, 1.0], [0.0, 1.0]],
                   [[0.0, 0.0], [0.0, 0.0]]]
  compatibilities = np.array([compat_with_a,
                              compat_with_b])
  assert mp_doc(emissions, compatibilities, 0.5).tolist() == [0, 1]
