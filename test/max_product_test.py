import numpy as np

from max_product import mp_tree_depth_1

def test_mp_tree_depth_1_1():
  emissions = np.array([[1.0, 0.0],
                        [1.0, 0.0]])
  compatibilities = np.array([[[1.0, 0.0], [1.0, 0.0]]])
  calc_emissions = [lambda i=i: emissions[i] for i in range(len(emissions))]
  calc_compatibilities = [lambda i=i: compatibilities[i] for i in range(len(emissions))]
  assert mp_tree_depth_1(calc_emissions, calc_compatibilities) == 0

def test_mp_tree_depth_1_2():
  emissions = np.array([[0.5, 0.5],
                        [0.5, 0.5]])
  compatibilities = np.array([[[1.0, 0.0], [1.0, 0.0]]])
  calc_emissions = [lambda i=i: emissions[i] for i in range(len(emissions))]
  calc_compatibilities = [lambda i=i: compatibilities[i] for i in range(len(emissions))]
  assert mp_tree_depth_1(calc_emissions, calc_compatibilities) == 0

def test_mp_tree_depth_1_3():
  emissions = np.array([[0.5, 0.5],
                        [0.5, 0.5]])
  compatibilities = np.array([[[0.0, 1.0], [0.9, 0.1]]])
  calc_emissions = [lambda i=i: emissions[i] for i in range(len(emissions))]
  calc_compatibilities = [lambda i=i: compatibilities[i] for i in range(len(emissions))]
  assert mp_tree_depth_1(calc_emissions, calc_compatibilities) == 1

def test_mp_tree_depth_1_4():
  emissions = np.array([[0.3, 0.7],
                        [0.2, 0.8],
                        [1.0, 0.0]])
  compatibilities = np.array([[[0.9, 0.1], [0.6, 0.4]],
                              [[0.99, 0.01], [0.0, 0.0]]])
  calc_emissions = [lambda i=i: emissions[i] for i in range(len(emissions))]
  calc_compatibilities = [lambda i=i: compatibilities[i] for i in range(len(emissions))]
  assert mp_tree_depth_1(calc_emissions, calc_compatibilities) == 0
