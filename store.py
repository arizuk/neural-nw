import numpy as np

store_path = './network.pickle.npy'
def save_params(network):
  params = [[l.weight, l.bias] for l in network.layers]
  np.save(store_path, params)

def restore_params():
  return np.load(store_path)
