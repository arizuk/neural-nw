import numpy as np
# import gzip
# import matplotlib.pyplot as plt
# from PIL import Image

# loader
img_size = 784
file_map = {
  'train_image': 'train-images-idx3-ubyte.gz',
  'train_label': 'train-labels-idx1-ubyte.gz',
  'test_image': 't10k-images-idx3-ubyte.gz',
  'test_label': 't10k-labels-idx1-ubyte.gz'
}

def load_mnist(key):
  images = _load_mnist_image(file_map[ "%s_image" % (key) ])
  labels = _load_mnist_label(file_map[ "%s_label" % (key) ])
  return (images, labels)

def _load_mnist_image(filename):
  file_path = './data/' + filename
  with gzip.open(file_path, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
  return data.reshape(-1, img_size)

def _load_mnist_label(filename):
  file_path = './data/' + filename
  with gzip.open(file_path, 'rb') as f:
    labels = np.frombuffer(f.read(), np.uint8, offset=8)
  return labels

class Fn:
  @staticmethod
  def sigmoid(x):
    return 1 / (1 + np.exp(-x))

  @staticmethod
  def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

  @staticmethod
  def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

  @staticmethod
  def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
      tmp = x[idx]

      x[idx] = tmp + h
      fxh1 = f(x)

      x[idx] = tmp - h
      fxh2 = f(x)

      grad[idx] = (fxh1 - fxh2) / (2*h)
      x[idx] = tmp

    return grad

class Network:
  def __init__(self):
    self.layers = []

  def add_layer(self, layer):
    self.layers.append(layer)

  def loss(self, x, t):
    y = self.predict(x, t)
    return Fn.cross_entropy_error(y, t)

  # def train():
  #   for layer in self.layers:
  #     print('Todo')

  def predict(self, x):
    for layer in self.layers:
      x = layer.forward(x)
    return x

class Layer:
  def __init__(self, input_dim, output_dim, activation_fn=Fn.sigmoid):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.weight = 0.01 + np.random.randn(input_dim, output_dim)
    self.bias = np.zeros_like(output_dim)
    self.activation_fn = activation_fn

  def forward(self, x):
    y = np.dot(x, self.weight) + self.bias
    return self.activation_fn(y)

if __name__ == "__main__":
  network = Network()
  network.add_layer(Layer(10, 2))
  network.add_layer(Layer(2, 1, activation_fn=Fn.softmax))

  x = np.random.randn(2, 10)
  print(network.predict(x))

  # data = load_mnist('train')
  # for i in range(1, 10):
  #   img1 = data[i].reshape(28, 28)
  #   pil_img = Image.fromarray(np.uint8(img1))
  #   pil_img.show()
