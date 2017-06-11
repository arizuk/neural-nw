import numpy as np
import gzip
import matplotlib.pyplot as plt
import store

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
  images = images.astype(np.float32)/255.0 # normalize
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
  return _convert_labels(labels)

# 2 => [0, 0, 1, 0, 0, 0..]
def _convert_labels(x):
  t = np.zeros((x.size, 10))
  for idx, row in enumerate(t):
      row[x[idx]] = 1
  return t

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
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

  @staticmethod
  def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
      idx = it.multi_index
      tmp = x[idx]

      x[idx] = tmp + h
      fxh1 = f(x)

      x[idx] = tmp - h
      fxh2 = f(x)

      grad[idx] = (fxh1 - fxh2) / (2*h)
      x[idx] = tmp
      it.iternext()

    return grad

class Network:
  def __init__(self):
    self.layers = []
    self.loss_fn = Fn.cross_entropy_error

  def add_layer(self, layer):
    self.layers.append(layer)

  def loss(self, x, t):
    y = self.predict(x)
    return self.loss_fn(y, t)

  def predict(self, x):
    for layer in self.layers:
      x = layer.forward(x)
    return x

  def gradient(self, x, t):
    loss = lambda w: self.loss(x, t)
    grads = []
    for layer in self.layers:
      w_grads = Fn.numerical_gradient(loss, layer.weight)
      b_grads = Fn.numerical_gradient(loss, layer.bias)
      grads.append({ 'weight': w_grads, 'bias': b_grads })
    return grads

class Layer:
  def __init__(self, input_dim, output_dim, activation_fn=Fn.sigmoid, weight=None, bias=None):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.activation_fn = activation_fn

    if weight is None:
      self.weight = 0.01 + np.random.randn(input_dim, output_dim)
    else:
      self.weight = weight

    if bias is None:
      self.bias = np.zeros(output_dim)
    else:
      self.bias = bias

  def forward(self, x):
    y = np.dot(x, self.weight) + self.bias
    return self.activation_fn(y)

class AccuracyCalculator:
  def __init__(self, x, t):
    self.x = x
    self.t = t

  def accuracy(self, network):
    y = network.predict(self.x)
    y = np.argmax(y, axis=1)
    t = np.argmax(self.t, axis=1)
    accuracy = np.sum(y == t) / float(self.x.shape[0])
    return accuracy

class Trainer:
  def __init__(self, network, train_x, train_t, learning_rate=0.1, batch_size=100, iteration=10000):
    self.network = network
    self.train_x = train_x
    self.train_t = train_t
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.iteration = iteration
    self.listeners = []

  def add_listener(self, listener):
    self.listeners.append(listener)

  def train(self):
    for i in range(self.iteration):
      batch_mask = np.random.choice(self.train_x.shape[0], self.batch_size)
      x = self.train_x[batch_mask]
      t = self.train_t[batch_mask]

      self.update_network(x, t)

      message = {
        'iter': i,
        'x': x,
        't': t,
        'network': network,
      }
      for listener in self.listeners:
        listener(message)

  def update_network(self, x, t):
    network = self.network
    grads = network.gradient(x, t)
    for i in range(len(network.layers)):
      layer = network.layers[i]
      layer.weight -= grads[i]['weight'] * self.learning_rate
      layer.bias  -= grads[i]['bias'] * self.learning_rate

if __name__ == "__main__":
  (train_x, train_t) = load_mnist('train')
  (test_x, test_t) = load_mnist('test')

  if True:
    network = Network()
    network.add_layer(Layer(784, 100))
    network.add_layer(Layer(100, 50))
    network.add_layer(Layer(50, 10, activation_fn=Fn.softmax))

    trainer = Trainer(network, train_x, train_t, iteration=10000)
    test_acc = AccuracyCalculator(test_x, test_t)

    def test_acc_logger(msg):
      i, network = msg['iter'], msg['network']
      if i % 30 == 0:
        print("iter=%s, accuracy=%s %%" % (i, test_acc.accuracy(network) * 100))

    def loss_logger(msg):
      i, network, x, t = msg['iter'], msg['network'], msg['x'], msg['t']
      if i % 1 == 0:
        loss = network.loss(x, t)
        print("iter=%s, loss=%s %%" % (i, network.loss(x, t)))

    def save_point(msg):
      i, network = msg['iter'], msg['network']
      if i % 5 == 0:
        store.save_params(network)

    trainer.add_listener(test_acc_logger)
    trainer.add_listener(save_point)
    trainer.add_listener(loss_logger)
    trainer.train()
    store.save_params(network)
    print("accuracy=%s %%" % (test_acc.accuracy(network) * 100))

  else:
    # params = store.restore_params()
    network = Network()
    # network.add_layer(Layer(784, 10, weight=params[0][0], bias=params[0][1]))
    # network.add_layer(Layer(10, 10, activation_fn=Fn.softmax, weight=params[1][0], bias=params[1][1]))

    # test_acc = AccuracyCalculator(test_x, test_t)
    # print("accuracy=%s %%" % (test_acc.accuracy(network) * 100))

  # x = np.arange(1, len(loss_history)+1)
  # plt.plot(x, loss_history, label='loss')
  # plt.xlabel('iteration')
  # plt.ylabel('loss')
  # plt.show()
