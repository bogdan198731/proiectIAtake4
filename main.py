import random

from neuron import *


train_data = pd.read_csv('C:/Users/bogda/PycharmProjects/date/mnist_train.csv')
test_data = pd.read_csv('C:/Users/bogda/PycharmProjects/date/mnist_test.csv')

train_data = np.array(train_data)
dev_data = np.array(test_data)
m, n = train_data.shape

t, l = dev_data.shape

dev_data_T = dev_data.T
Y_dev = dev_data_T[0]
X_dev = dev_data_T[1:n]
X_dev = X_dev / 255.

train_data_T = train_data.T
Y_train = train_data_T[0]

X_train = train_data_T[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


n =  retea()
n.m = m


W1, b1, W2, b2 = n.gradient_descent(X_train, Y_train, 0.15, 500)

n.verif_pred_test(W1, b1, W2, b2, 1000, X_dev, Y_dev)

j = random.randint(0, t)
n.test_prediction(j, W1, b1, W2, b2, X_dev, Y_dev)
j = random.randint(0, t)
n.test_prediction(j, W1, b1, W2, b2, X_dev, Y_dev)
j = random.randint(0, t)
n.test_prediction(j, W1, b1, W2, b2, X_dev, Y_dev)
j = random.randint(0, t)
n.test_prediction(j, W1, b1, W2, b2, X_dev, Y_dev)


