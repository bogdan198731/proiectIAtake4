import random
from neuron import *


#functie main, citirea fisierelor CSV, crearea unui obiect de tip retea si testarea rezultatelor

def main():

    m, X_train, Y_train, X_dev, Y_dev = getData()

# creare obiect tip retea
    n = retea(m)

# ajustare greutati si bias-uri
    n.gradient_descent(X_train, Y_train, 0.15, 500)

    verifData(n, X_dev, Y_dev)


def getData():
    train_data = pd.read_csv('C:/Users/bogda/PycharmProjects/date/mnist_train.csv')
    test_data = pd.read_csv('C:/Users/bogda/PycharmProjects/date/mnist_test.csv')

    train_data = np.array(train_data)
    dev_data = np.array(test_data)
    m, n = train_data.shape

    dev_data_T = dev_data.T
    Y_dev = dev_data_T[0]
    X_dev = dev_data_T[1:n]
    X_dev = X_dev / 255.

    train_data_T = train_data.T
    Y_train = train_data_T[0]

    X_train = train_data_T[1:n]
    X_train = X_train / 255.
    _,m_train = X_train.shape
    return m, X_train, Y_train, X_dev, Y_dev


def verifData(n, X_dev, Y_dev):
    # verificare date obtinute cu ajutorul datelor de test

    n.verif_pred_test(500, X_dev, Y_dev)
    t, _ = X_dev.shape
    j = random.randint(0, t)
    n.test_prediction(j, X_dev, Y_dev)
    j = random.randint(0, t)
    n.test_prediction(j, X_dev, Y_dev)
    j = random.randint(0, t)
    n.test_prediction(j, X_dev, Y_dev)
    j = random.randint(0, t)
    n.test_prediction(j, X_dev, Y_dev)

main()