
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class retea:

    def retea(self, m):
        self.m = m


    def init_params(self):
        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
        return W1, b1, W2, b2


    def ReLU(self, Z):
        return np.maximum(Z, 0)


    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A


    def forward_prop(self, W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = self.ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2


    def ReLU_deriv(self, Z):
        return Z > 0


    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y


    def backward_prop(self, Z1, A1, Z2, A2, W1, W2, X, Y):
        one_hot_Y = self.one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / self.m * dZ2.dot(A1.T)
        db2 = 1 / self.m * np.sum(dZ2)

        dZ1 = W2.T.dot(dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / self.m * dZ1.dot(X.T)
        db1 = 1 / self.m * np.sum(dZ1)

        return dW1, db1, dW2, db2


    def update_params(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        return W1, b1, W2, b2



    def get_predictions(self, A2):
        return np.argmax(A2, 0)

    def get_accuracy(self, predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size




    def gradient_descent(self, X, Y, alpha, iterations):
        W1, b1, W2, b2 = self.init_params()
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_prop(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
            W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            if i % 100 == 0:
                print("Iteration: ", i)
                predictions = self.get_predictions(A2)
                print("Accuracy = " + str(self.get_accuracy(predictions, Y)))
    #            alpha = alpha - (get_accuracy(predictions, Y) / 500)
    #            print('alpha = ' + str(alpha))
        return W1, b1, W2, b2

    def make_predictions(self, X, W1, b1, W2, b2):
        _, _, _, A2 = self.forward_prop(W1, b1, W2, b2, X)
        predictions = self.get_predictions(A2)
        return predictions

    def test_prediction(self, index, W1, b1, W2, b2, X_dev, Y_dev):
        print("test prediction")
        current_image = X_dev[:, index, None]
        prediction = self.make_predictions(X_dev[:, index, None], W1, b1, W2, b2)
        label = Y_dev[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

    def verif_pred_test(self, W1, b1, W2, b2, index, X_dev, Y_dev):
        k = 0
        o = 0
        for i in range(index):
            _, _, _, A2 = self.forward_prop(W1, b1, W2, b2, X_dev[:, i, None])
            test = self.get_predictions(A2)
            if test == Y_dev[i]:
                k = k + 1
            elif o < 3:
                self.test_prediction(i, W1, b1, W2, b2, X_dev, Y_dev)
                o = o + 1
        print("Acuratete teste dev = " + str(k / index))


