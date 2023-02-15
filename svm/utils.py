import numpy as np
import matplotlib.pyplot as plt

def linear(x, z):
    return np.dot(x, z.T)

def polynomial(x, z, p = 5):
    return (1 + np.dot(x, z.T)) ** p


def gaussian(x, z, sigma = 0.1):
    return np.exp(- np.linalg.norm(x - z, axis = 1) ** 2 / (2 * (sigma ** 2)))

def create_dataset(N, D = 2, K = 2):
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K)  # class labels

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.3  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    # lets visualize the data:
    plt.scatter(X[:, 0], X[:, 1], c = y, s= 40, cmap=plt.cm.Spectral)
    plt.savefig('assets/input_data.png')
    plt.show()

    y[y == 0] -= 1

    return X, y

def plot_contour(X, y, svm, kernel):
    # plot the resulting classifier
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    points = np.c_[xx.ravel(), yy.ravel()]

    Z = svm.predict(points)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # plt the points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.savefig(f'assets/output_data_{kernel}.png')
    plt.show()
