import numpy as np
import cvxopt
from utils import create_dataset, plot_contour


def linear(x, z):
    return np.dot(x, z.T)

def polynomial(x, z, p = 6):
    return (1 + np.dot(x, z.T)) ** p


def gaussian(x, z, sigma=0.1):
    return np.exp(-np.linalg.norm(x - z, axis=1) ** 2 / (2 * (sigma ** 2)))


class support_vectorsM:
    def __init__(self, kernel=gaussian, C=1):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        self.y = y
        self.X = X
        m, _ = X.shape

        # Calculate Kernel
        self.K = np.zeros((m, m))
        for i in range(m):
            self.K[i, :] = self.kernel(X[i, np.newaxis], self.X)

        P = np.outer(y, y) * self.K
        q = -np.ones((m, 1))
        G = np.vstack((np.eye(m) * -1, np.eye(m)))
        h = np.hstack((np.zeros(m), np.ones(m) * self.C))
        A = None
        b = np.zeros(1)
        cvxopt.solvers.options["show_progress"] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(sol["x"])

    def predict(self, X):
        y_predict = np.zeros((X.shape[0]))
        support_vectors = self.get_parameters(self.alphas)

        for i in range(X.shape[0]):
            y_predict[i] = np.sum(
                self.alphas[support_vectors]
                * self.y[support_vectors, np.newaxis]
                * self.kernel(X[i], self.X[support_vectors])[:, np.newaxis]
            )

        return np.sign(y_predict + self.b)

    def get_parameters(self, alphas):
        threshold = 1e-5

        support_vectors = ((alphas > threshold) * (alphas < self.C)).flatten()
        self.w = np.dot(self.X[support_vectors].T, alphas[support_vectors] * self.y[support_vectors, np.newaxis])
        self.b = np.mean(self.y[support_vectors, np.newaxis]
            - self.alphas[support_vectors] * self.y[support_vectors, np.newaxis] * self.K[support_vectors, support_vectors][:, np.newaxis]
        )
        return 