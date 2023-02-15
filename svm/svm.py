import numpy as np
import cvxopt
from utils import *

class SVM:
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

        # Solve with cvxopt final QP needs to be reformulated
        # to match the input form for cvxopt.solvers.qp
        P = cvxopt.matrix(np.outer(y, y) * self.K)
        q = cvxopt.matrix(-np.ones((m, 1)))
        G = cvxopt.matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        A = cvxopt.matrix(y, (1, m), "d")
        b = cvxopt.matrix(np.zeros(1))
        cvxopt.solvers.options["show_progress"] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(sol["x"])

    def get_parameters(self, alphas):
        """
        Computes W, b parameter using support vectors 
        and alphas for optimization. 
        """
        threshold = 1e-5

        support_vectors = ((alphas > threshold) * (alphas < self.C)).flatten()

        self.w = np.dot(
            X[support_vectors].T, alphas[support_vectors] 
            * self.y[support_vectors, np.newaxis]
        )

        self.b = np.mean(
            self.y[support_vectors, np.newaxis]
            - self.alphas[support_vectors] * self.y[support_vectors, np.newaxis] 
            * self.K[support_vectors, support_vectors][:, np.newaxis]
        )
        return support_vectors

    def predict(self, X):
        y_predict = np.zeros((X.shape[0]))
        sv = self.get_parameters(self.alphas)

        for i in range(X.shape[0]):
            y_predict[i] = np.sum(
                self.alphas[sv]
                * self.y[sv, np.newaxis]
                * self.kernel(X[i], self.X[sv])[:, np.newaxis]
            )

        return np.sign(y_predict + self.b)

if __name__ == "__main__":
    np.random.seed(1)
    X, y = create_dataset(N = 100)
    kernel = [linear, polynomial, gaussian]

    for k in kernel:
        svm = SVM(kernel=k)
        svm.fit(X, y)
        y_pred = svm.predict(X)
        plot_contour(X, y, svm, kernel = k.__name__)

        print(f"Accuracy using {k.__name__}: {sum(y == y_pred) / y.shape[0]}")
        