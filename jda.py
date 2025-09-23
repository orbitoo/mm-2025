import numpy as np
import scipy.linalg
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist


def kernel(ker, X1, X2, gamma):
    K = None
    if ker == "linear":
        if X2 is not None:
            K = np.dot(X1, X2.T)
        else:
            K = np.dot(X1, X1.T)
    elif ker == "rbf":
        sq_dists = cdist(X1, X2, "sqeuclidean")
        K = np.exp(-gamma * sq_dists)
    return K


class JDA:
    def __init__(self, k=30, lambda_=0.1, ker="linear", gamma=1.0, T=10):
        self.k = k
        self.lambda_ = lambda_
        self.ker = ker
        self.gamma = gamma
        self.T = T
        self.A = None  # Projection matrix

    def fit_predict(self, Xs, Ys, Xt):
        X = np.vstack((Xs, Xt))
        ns, nt = Xs.shape[0], Xt.shape[0]
        C = len(np.unique(Ys))

        M, Mc = 0, 0
        H = np.eye(ns + nt) - 1 / (ns + nt) * np.ones((ns + nt, ns + nt))

        Zs_new, Zt_new = None, None
        Yt_pseudo = None

        for t in range(self.T):
            # MMD Matrix M0 for marginal distribution alignment
            e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
            M0 = e.dot(e.T)

            # MMD Matrix Mc for conditional distribution alignment
            if Yt_pseudo is not None and len(Yt_pseudo) == nt:
                Mc = 0
                for c in range(C):
                    e = np.zeros((ns + nt, 1))
                    source_class_idx = np.where(Ys == c)[0]
                    target_class_idx = np.where(Yt_pseudo == c)[0]
                    ns_c = len(source_class_idx)
                    nt_c = len(target_class_idx)
                    if ns_c > 0:
                        e[source_class_idx, 0] = 1 / ns_c
                    if nt_c > 0:
                        e[ns + target_class_idx, 0] = -1 / nt_c

                    Mc += e.dot(e.T)

            M = M0 + Mc
            K = kernel(self.ker, X, X, self.gamma)

            # Solve the generalized eigenvalue problem
            # A = (K M K^T + lambda*I)^{-1} K H K^T B
            # --> (K M K^T + lambda*I) A = K H K^T A \Lambda
            term_a = np.dot(np.dot(K, M), K.T) + self.lambda_ * np.eye(ns + nt)  # type: ignore
            term_b = np.dot(np.dot(K, H), K.T)  # type: ignore
            term_b += 1e-6 * np.eye(ns + nt)  # for numerical stability

            eig_val, eig_vec = scipy.linalg.eigh(term_a, term_b)
            self.A = eig_vec[
                :, : self.k
            ]  # smallest k eigenvalues' corresponding eigenvectors

            # projected data
            Z = np.dot(K, self.A)  # type: ignore
            Zs_new, Zt_new = Z[:ns, :], Z[ns:, :]

            # Generate pseudo labels for target domain using KNN
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Zs_new, Ys.ravel())
            Yt_pseudo = clf.predict(Zt_new)

            print(f"JDA | Iteration {t + 1} / {self.T} completed.")

        return Zs_new, Zt_new
