import numpy as np
from numpy.linalg import inv
from sklearn.cross_validation import StratifiedKFold
from ksvd import ApproximateKSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import SparseCoder
import sys
import scipy

ddot = scipy.linalg.blas.ddot
def fast_dot(a,b):
    """a fast implementation of matrix-matrix,
       matrix-vector,vector-vector products
    """
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 2 and b_dim == 2:
        return np.dot(a,np.transpose(b))
    elif a_dim == 2 and b_dim == 1 or a_dim == 1 and b_dim == 2:
        # matrix to vector product
        # GEMV is slower than np.dot-why?
        return np.dot(a,b)
    elif a_dim == 1 and b_dim == 1:
        return ddot(a,b)


def ksvd(Y, D, X, n_cycles=1, verbose=True):
    n_atoms = D.shape[1]
    n_features, n_samples = Y.shape
    unused_atoms = []
    R = Y - fast_dot(D, X)

    for c in range(n_cycles):
        for k in range(n_atoms):
            if verbose:
                sys.stdout.write("\r" + "k-svd..." + ":%3.2f%%" % ((k / float(n_atoms)) * 100))
                sys.stdout.flush()
            # find all the datapoints that use the kth atom
            omega_k = X[k, :] != 0
            if not np.any(omega_k):
                unused_atoms.append(k)
                continue
            # the residual due to all the other atoms but k
            Rk = R[:, omega_k] + np.outer(D[:, k], X[k, omega_k])
            U, S, V = randomized_svd(np.nan_to_num(Rk), n_components=1, n_iter=10, flip_sign=False)
            D[:, k] = U[:, 0]
            X[k, omega_k] = V[0, :] * S[0]
            # update the residual
            R[:, omega_k] = Rk - np.outer(D[:, k], X[k, omega_k])
        print("")
    return D, X, unused_atoms

def class_accuracy(y_pred, y_test):
    """the classification accuracy"""
    n_correct = np.sum(y_test == y_pred)
    return n_correct / float(y_test.size)

def class_dict_coder(X, y, n_class_atoms=None, n_class=None):
    # For each class do ksvd using library
    D = np.zeros((X.shape[0], n_class_atoms*4))

    for i in range(n_class):
        print("Doing dictionary initialization for ", i)
        X1 = X[:, y==i]
        aksvd = ApproximateKSVD(n_components=n_class_atoms)
        dictionary = aksvd.fit(np.transpose(X1)).components_
        D[:, i*n_class_atoms:(i+1)*n_class_atoms] = np.transpose(dictionary)
        print("Done dictionary initialization for ", i)
    return D


class lc_ksvd():

    def __init__(self, n_class_atoms = 300, n_folds = 10, n_class=4, lambda1=1,
                 lambda2=1, alpha=1, beta=1, max_iter=4, n_nonzero=30):
        self.n_class = n_class
        self.D = None
        self.W = None
        self.A = None
        self.n_folds = n_folds
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.alpha = alpha
        self.beta = beta
        self.n_class_atoms = n_class_atoms
        self.max_iter = max_iter
        self.n_nonzero = n_nonzero
    
    def fit(self, X, y):
        self.__call__(X, y)

    def __call__(self, X, y):
        self.evaluate_(X, y)
    
    def evaluate_(self, X, y):
        folds = StratifiedKFold(y, n_folds=self.n_folds, shuffle=True, random_state=None)

        for train_index, test_index in folds:
            X_train, X_test = X[:, train_index], X[:, test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.train(X_train, y_train)
            print(self.D.shape)

            #y_pred = self.predict(X_test)
            #y_pred = np.array(y_pred)

            #class_acc = class_accuracy(y_pred, y_test)
            #print(class_acc)

    def train(self, X, y):
        total_atom_dict = self.n_class_atoms*4
        n_features, n_samples = X.shape


        D = class_dict_coder(X, y, self.n_class_atoms, self.n_class)
        Z = np.zeros((total_atom_dict, n_samples))

        Q = np.zeros((total_atom_dict, n_samples))
        for i in range(self.n_class):
            Q[i*self.n_class_atoms:(i+1)*self.n_class_atoms,y==i]=1

        H = np.zeros((self.n_class, X.shape[1])).astype(int)
        for i in range(X.shape[1]):
            H[y[i].astype(int), i] = 1
        
        I = np.eye(total_atom_dict)

        W = np.dot(inv(np.dot(Z, Z.T) + self.lambda1 * I), np.dot(Z, H.T)).T
        G = np.dot(inv(np.dot(Z, Z.T) + self.lambda2 * I), np.dot(Z, Q.T)).T

        print(Z.shape, Q.shape, H.shape, W.shape, G.shape)

        _X = np.vstack((X, np.sqrt(self.alpha) * Q))
        _X = np.vstack((_X, np.sqrt(self.beta) * H))

        D = D / np.linalg.norm(D)
        G = G / np.linalg.norm(G)
        W = W / np.linalg.norm(W)

        _D = np.vstack((D, np.sqrt(self.alpha) * G))
        _D = np.vstack((_D, np.sqrt(self.beta) * W))

        _D = _D / np.linalg.norm(_D)

        for it in range(self.max_iter):

            D = np.nan_to_num(D)
            X = np.nan_to_num(X)

            coder = SparseCoder(dictionary=np.transpose(D), transform_n_nonzero_coefs=self.n_nonzero,
                                 transform_algorithm='omp')
            Z = coder.transform(np.transpose(X))

            Z = np.nan_to_num(Z)
            _D, _, unused_atoms = ksvd(_X, _D, Z, verbose=True)

            D = _D[:n_features, :]
            G = _D[n_features:n_features + total_atom_dict, :]
            W = _D[n_features + total_atom_dict:, :]

            D = D / np.linalg.norm(D)
            G = G / np.linalg.norm(G)
            W = W / np.linalg.norm(W)

            _D = np.vstack((D, np.sqrt(self.alpha) * G))
            _D = np.vstack((_D, np.sqrt(self.beta) * W))

            _D = _D / np.linalg.norm(_D)

        self.D, Z, self.W = D, Z, W