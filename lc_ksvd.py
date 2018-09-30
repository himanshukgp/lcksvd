import numpy as np
from numpy.linalg import inv
from sklearn.cross_validation import StratifiedKFold
from ksvd import ApproximateKSVD


def class_accuracy(y_pred, y_test):
    """the classification accuracy"""
    n_correct = np.sum(y_test == y_pred)
    return n_correct / float(y_test.size)

def class_dict_coder(X, y, n_class_atoms=None, n_class=None):
    # For each class do ksvd using library
    D = np.zeros((X.shape[0], n_class_atoms*4))

    for i in range(n_class):
        X1 = X[:, y==i]
        aksvd = ApproximateKSVD(n_components=n_class_atoms)
        dictionary = aksvd.fit(X1).components_
        D[:, i*n_class_atoms:(i+1)*n_class_atoms] = dictionary
    return D


class lc_ksvd():

    def __init__(self, n_class_atoms = 600, n_folds = 10, n_class=4, lambda1=1, lambda2=1):
        self.n_class = n_class
        self.D = None
        self.W = None
        self.A = None
        self.n_folds = n_folds
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.n_class_atoms = n_class_atoms
    
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

            #y_pred = self.predict(X_test)
            #y_pred = np.array(y_pred)

            class_acc = class_accuracy(y_pred, y_test)
            print(class_acc)

    def train(self, X, y):
        total_atom_dict = self.n_class_atoms*4
        n_samples = X.shape[1]

        D = class_dict_coder(X, y, self.n_class_atoms, self.n_class)
        Z = np.zeros((total_atom_dict, n_samples))

        Q = np.zeros((total_atom_dict, n_samples))
        for i in range(self.n_class):
            Q[i*self.n_class_atoms:(i+1)*self.n_class_atoms,y==i]=1

        H = np.zeros((self.n_class, X.shape[1])).astype(int)
        for i in range(X.shape[1]):
            H[y[i], i] = 1
        
        I = np.eye(total_atom_dict)

        W = np.dot(inv(np.dot(Z, Z.T) + self.lambda1 * I), np.dot(Z, H.T)).T
        G = np.dot(inv(np.dot(Z, Z.T) + self.lambda2 * I), np.dot(Z, Q.T)).T

        print(D.shape, Z.shape)
        

