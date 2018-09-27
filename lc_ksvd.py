import numpy as np
from sklearn.cross_validation import StratifiedKFold




def class_accuracy(y_pred, y_test):
    """the classification accuracy"""
    n_correct = np.sum(y_test == y_pred)
    return n_correct / float(y_test.size)


def avg_class_accuracy(y_pred, y_test):
    """the classification accuracy averaged over the classes"""
    n_classes = len(set(y_test))
    class_accs = []

    for c in range(n_classes):
        n_correct = np.sum(y_test[y_test == c] == y_pred[y_test == c])
        print ("number of correct = ", n_correct)
        n_class_samples = y_test[y_test == c].size
        class_accs.append(n_correct / float(n_class_samples))
    return np.mean(class_accs)



def lc_ksvd():

    def fit(self, X, y):
        self.__call__(X, y)


    def __call__(self, X, y):
        """
        Simply do a kfold and call the next function.
        """
        n_classes = len(set(y))
        self.folds = StratifiedKFold(y, n_folds=self.n_folds, shuffle=False, random_state=None)
        score = self.evaluate(X, y)

    
    def evaluate(self, X, y):
        """
        evaluate the performance of the classifier
        trained with the parameters in <param_set>
        """
        cv_scores = []
        # avg_class_accs = []
        t=0
        for train_index, test_index in self.folds:
            # Print fold number before executing each fold
            print("fold number = ",t)
            t+=1

            X_train, X_test = X[:, train_index], X[:, test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("all input shapes", X_train.shape, X_test.shape, y_train.shape )

            # Train the given dataset
            self.train(X_train, y_train)
            
            y_pred = self.predict(X_test)
            y_pred = np.array(y_pred)
            print("y is predicted", y_pred.shape)

            class_acc = class_accuracy(y_pred, y_test)
            # avg_class_acc  = avg_class_accuracy(y_pred,y_test)
            cv_scores.append(class_acc)
            # avg_class_accs.append(avg_class_acc)
            print("average class accuracy:", avg_class_accuracy(y_pred, y_test))

        avg_cv_score = np.mean(cv_scores)
        print ("accuracy:", avg_cv_score)
        return avg_cv_score