
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


class classifiers():
    
    def __init__(self, data, idx_train, idx_test, idx_y, idx_a): 
        self.y_data = data[:, idx_y]
        self.X_data = np.delete(data, [idx_y, idx_a], axis=1)
        self.X_train = self.X_data[idx_train]
        self.y_train = self.y_data[idx_train]
        self.X_test = self.X_data[idx_test]
        self.y_test = self.y_data[idx_test]
    
    def train_logistic_regression(self):
        clf = LogisticRegression(random_state=100,max_iter=500).fit(self.X_train, self.y_train)
        clf_predictions_train = clf.predict(self.X_train)
        clf_predictions_test = clf.predict(self.X_test)
        clf_probs_all = clf.predict_proba(self.X_data)[:,1]
        
        p_correct_train = 1 - (abs(clf_predictions_train-self.y_train).sum() / len(self.y_train))
        p_correct_test = 1 - (abs(clf_predictions_test-self.y_test).sum() / len(self.y_test))
        
        print('correct predictions train set' ,'\t', p_correct_train*100)
        print('correct predictions test set' , '\t',  p_correct_test*100)
        return clf, clf_probs_all