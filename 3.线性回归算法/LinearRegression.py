import numpy as np

class LinearRegression:

    '''
    输入时记得将b加进去
    '''
    def train(self,X,y):
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
    def predict(self,X):
        return X.T.dot(self.w)
    
    def mean_squared_error(y_true,y_pred):
            return np.average((y_true-y_pred)**2,axis=0)
    
    def R2_score(y_true,y_pred):
        mMES = (y_true-y_pred)**2
        average_error = (y_true-np.average(y_true,axis=0))**2
        return 1 - mMES.sum(axis = 0)/average_error.sum(axis = 0)