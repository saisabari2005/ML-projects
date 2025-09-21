import numpy as np

class LDA:

    def __init__(self,n_compnents):
        self.n_components=n_compnents
        self.linear_discriminants=None
    
    def fit(self,X,y):
        n_features=X.shape[1]
        class_labels= np.unique(y)


        #S_W,S_B

        mean_overall = np.mean(X,axis=0)
        S_w=np.zeros((n_features,n_features))
        S_b= np.zeros((n_features,n_features))
        for c in class_labels:
            X_c = X[y==c]
            mean_c=np.mean(X_c,axis=0)
            S_w+=(X_c-mean_c).T.dot(X_c-mean_c)    

            n_c = X.shape[0]
            mean_diff = (mean_c-mean_overall).reshape(n_features,1)
            S_B+=(mean_diff).T.dot(mean_diff)

        A=np.linalg.inv(S_W).dot(S_b)
        eigenvalues,eigenvectors=np.linalg.eig(A)
        eigenvectors=eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues=eigenvalues[idxs]
        eigenvectors=eigenvectors[idxs]
        self.linear_discriminants=eigenvectors[0:self.n_components]

    def transform(self,X):
        return np.dot(X,self.linear_discriminants.T)
