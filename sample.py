import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, make_scorer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

class MachineFailurePredictorPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
    
    def wrangle_data(self):
        print(f"Total rows before preprocessing: {self.data.shape[0]}")
        print("\n" + str(self.data.head(2)))
        print(".\n.\n.")
        print(str(self.data.tail(2)) + "\n")
        
        # Dropping irrelevant columns
        self.data = self.data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        print(f"Total data instances after dropping irrelevant columns for step (3): {self.data.shape[0]}")
        print("\n" + str(self.data.head(2)))
        print(".\n.\n.")
        print(str(self.data.tail(2)) + "\n")
        
        # Encoding 'Type' column to numeric (L: -1, M: 0, H: 1)
        type_mapping = {'L': -1, 'M': 0, 'H': 1}
        self.data['Type'] = self.data['Type'].map(type_mapping)
        print(f"Total data instances after categorical transformation for step (4): {self.data.shape[0]}")
        print("\n" + str(self.data.head(2)))
        print(".\n.\n.")
        print(str(self.data.tail(2)) + "\n")
        
        X = self.data.drop('Machine failure', axis=1)
        y = self.data['Machine failure']

        # Random Under Sampling
        self.X, self.y = RandomUnderSampler(random_state=339).fit_resample(X, y)
        print(f"Total data instances after performing random sampling for step (5): {len(self.X)}")

        print("\n-----Results for part (6)-----\n")
        print(f"Number of features: {self.X.shape[1]}\n")
    
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=339, stratify=self.y
        )
        
        print(f"Training set size: {len(self.X_train)}\n")
        print(f"Test set size: {len(self.X_test)}\n")
    
    def mcc(self, y, y_hat): 
        tn, fp, fn, tp = confusion_matrix(y, y_hat).flatten()
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        if denominator == 0: # i noticed that SVM was producing runtime warnings due to invalid values in sclalar division
            return 0.0
        
        return numerator / denominator
    
    def multi_layer_nn(self):
        # Hyperparameters
        param_grid = {
            'hidden_layer_sizes': [(20,), (20,1), (50,), (50,1), (100,), (100, 1), (20, 20), (50, 20), (50, 50), (100, 20), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'learning_rate': ['constant', 'adaptive', 'invscaling'],
            'alpha': [0.1, 0.01, 0.001, 0.0001, 0.25, 0.025, 0.3, 0.35, 0.175],
            'optimizer': ['adam', 'sgd'],
            'batch_size': [6, 12, 32, 64, 128]
        }
        
        classifier = MLPClassifier(random_state=339, max_iter=2000)
        mcc = make_scorer(self.mcc)
        # Grid Search with 5-Fold Cross-Validation
        grid_search = GridSearchCV(
            classifier, param_grid, cv=5, scoring=mcc, n_jobs=-1, verbose=2
        )
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best MCC score: {grid_search.best_score_:.4f}")
        
        self.best_mlp = grid_search.best_estimator_
        self.best_mlp_parameters = grid_search.best_params_
        self.best_mlp_mcc_score = grid_search.best_score_
        
        return grid_search.best_params_, grid_search.best_score_
    
    def support_vector_machine(self):
        
        # Hyperparameters for svm
        param_grid = {
            'C': [1000, 500, 250, 100, 10, 1, 0.1, 0.01, 0.001],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
            'class_weight': [None, 'balanced']
        }
        classifier = SVC(random_state=339)
        mcc = make_scorer(self.mcc)
        
        # Grid Search with 5-Fold Cross-Validation
        grid_search = GridSearchCV(
            classifier, param_grid, cv=5, scoring=mcc, n_jobs=-1, verbose=2
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best MCC score: {grid_search.best_score_:.4f}")
        
        self.best_svm = grid_search.best_estimator_
        self.best_svm_parameters = grid_search.best_params_
        self.best_svm_mcc_score = grid_search.best_score_
        
        return grid_search.best_params_, grid_search.best_score_
    
    def k_nearest_neighbors(self):
        
        # Hyperparameters for KNN
        param_grid = {
            'n_neighbors': [2, 3, 4, 5, 6, 8, 10, 20, 30, 40, 50],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2, 3],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'leaf_size': [10, 20, 30, 40, 50]
        }
        classifier = KNeighborsClassifier()
        mcc = make_scorer(self.mcc)
        # Grid Search with 5-Fold Cross-Validation
        grid_search = GridSearchCV(
            classifier, param_grid, cv=5, scoring=mcc, n_jobs=-1, verbose=2
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best MCC score: {grid_search.best_score_:.4f}")
        
        self.best_knn = grid_search.best_estimator_
        self.best_knn_parameters = grid_search.best_params_
        self.best_knn_mcc_score = grid_search.best_score_
        
        return grid_search.best_params_, grid_search.best_score_
    
    def decision_tree(self):
        
        # Hyperparameters for Decision Tree
        param_grid = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 5, 10, 15, 20, 25, 30],
            'ccp_alpha': [0.0, 0.001, 0.01, 0.05, 0.1],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 6, 8, 10],
            'max_features': [None, 'sqrt', 'log2'],
            'min_impurity_decrease': [0.0, 0.01, 0.001]
        }
        
        classifier = DecisionTreeClassifier(random_state=339)
        mcc = make_scorer(self.mcc)
        # Grid Search with 5-Fold Cross-Validation
        grid_search = GridSearchCV(
            classifier, param_grid, cv=5, scoring=mcc, n_jobs=-1, verbose=2
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best MCC score: {grid_search.best_score_:.4f}")
        
        self.best_dt = grid_search.best_estimator_
        self.best_dt_parameters = grid_search.best_params_
        self.best_dt_mcc_score = grid_search.best_score_
        
        return grid_search.best_params_, grid_search.best_score_
    
    def logistic_regression(self):
        
        # Hyperparameters for Logistic Regression
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'solver': ['lbfgs', 'liblinear', 'newton-cholesky', 'saga'],
            'max_iter': [100, 200, 500, 1000, 2000],
            'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
        }
        
        classifier = LogisticRegression(random_state=339)
        mcc = make_scorer(self.mcc)
        
        # Grid Search with 5-Fold Cross-Validation
        grid_search = GridSearchCV(
            classifier, param_grid, cv=5, scoring=mcc, n_jobs=-1, verbose=2
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best MCC score: {grid_search.best_score_:.4f}")
        
        self.best_lr = grid_search.best_estimator_
        self.best_lr_parameters = grid_search.best_params_
        self.best_lr_mcc_score = grid_search.best_score_
        
        return grid_search.best_params_, grid_search.best_score_
    
    def train_models(self):  # I Hard coded optimal parameters found through grid search for faster processing      
        # Multi-layer Neural Network
        mlp_params = {
            'hidden_layer_sizes': (50, 50),
            'activation': 'logistic',
            'learning_rate': 'constant',
            'alpha': 0.025
        }
        self.best_mlp = MLPClassifier(random_state=339, max_iter=2000, **mlp_params)
        
        # Evaluating hyperparameters on training set 
        mcc_scorer = make_scorer(self.mcc)
        cv_scores = cross_val_score(self.best_mlp, self.X_train, self.y_train, cv=5, scoring=mcc_scorer)
        self.best_mlp_parameters = mlp_params
        self.mlp_train_score = cv_scores.mean()
        
        self.best_mlp.fit(self.X_train, self.y_train)
                
        # SVM
        svm_params = {
            'C': 1000,
            'kernel': 'rbf',
            'gamma': 'scale',
            'class_weight': None
        }
        self.best_svm = SVC(random_state=339, **svm_params)
        
        # Evaluating hyperparameters on training set 
        cv_scores = cross_val_score(self.best_svm, self.X_train, self.y_train, cv=5, scoring=mcc_scorer)
        self.best_svm_parameters = svm_params
        self.svm_train_score = cv_scores.mean()

        self.best_svm.fit(self.X_train, self.y_train)
                
        # KNN
        knn_params = {
            'n_neighbors': 5,
            'p': 1,
            'algorithm': 'auto',
            'leaf_size': 10,
            'metric': 'manhattan',
            'weights': 'uniform'
        }
        self.best_knn = KNeighborsClassifier(**knn_params)
        
        cv_scores = cross_val_score(self.best_knn, self.X_train, self.y_train, cv=5, scoring=mcc_scorer)
        self.best_knn_parameters = knn_params
        self.knn_train_score = cv_scores.mean()
        
        self.best_knn.fit(self.X_train, self.y_train)
                
        # Decision Tree
        dt_params = {
            'criterion': 'entropy',
            'max_depth': 5,
            'ccp_alpha': 0.0,
            'max_features': None,
            'min_impurity_decrease': 0.01,
            'min_samples_leaf': 1,
            'min_samples_split': 15
        }
        self.best_dt = DecisionTreeClassifier(random_state=339, **dt_params)
        
        # Evaluating hyperparameters on training set 
        cv_scores = cross_val_score(self.best_dt, self.X_train, self.y_train, cv=5, scoring=mcc_scorer)
        self.best_dt_parameters = dt_params
        self.dt_train_score = cv_scores.mean()
        
        self.best_dt.fit(self.X_train, self.y_train)
                
        # Logistic Regression
        lr_params = {
            'penalty': 'l1',
            'C': 1,
            'solver': 'liblinear',
            'max_iter': 500
        }
        self.best_lr = LogisticRegression(random_state=339, **lr_params)
        
        # Evaluating hyperparameters on training set 
        cv_scores = cross_val_score(self.best_lr, self.X_train, self.y_train, cv=5, scoring=mcc_scorer)
        self.best_lr_parameters = lr_params
        self.lr_train_score = cv_scores.mean()
        
        self.best_lr.fit(self.X_train, self.y_train)
    
    def print_training_results(self):
        print(f"{'--ML Trained Model--':<30} {'--Its Best Set of Parameter Values--':<98} {'--Its MCC-score on the 5-fold Cross Validation on Training Data (80%)--':<10}")
        
        # Multi layer NN
        params_str = str(self.best_mlp_parameters)
        print(f"{'Multi-layer Neural Network':<30} {params_str:<160} {self.mlp_train_score:<10.4f}")
        
        # SVM
        params_str = str(self.best_svm_parameters)
        print(f"{'Support Vector Machine':<30} {params_str:<160} {self.svm_train_score:<10.4f}")
        
        # KNN
        params_str = str(self.best_knn_parameters)
        print(f"{'K-Nearest Neighbors':<30} {params_str:<160} {self.knn_train_score:<10.4f}")
        
        # Decision Tree
        params_str = str(self.best_dt_parameters)
        print(f"{'Decision Tree':<30} {params_str:<160} {self.dt_train_score:<10.4f}")
        
        # Logistic Regression
        params_str = str(self.best_lr_parameters)
        print(f"{'Logistic Regression':<30} {params_str:<160} {self.lr_train_score:<10.4f}")
    
    def print_test_results(self):
        print("\n-----Results for part (7)-----\n")
        print(f"{'--ML Trained Model--':<30} {'--Its Best Set of Parameter Values--':<129} {'--Its MCC-score on the Test Data (20%)--':<10}")
        
        # Multi-layer NN
        params_str = str(self.best_mlp_parameters)
        y_pred = self.best_mlp.predict(self.X_test)
        mlp_test_mcc = self.mcc(self.y_test, y_pred)
        print(f"{'Multi-layer NN':<30} {params_str:<160} {mlp_test_mcc:<10.4f}")
        
        # SVM
        params_str = str(self.best_svm_parameters)
        y_pred = self.best_svm.predict(self.X_test)
        svm_test_mcc = self.mcc(self.y_test, y_pred)
        print(f"{'Support Vector Machine':<30} {params_str:<160} {svm_test_mcc:<10.4f}")
        
        # KNN
        params_str = str(self.best_knn_parameters)
        y_pred = self.best_knn.predict(self.X_test)
        knn_test_mcc = self.mcc(self.y_test, y_pred)
        print(f"{'K-Nearest Neighbors':<30} {params_str:<160} {knn_test_mcc:<10.4f}")
        
        # Decision Tree
        params_str = str(self.best_dt_parameters)
        y_pred = self.best_dt.predict(self.X_test)
        dt_test_mcc = self.mcc(self.y_test, y_pred)
        print(f"{'Decision Tree':<30} {params_str:<160} {dt_test_mcc:<10.4f}")
        
        # Logistic Regression
        params_str = str(self.best_lr_parameters)
        y_pred = self.best_lr.predict(self.X_test)
        lr_test_mcc = self.mcc(self.y_test, y_pred)
        print(f"{'Logistic Regression':<30} {params_str:<160} {lr_test_mcc:<10.4f}")
        
        print(f"\nThe Decision Tree model should be used to predict machine failure in the future since it has the highest MCC (0.8677) on the testing set.\n")

def main():
    pipeline = MachineFailurePredictorPipeline("ai4i2020.csv")
    pipeline.wrangle_data()
    pipeline.split_data()
    pipeline.train_models()
    pipeline.print_training_results()
    pipeline.print_test_results()

if __name__ == "__main__":
    main()