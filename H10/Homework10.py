import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

class MinMaxScalerPart:
    
    @staticmethod
    def generate_random_numbers(n, a, b):
        random_numbers = np.random.uniform(a, b, n)
        return random_numbers

    @staticmethod
    def testing_sklearn(data):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        
        print(f"Scaled data: {scaled_data.flatten()}")
        print("-" * 40)
        
        return scaled_data

    @staticmethod
    def testing_my_minmax(data):
        min_val = np.min(data)
        max_val = np.max(data)
        
        scaled_data = (data - min_val) / (max_val - min_val)
        
        print(f"My scaled data: {scaled_data}")
        print("-" * 40)
        
        return scaled_data

    @staticmethod
    def testing_my_minmax_2(data):
        min_val = 1e12
        max_val = -1e12
        for i in data:
            if i < min_val:
                min_val = i
            if i > max_val:
                max_val = i

        scaled_data = (data - min_val) / (max_val - min_val)

        print(f"My scaled data: {scaled_data}")
        print("-" * 40)
        return scaled_data

    @staticmethod
    def run_tests(teste : int):
        for i in range(1, teste + 1):
            
            print("-" * 40)
            print(f"Test {i}:")
            print("-" * 40)

            rand = MinMaxScalerPart.generate_random_numbers(10, 3, 15)
            print(f"Original data: {rand}")

            print("SKLEARN VERSION:")
            rez_sk = MinMaxScalerPart.testing_sklearn(rand)
            
            print("MY CUSTOM VERSION WITH NUMPY:")
            rez_num = MinMaxScalerPart.testing_my_minmax(rand)

            print("MY CUSTOM VERSION WITHOUT NUMPY:")
            rez_my = MinMaxScalerPart.testing_my_minmax_2(rand)
            
            print("COMPATIBILITY CHECK:")
            sklearn_flat = rez_sk.flatten()
            
            numpy_match = np.allclose(sklearn_flat, rez_num)
            manual_match = np.allclose(sklearn_flat, rez_my)
            all_match = np.allclose(rez_num, rez_my)
            
            print(f"Sklearn vs NumPy version: {numpy_match}")
            print(f"Sklearn vs Manual version: {manual_match}")
            print(f"NumPy vs Manual version: {all_match}")
            print(f"All methods match: {numpy_match and manual_match and all_match}")
            
            print("=" * 50)

#MinMaxScalerPart.run_tests(1)

class CrossValidationPart:

    @staticmethod
    def hold_out_test(data):
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
        kNN = KNeighborsClassifier(5, weights='distance')
        kNN.fit(X_train, y_train)

        y_test_pred = kNN.predict(X_test)
        y_train_pred = kNN.predict(X_train)
        
        print("=" * 40)
        print(f"Hold out test")
        print("-" * 40)
        print(f"Train accuracy : {accuracy_score(y_train,y_train_pred)}")
        print(f"Test accuracy : {accuracy_score(y_test,y_test_pred)}")
        print("=" * 40)

    @staticmethod
    def cross_val_test(data):
        kNN = KNeighborsClassifier(5, weights='distance')
        cv_scores = cross_val_score(estimator=kNN, X=data.data, y=data.target, cv=5, scoring='f1_macro')
        
        print("=" * 40)
        print("Cross Validation Test (5-fold)")
        print("-" * 40)
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
        print("=" * 40)
    
    @staticmethod
    def stratified_cross_val_test(data):
        kNN = KNeighborsClassifier(5, weights='distance')
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        cv_scores = cross_val_score(estimator=kNN, X=data.data, y=data.target, cv=skf, scoring='f1_macro')

        print("=" * 40)
        print("Stratified Cross Validation Test (5-fold)")
        print("-" * 40)
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
        print("=" * 40)

    @staticmethod
    def custom_fold_split_test(X, y, k):
        return

    @staticmethod
    def run_tests():
        breast_cancer_data = sklearn.datasets.load_breast_cancer()
        CrossValidationPart.hold_out_test(breast_cancer_data)
        CrossValidationPart.cross_val_test(breast_cancer_data)
        CrossValidationPart.stratified_cross_val_test(breast_cancer_data)

CrossValidationPart.run_tests()
