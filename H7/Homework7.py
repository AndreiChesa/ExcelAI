from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import sklearn.datasets
    
breast_cancer_data = sklearn.datasets.load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=42)

def evaluate_model(n_neighbors):
    model = KNeighborsClassifier(n_neighbors=n_neighbors,weights='distance')
    model.fit(x_train, y_train)
    
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    #print(classification_report(y_test, prediction, digits=5))
    print(f"k = {n_neighbors} acc_train = {accuracy_score(y_train, y_train_pred)} acc_test = {accuracy_score(y_test, y_test_pred)}")

for i in range(1,40,2):
    evaluate_model(i)
    
# plt.scatter(breast_cancer_data.data[:, 0], breast_cancer_data.data[:, 1], c=breast_cancer_data.target)
# plt.title("Breast Cancer Data")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.colorbar()
# plt.show()

