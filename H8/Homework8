from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

y_true = np.array([1,0,0,1,1])
y_pred = np.array([1,0,1,1,0])

def my_accuracy(y_true, y_pred):
    count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            count += 1  
    return count / len(y_true)

def my_precision(y_true, y_pred):
    true_positives = 0
    false_positives = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            true_positives += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            false_positives += 1
    
    if true_positives + false_positives == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)

def my_recall(y_true, y_pred):
    true_positives = 0
    false_negatives = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            true_positives+=1
        elif y_true[i] == 1 and y_pred[i] == 0:
            false_negatives+=1
    return true_positives / (true_positives + false_negatives)

def my_f1_score(y_true, y_pred):
    p = my_precision(y_true,y_pred)
    r = my_recall(y_true, y_pred)
    return (2 * p * r) / (p + r)

print("Accuracy check:", my_accuracy(y_true, y_pred) == accuracy_score(y_true, y_pred))
print("Precision check:", my_precision(y_true, y_pred) == precision_score(y_true, y_pred))
print("Recall check:", my_recall(y_true, y_pred) == recall_score(y_true, y_pred))
print("F1-score check:", my_f1_score(y_true, y_pred) == f1_score(y_true, y_pred))

print("\nMy functions:")
print(f"My Accuracy: {my_accuracy(y_true, y_pred)}")
print(f"My Precision: {my_precision(y_true, y_pred)}")
print(f"My Recall: {my_recall(y_true, y_pred)}")
print(f"My F1-score: {my_f1_score(y_true, y_pred)}")

print("\nSklearn functions:")
print(f"Sklearn Accuracy: {accuracy_score(y_true, y_pred)}")
print(f"Sklearn Precision: {precision_score(y_true, y_pred)}")
print(f"Sklearn Recall: {recall_score(y_true, y_pred)}")
print(f"Sklearn F1-score: {f1_score(y_true, y_pred)}")

