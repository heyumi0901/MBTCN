import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import seaborn as sns

def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    print("Number of misclassified points ", (len(test_y) - np.trace(C)) / len(test_y) * 100)
    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j

    A = (((C.T) / (C.sum(axis=1))).T)
    # divid each element of the confusion matrix with the sum of elements in that column

    # C = [[1, 2],
    #     [3, 4]]
    # C.T = [[1, 3],
    #        [2, 4]]
    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =1) = [[3, 7]]
    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]
    #                           [2/3, 4/7]]

    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]
    #                           [3/7, 4/7]]
    # sum of row elements = 1

    B = (C / C.sum(axis=0))
    # divid each element of the confusion matrix with the sum of elements in that row
    # C = [[1, 2],
    #     [3, 4]]
    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =0) = [[4, 6]]
    # (C/C.sum(axis=0)) = [[1/4, 2/6],
    #                      [3/4, 4/6]]

    labels = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
    cmap = sns.light_palette("cornflowerblue")
    # representing A in heatmap format
    print("-" * 50, "Confusion matrix", "-" * 50)
    plt.figure(figsize=(20, 20))
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground truth Class')
    plt.show()

    print("-" * 50, "Precision matrix", "-" * 50)
    plt.figure(figsize=(15, 15))
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground truth Class')
    plt.show()
    print("Sum of columns in precision matrix", B.sum(axis=0))

    # representing B in heatmap format
    print("-" * 50, "Recall matrix", "-" * 50)
    plt.figure(figsize=(15, 15))
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground truth Class')
    plt.show()
    print("Sum of rows in precision matrix", A.sum(axis=1))


def plot_confusion_matrix_CSTR(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    print("Number of misclassified points ", (len(test_y) - np.trace(C)) / len(test_y) * 100)
    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j

    A = (((C.T) / (C.sum(axis=1))).T)
    # divid each element of the confusion matrix with the sum of elements in that column

    # C = [[1, 2],
    #     [3, 4]]
    # C.T = [[1, 3],
    #        [2, 4]]
    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =1) = [[3, 7]]
    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]
    #                           [2/3, 4/7]]

    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]
    #                           [3/7, 4/7]]
    # sum of row elements = 1

    B = (C / C.sum(axis=0))
    # divid each element of the confusion matrix with the sum of elements in that row
    # C = [[1, 2],
    #     [3, 4]]
    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =0) = [[4, 6]]
    # (C/C.sum(axis=0)) = [[1/4, 2/6],
    #                      [3/4, 4/6]]

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    cmap = sns.light_palette("cornflowerblue")
    # representing A in heatmap format
    print("-" * 50, "Confusion matrix", "-" * 50)
    plt.figure()
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground truth Class')
    plt.show()

    print("-" * 50, "Precision matrix", "-" * 50)
    plt.figure(figsize=(15, 15))
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground truth Class')
    plt.show()
    print("Sum of columns in precision matrix", B.sum(axis=0))

    # representing B in heatmap format
    print("-" * 50, "Recall matrix", "-" * 50)
    plt.figure(figsize=(15, 15))
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground truth Class')
    plt.show()
    print("Sum of rows in precision matrix", A.sum(axis=1))