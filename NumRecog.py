import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import sys

# Importing data from datasets
data = ds.load_digits() # 1797 images of 8x8 containing numbers written by hand 
x,y = data.images, data.target
x = x.reshape(1797,64) # Reshaping data since each image is a matrix
x_train, x_test ,y_train, y_test = train_test_split(x,y)

if sys.argv[1] == "gaussian":
    # Using gaussian model
    clf = GaussianNB()
elif sys.argv[1] == "tree":
    # Using a decision tree
    clf = DecisionTreeClassifier()
elif sys.argv[1] == "svm":
    # Using support vector machine
    clf = SVC()
elif sys.argv[1] == "knn":
    # Using K Nearest Neighbors
    clf = KNeighborsClassifier()
elif sys.argv[1] == "mlp":
    # Using a multlayer perceptron
    clf = MLPClassifier(hidden_layer_sizes=[32,16])
clf.fit(x_train,y_train)
pred = clf.predict(x_test)
print(confusion_matrix(pred,y_test))
print(accuracy_score(pred,y_test))
