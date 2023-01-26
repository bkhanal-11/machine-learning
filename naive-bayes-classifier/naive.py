import numpy as np
import sys

from classifier import Classifier

gnb = Classifier()

## Loading datasets

# path_1 = str(sys.argv[1])
# path_2 = str(sys.argv[2])

# embeddings = gnb.load_file(path_1)
# labels = gnb.load_file(path_2)

embeddings = gnb.load_file('Embeddings/emb_array.npz')
labels = gnb.load_file('Embeddings/label_array.npz')

# Counting Number of classes and their components
classes, classes_count = gnb.count(labels)

# Separating embeddings by class
embeddings_by_classes = []
labels_by_classes = []

index = 0
for i in range(len(classes)):
    temp = embeddings[index:index+classes_count[i]]
    temp2 = labels[index:index+classes_count[i]]
    index += classes_count[i]
    embeddings_by_classes.append(temp)
    labels_by_classes.append(temp2)

# Dividing data into train and test
train, test = gnb.divide_data(0.7, embeddings_by_classes)
train_labels, test_labels = gnb.divide_data(0.7, labels_by_classes)

x = embeddings.shape[1]

# Means and Standard deviation of indivdual classes and 
# column vectors of embeddings

train_mean, train_std = gnb.mean_std(train, x)

# Predicting from test dataset
predictions = []

for i in range(len(test)):
    temp = gnb.naive_bayes(train_mean, train_std, test[i])
    predictions.append(temp)

# Calculating accuracy
accuracy = 0
for i in range(len(predictions)):
    accuracy += gnb.calc_accuracy(test_labels[i], predictions[i])

accuracy = accuracy / len(predictions)

print(accuracy)