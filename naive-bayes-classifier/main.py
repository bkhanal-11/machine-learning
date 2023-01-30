from classifier import Classifier
from tqdm import tqdm

gnb = Classifier()

## Loading datasets
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
train_embeddings, test_embeddings, train_labels, test_labels = gnb.divide_data(0.7, embeddings_by_classes, labels_by_classes)

x = embeddings.shape[1]

# Means and Standard deviation of indivdual classes and 
# column vectors of embeddings

gnb.mean_std(train_embeddings, x)

# Predicting from test dataset
predictions = []

for item in tqdm(test_embeddings):
    predictions.append(gnb.naive_bayes(item))

# Calculating accuracy
accuracy = 0
for i in range(len(predictions)):
    accuracy += gnb.calc_accuracy(test_labels[i], predictions[i])

accuracy = accuracy / len(predictions)

print(f'Accuracy of the algorithm: {accuracy}')