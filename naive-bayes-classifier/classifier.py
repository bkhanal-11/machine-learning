import numpy as np

class Classifier:
    def __init__(self):
        '''
        Gaussian Naive Bayes Classifier
        '''
        self.classes = None
        self.classes_count = None
        self.train_mean = None
        self.train_std = None
    
    def load_file(self, filename):
        temp = np.load(filename, allow_pickle=True)
    
        for item in temp.files:
            data = temp[item]
            
        return data
    
    def mean(self, X):
        return sum(X)/float(len(X))

    def stdev(self, X):
        avg = self.mean(X)
        variance = sum([(x-avg)**2 for x in X]) / float(len(X) - 1)
        
        return np.sqrt(variance)

    def gaussian_probability(self, x, mean, stdev):
        exponent = np.exp(-((x - mean)**2 / (2 * stdev**2 )))
        exponent = (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent
        
        return exponent
    
    def count(self, labels):
        self.classes, self.classes_count = np.unique(labels, return_counts = True)

        return [self.classes, self.classes_count]

    def divide_data(self, ratio, embeddings, labels):
        train_emb = []
        train_label = []
        test_emb = []
        test_label = []

        for embedding, label in zip(embeddings, labels):
            partition = int(embedding.shape[0]*ratio)
            train_e = embedding[0:partition]
            test_e = embedding[partition:]

            train_l = label[0:partition]
            test_l = label[partition:]
            
            train_emb.append(train_e)
            test_emb.append(test_e)

            train_label.append(train_l)
            test_label.append(test_l)

        return [train_emb, test_emb, train_label, test_label] 

    def mean_std(self, data, x):
        self.train_mean = np.zeros((len(self.classes_count), x))
        self.train_std = np.zeros((len(self.classes_count), x))

        x ,y = self.train_mean.shape

        for i in range(x):
            for j in range(y):
                self.train_mean[i,j] = self.mean(data[i][:,j])
                self.train_std[i,j] = self.stdev(data[i][:,j])
    
    def predict(self, test):
        x, y = self.train_mean.shape
        probabilities = np.zeros(x)
        
        for i in range(x):
            prod = 1
            for j in range(y):
                prod *= self.gaussian_probability(test[j], self.train_mean[i,j], self.train_std[i,j]) / 5 # normalize
            probabilities[i] = prod / x
        
        best_label, best_prob = None, -1
        for i in range(x):
            if best_label is None or probabilities[i] > best_prob:
                best_prob = probabilities[i]
                best_label = self.classes[i]
        return best_label

    def naive_bayes(self, test):
        predictions = []
        for i in range(test.shape[0]):
            output = self.predict(test[i])
            predictions.append(output)
        
        return np.array(predictions)
    
    def calc_accuracy(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1

        return correct / float(len(actual)) * 100.0
    
    