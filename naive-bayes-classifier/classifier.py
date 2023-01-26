import numpy as np

class Classifier:
    def __init__(self):
        '''
        Gaussian Naive Bayes Classifier
        '''
        self.classes = None
        self.classes_count = None
    
    def load_file(self, filename):
        temp = np.load(filename, allow_pickle=True)
    
        for item in temp.files:
            data = temp[item]
            
        return data
    
    def mean(self, embeddings):

        return sum(embeddings)/float(len(embeddings))

    def stdev(self, embeddings):
        avg = self.mean(embeddings)
        variance = sum([(x-avg)**2 for x in embeddings]) / float(len(embeddings) - 1)
        
        return np.sqrt(variance)

    def gaussian_probability(self, x, mean, stdev):
        exponent = np.exp(-((x - mean)**2 / (2 * stdev**2 )))
        exponent = (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent
        
        return exponent
    def count(self, labels):
        self.classes, self.classes_count = np.unique(labels, return_counts = True)

        return [self.classes, self.classes_count]

    def divide_data(self, ratio, embeddings):
        train = []
        test = []

        for i in range(len(embeddings)):
            temp = embeddings[i]
            temp2 = temp[0:int(temp.shape[0]*ratio)]
            temp3 = temp[int(temp.shape[0]*ratio):]
            
            train.append(temp2)
            test.append(temp3)
        return [train, test] 

    def mean_std(self, data, x):
        train_mean = np.zeros((len(self.classes_count), x))
        train_std = np.zeros((len(self.classes_count), x))

        x ,y = train_mean.shape

        for i in range(x):
            for j in range(y):
                train_mean[i,j] = self.mean(data[i][:,j])
                train_std[i,j] = self.stdev(data[i][:,j])
        
        return [train_mean, train_std]
    
    def predict(self, mean, std, test):
        x, y = mean.shape
        probabilities = np.zeros(x)
        
        for i in range(x):
            prod = 1
            for j in range(y):
                prod *= self.gaussian_probability(test[j], mean[i,j], std[i,j]) / 5 # normalize
            probabilities[i] = prod / x
        
        best_label, best_prob = None, -1
        for i in range(x):
            if best_label is None or probabilities[i] > best_prob:
                best_prob = probabilities[i]
                best_label = self.classes[i]
        return best_label

    def naive_bayes(self, mean, std, test):
        predictions = []
        for i in range(test.shape[0]):
            output = self.predict(mean, std, test[i])
            predictions.append(output)
        
        return np.array(predictions)
    
    def calc_accuracy(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1

        return correct / float(len(actual)) * 100.0
    
    