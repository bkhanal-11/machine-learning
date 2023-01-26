# Facial Recognition

## FaceNet

It learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. Once this space has been produced, task such as face recognition, verification and clustering can be easily implemented using standard techniques with FaceNet embeddings as features vectors.

It uses a deep convolutional network trained to directly optimize the embedding itself rather than an intermidiate bottleneck layer. The model structure is given as:

```math
Batch \to Deep CNN \to L_{2}-normalization \to Embeddings \to Triplet Loss
```
**Triplet Loss** minimizes the distance between an anchor and a positive, both of which have same identity, and maximizes the distance between the anchor and a negative of a different identity.

### Loss function

Given 3 images Anchor $`A`$, Positive $`P`$ and Negative $`N`$,

```math
\mathcal{L} (A, P, N)  = max({||f(A) - f(P)||}^{2} - {||f(A) - f(N)||}^{2} + \alpha, 0)
```

where $`f()`$ is embedding of images, $`\alpha`$ is the margin. The main objective is to minimize the following cost function:


```math
\mathcal{J} = \sum_{i = 0}^{N} \mathcal{L} (A^{(i)}, P^{(i)}, N^{(i)})
```

During training, if $`A, P, N`$ are chosen randomly, $`d(A,P) + \alpha \leq d(A,N)`$ is easily satisfied. Hence, choose triplets that are "hard" to train on such that SGD works hard to minimize the cost function.
## Naïve Bayes Classifier

### Conditional Probability

Simply speaking, conditional probabilty is the probabilty of occurance of an event given the probabilty of another event. Mathematically,  conditional probabilty of event A given probability of event B is given as

```math
P(A|B) = \frac{P(A \cap B)}{P(B)}
```

### Bayes Theorem

Similar to conditional probability, Bayes Theorem also describes the probability of an event, based on prior knowledge of conditions that might be related to the event. Mathematically,

```math
P(A|B) = \frac{P(B|A) P(A)}{P(B)}
```

where $`A`$ and $`B`$ are events, $`P(B) \neq 0`$, $`P(A|B), P(B|A)`$  are the likelihood of event $`A`$ occurring given that occurance of $`B`$ and viceversa.

Naïve Bayes Classifiers are a collection of classification algorithms based on Bayes Theorem. It is not a single algorithm bit a family of algorithms where all them share a common principle i.e. every pair of features being classified is independent of each other, hence the name.

The dataset is divided into two parts, namely feature matrix and reponse vector. Feature matrix contains all the vectors (row) of dataset in which each vector consists of the value of dependent features. Response vector contains the value of class variable for each row of feature matrix. The fundamental Naïve Bayes assumption is that each feature makes an independent and equal contribution to the outcome. 

For a dataset, we can write Bayes Theorem as

```math
P(y|X) = \frac{P(X|y) P(y)}{P(X)}
```

where, $`y`$ is class variable and $`X`$ is dependent feature vector (of size $`n`$), i.e.

```math
X = (x_{1},x_{2},x_{3},.....,x_{n})
```

For our dataset, where, $`y`$ is labels and $`X`$ is facial embedding vector (of size $`n = 512`$).

For two independent events $`A`$ and $`B`$, we have relation $`P(A \cap B) = P(A)P(B)`$. Putting these assumption to Bayes Theorem, which is independence among the features, we get

```math
P(y|x_{1},x_{2},x_{3},.....,x_{n}) = \frac{P(x_{1}|y)P(x_{2}|y),..,P(x_{n}|y)P(y)}{P(x_{1})P(x_{2}).....P(x_{n})}
```

```math
P(y|x_{1},x_{2},x_{3},.....,x_{n}) = \frac{P(y) \prod_{i=0}^{n} P(x_{i}|y)}{P(x_{1})P(x_{2}).....P(x_{n})}
```

Since denominator remains constant for given input,

```math
P(y|x_{1},x_{2},x_{3},.....,x_{n}) \propto P(y) \prod_{i=0}^{n} P(x_{i}|y)
```

We need to create a classifier model. For this, we find the probability of given set of inputs for all possible values of the class variable y and pick up the output with maximum probability. This can be expressed mathematically as:

```math
y = \underset{y}{\mathrm{argmax}}P(y) \prod_{i=0}^{n} P(x_{i}|y)
```

So, we need to calculate $`P(y)`$ and $`P(x_{i}|y)`$. The different Naïve Bayes classifiers differ mainly by the assumptions they make regarding the distribution of $`P(x_{i}|y)`$.

One of those classifier is Gaussian Naïve Bayes. The continuous values associated with each feature are assumed to be distributed according to a Gaussian distribution. When plotted, it gives a bell shaped curve which is symmetric about the mean of the feature values. Hence, conditional probabilty is given as 

```math
P(x_{i}|y) = \frac{1}{\sqrt{2 \pi {\sigma_{y}}^{2}}} e^{\frac{- (x_{i}- \mu_{y})^{2}}{2{\sigma_{y}}^{2}}}
```

And we can calculate $`P(y)`$ using the $`y`$ dataset.