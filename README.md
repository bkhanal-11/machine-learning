# Machine Learning

For resources: ![Machine Learning Collections](https://github.com/aladdinpersson/Machine-Learning-Collection.git)

For datasets: ![Standard Machine Learning Datasets](https://machinelearningmastery.com/standard-machine-learning-datasets/)

# ML Algorithms

## Decision Trees and Random Forest

### Decision Trees

A decision tree is a type of flowchart that shows a clear pathway to a decision. In  term of data analytics, it is a type of algorithm that includes conditional 'control' statements to classify data. A decision tree starts at a single point (or 'node') which then branches (or 'splits') in two or more directions. Each branch offers different possible outcomes, incorporating a variety of decisions and chance events until a final outcome is achieved.

In general, we can break down the decision tree algorithm into a series of steps common across different implementations of decision tree.

1. Start with the entire dataset and look at every feature or attribute. Look at all of the possible values of that attribute and pick a value which best splits the dataset into different regions. What constitutes ‘a best split’ depends very much on whether we are building a regression tree algorithm or a classification tree algorithm. We’ll expand upon the different methods for finding the best split below.

2. Split the dataset at the root node of the tree and move to the child nodes in each branch. For each decision node, repeat the attribute selection and value for best split determination. This is a greedy algorithm: it only looks at the best local split (not global optimum) given the attributes in its region to improve the efficiency of building a tree.

3. Continue iteratively until either:

  - We have grown terminal or leaf nodes so they reach each individual sample (there were no stopping criteria).

  - We reached some stopping criteria. For example, we might have set a maximum depth, which only allows a certain number of splits from the root node to the terminal nodes. Or we might have set a minimum number of samples in each terminal node, in order to prevent terminal nodes from splitting beyond a certain point.

Decision tree regressors use variance reduction as a measure of the best split. Variance reduction indicates how homogenous our nodes are. If a node is extremely homogeneous, its variance (and the variance of its child nodes) will not be as big. The formula for variance is:

```math
Var = \frac{\sum (X - \mu)^{2}}{n}
```

### Random Forest

Random forest is a supervised learning algorithms. The "forest" it builds, is an essemble of decision trees, usually trained with the "bagging" method. The general idea of the bagging method is that a combination of learning models increases the overall result.

Simply, random forest builds multiple decision trees and merges them together to get a more accurate and stable predictions. One of the advantage of random forest is that it can be used for both classification and regression problem.
