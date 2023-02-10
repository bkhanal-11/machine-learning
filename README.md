# Machine Learning
Implementation of different machine learning techniques,

For resources: ![Machine Learning Collections](https://github.com/aladdinpersson/Machine-Learning-Collection.git)

For datasets: ![Standard Machine Learning Datasets](https://machinelearningmastery.com/standard-machine-learning-datasets/)

# ML Algorithms

## Linear Regression

Just like Naive Bayes is a good starting point for classification tasks, linear regression models are a good starting point for regression tasks. It is a machine learning algorithm based on supervised learning. It models a target prediction value based on independent varaibles. It is mostly used for finding out the relationship between variables and forecasting. Linear regression perform the task to predict a dependent variable value $y$ based on a given independent variable $x$. Hypothesis function for Linear Regression:

```math
y = \theta_{1} + \theta_{2} x
```

where $x$ is input training data and $y$ is the label to the data.

When training the model, it fits the best line to predict the value of $y$ for a given value of $x$. The model gets the best regression fit line by finding intercept $\theta_{1}$ and coefficient (slope) of $x$, $\theta_{2}$. 

By achieving the best fit regression line, the model aims to predict $y$ value such that the error difference between predicted value and true value is minimum. So, it is very important to update the $\theta_{1}$ and $\theta_{2}$ values, to reach the best value that minimize the error between predicted $y$, $pred$ and true $y$.

```math
\mathcal{J} = \frac{1}{n} \sum_{i=1}^{n} (pred_{i} - y_{i})^{2}
```

Cost function $\mathcal{J}$ of Linear Regression is Root Mean Square Error (RMSE) between predicted $y$ and true $y$.

To update $\theta_{1}$ and $\theta_{2}$ values in order to reduce cost function (minimizing RMSE) and achieving the best fit line the model uses Gradient Descent. The idea is to start with random $\theta_{1}$ and $\theta_{2}$ values and then iteratively updating the values, reaching minimum cost. 

## Support Vector Machine (SVM)

SVM are powerful yet flexible supervised machine learning algorithms which are used both for classification and regression. But generally, they are used in classification problem. They are popular because of their ability to handle multiple continuous and categorical variables.

An SVM model is basically a representation of different classes in a hyperplane in multidimensional space. The hyperplane will be generated in an iterative manner by SVM so that error can be minimized. The goal of SVM is to divide the datasets into classes to mind a minimum marginal hyperplane.

Some important concepts in SVM:

1. Support vectors

Datapoints that are closest to the hyperplane is called support vectors. Separating line will be defined with the help of these data points.

2. Hyperplane

It is a decision plane or space which is divided between a set of objects having different classes.

3. Margin

It may be defined as the gap between two lines on the closest data point of different classes. It can be calculated as the perpendicular distance from the line to the support vectors. Large margin is considered as a good margin while samll margin is considered as a bad margin.

The main goal of SVM is to divide the dataset into classes to find a maximum marginal hyperplane as follow:

- First, SVM will generate hyperplanes iiteratively that segregates the classes in best way.

- Then, it will choose the hyperplane that separate the classes correctly.

Building an optimized hyperplane in a non linearly separable problem is done using kernels. The kernels are mathematically functions that convert the complex problem using the linear algebric form.

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
