# Logistic Regression

Logistic Regression is the most widely used machine learning algorithm for classification problem. In its original form, it is used for binary classification problem.

## Sigmoid Function

A sigmoid function is a mathematical function having a characteristic "s"-shaped curve. Mathematically,

```math
\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^{z}}{1 + e^{z}}
```

![Sigmoid Function](assets/sigmoid.png)

Sigmoid function gives us the probability of being into class 1 or class 0. So, generaly we take the threshold as 0.5 and say if $P \geq 0.5$, then it belongs to class 1 and if $P \leq 0.5$, then it belongs to class 0.


In comparison to linear regression where we minimize sum of squared errors (SSE) i.e. 

```math
SSE = \sum (y_{i} - {\hat{y}}_{i})^{2}
```

we maximize log-likelihood instead. The main reason behind this is that SSE is not a convex function hence finding a single minima would not be easy, there could be more than one minima. However, log-likelihood is a convex function and hence finding optimal parameters is easier.

```math
log-likelihood = \sum y_{i} log \hat{y}_{i} + (1 - y_{i}) log (1 - \hat{y}_{i})
```