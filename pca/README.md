# Principal Component Analysis (PCA)

PCA is a dimensionality-reduction method that is often used to reduce the dimensionality of large datasets, by transforming a large dataset of variables into a smaller one that still contains most of the information in the large set.

Generally, a good dimension reduction method, that is, a good feature map $f = R^{n} \to R^{m}$ should preserve much information contained in the high dimensional patterns $x \in R^{n}$ and encode it robustly in the feature vectors $y = f(x)$. A feature representation $f(x)$ preserves information about $x$ to the extent that $x$ can be reconstructed from $f(x)$. That is, we wish to have a decoding function $d:R^{m} \to R^{n}$ which leads back from the feature vector encoding $f(x)$ to $x$ that is we wish to achieve

```math
x \approx d \circ f(x)
```

When $f$ and $d$ are confined to linear functions, and when the similarity $x \approx d \circ f(x)$ is measured by mean squared error, the optimal solution for $f$ and $d$ can be easily and cheaply computed by a method that is PCA.

The first step in PCA is to center the training pattern $x_{i}$, that is, subtract their mean $\mu = \frac{1}{N} \sum_{i} x_{i}$ from each pattern, obtaining centered patterns $\bar{x_{i}} = x_{i}-\mu$. The centered pattern form a point  cloud in $R^{n}$ whose center of gravity is the origin.

The point cloud will usually not be perfectly spherically shaped, but instead extended in some direction more than in others. "Directions" in $R^{n}$ are characterized by unit-norm "direction" vectors $u \in R^{n}$. The distance of a point $\bar{x_{i}}$ from the origin in the direction of $u$ is given by the projection of $\bar{x_{i}}$ on $u$, that is, the inner product $u'\bar{x_{i}}$.

The "extension" of a centered point cloud {$\bar{x_{i}}$} in a direction $u$ is defined to be the mean squared distance to the origin of the points $\bar{x_{i}}$ in the direction of $u$. the direction of the largest extension of the point cloud is hence the direction vector given by 

```math
u_{1} = \underset{u, ||u||=1}{\mathrm{argmax}} \frac{1}{N} \sum_{i} (u' \bar{{x}_i})^2
```

Notice that since the cloud $\bar{x_{i}}$ is centered, the mean of the all $u' \bar{x_{i}}$ is zero, and hence the number $\frac{1}{N} \sum_{i} (u' \bar{x_{i}})^2$ is the variance of the numbers $u'\bar{x_{i}}$.

The $u_{1}$ points in the "longest" direction of the pattern cloud. The vector $u_{1}$ is called the first principal component (PC) of the centered point cloud.

Next step is to project patterns on the $(n-1)$ dimensional linear subspace of $R^{n}$ that is orthogonal to $u_{1}$. That is, map pattern point $\bar{x}$ to $\bar{x^{\ast}} = \bar{x} - (u'x)u_{1}$. Within the "flattened" pattern cloud, again find the direction vector of greatest variance

```math
u_{2} = \underset{u, ||u||=1}{\mathrm{argmax}} \frac{1}{N} \sum_{i} (u' \bar{{x^{\ast}}_i})^2
```

and call it the second PC of the centered pattern sample. From the procedure, it is clear that $u_{1}$ and $u_{2}$ are orthogonal, because $u_{2}$ lies in the orthogonal subspace of $u_{1}$.

Now repeat this procedure: In iteration $k$, the $k^{th}$ PC $u_{k}$ is constructed by projecting pattern points to the linear subspace that is orthogonal to the already computed PCs $u_{1},...,u_{k}$ is obtained as the unit-length vector pointing in the "longest" direction of the current $(n-k+1)$-dimensional pattern point distribution. This can be repeated until $n$ PCs $u_{1},...,u_{n}$ have been determined.

### A brief Summary

**Data** A set $(x_i)_{i=1,...,N}$ of n-dimensional pattern vectors.

**Result** An n-dimensional mean pattern vector $\mu$ and $m$ principal component vectors arranged column-wise in an $n \times m$ sized matrix $U_{m}$.

**Procedure** 

- Step 1: Compute the pattern $\mu$ and center patterns to obtain a centered pattern matrix $\bar{X} = [\bar{x_{1}},...,\bar{x_{N}}]$.

- Step 2: Compute SVD $U \sum U'$ of $c = \frac{1}{N} \bar{X} \bar{X'}$ and keep from $U$ only the first $m$ columns making for a $n \times m$ sized matrix $U_{m}$.

**Usage for compression**

In order to compress a new n-dimensional pattern to a m-dimensional feature vector $f(x)$, compute $f(x)={U'}_{m} \bar{x}$.

**Usage for uncompression (decoding)**

In order to approximately restore $x$ from its feature vector $f(x)$, compute $x_{restored} = \mu + U_{m} f(x)$. Equivalently, in a more compact notion, compute $x_{restored} = [U_{m}, \mu][f(x);1]$.
