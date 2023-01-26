# K-means Clustering

Generally, the goal in "clustering" is to minimize variance in data given clusters. It gives us an "chicken-egg" problem i.e. 

1. If we knew the *cluster centers*, we could allocate points to groups by assigning each to its closest center.

2. If we knew the group elements, we could get the centers by computing the mean per group.

Best cluster centers are those that minimize Sum of Squared Distance (SSD) between all points and their nearest cluster $`c_{i}`$,

```math
SSD = \sum_{cluster i} \sum_{x \in cluster i} ( x - {c_{i}}^2)
```

Hence, k-means clustering can be carried out as follow:

1. Initialize ($`t = 0`$): cluster centers $`c_{1}. c_{2},...,c_{k}`$
     - commonly used: random initialization 
     - or greedy choose K to minimize residual

2. Compute $`\delta^{t}`$: assign each point to the closest center
     - Typical distance measure:
          - Euclidean, $`dist(x, x') = x^{T}x`$
          - Cosine, $`dist(x, x') = \frac{x^{T}x}{||x|| ||x'||}`$

3. Compute $`c^{t}`$: update cluster centers as the means of points
```math
c^{t} = \underset{c}{\mathrm{argmin}} \frac{1}{N} \sum_{i}^{N} \sum_{i}^{k} {\delta_{ij}}^{t}({c_{i}}^{t-1}-x_{j})^2
```

4. Update $`t=t+1`$, repeat Step 2 and 3 until $`c^{t}`$ does not change anymore

