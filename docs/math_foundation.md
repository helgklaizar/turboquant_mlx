# Mathematical Foundation of TurboQuant

The KV sequence logic operates upon 2 independent variables essential to inner dot-products typical in generative Transformers. The query $Q$ interacts symmetrically across the previously processed $K$ states. However, directly quantizing $K$ yields a substantial inner product matrix bias linearly scaling with bit depths constraints.

## 1. PolarQuant (Quantized Spherical Variables)
Instead of quantizing values across standard Cartesian grids which introduce unbounded absolute errors on extreme vectors:
1. Vectors naturally reshape from Cartesian into Polar.
2. Angular vectors distribute evenly bounded via $- \pi$ to $\pi$.
3. Radii scalar dimensions are restricted and grouped without implicit distribution overheads. 
This bypasses standard normalizations needed in INT4 configurations.

## 2. Quantized Johnson-Lindenstrauss (QJL) Transform
The core issue with naive 1-bit / 3-bit matrices: the Mean-Squared Error (MSE) is minimized, but the inner product contains a consistent scalar bias due to spatial drifting.
1. The difference $x - x_{approx} = x_{residual}$.
2. Project $x_{residual}$ into an explicitly highly randomized plane $P_{rand}$.
3. Discretize $P_{rand}$ strictly to $+1 / -1$ variables.
4. During evaluation, estimate the exact inner product by multiplying $Q$ along the highly sparse scalar binary residual matrix.

This theorem ensures unbiased estimations, essentially pushing low-bit quantized caches to function strictly equivalently alongside FP16 caching networks absent any hallucination degradation.
