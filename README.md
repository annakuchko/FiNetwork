# FiNetwork
### Portfolio optimisation based on network cluster analysis
This library was developed as a part of bachelor's thesis project (presentation can be found [here](https://github.com/annakuchko/FiNetwork/blob/main/Diploma_eng.pdf)).

[Meduim article](https://medium.com/@anna.kuchko/finetwork-dynamic-portfolio-optimisation-based-on-network-cluster-analysis-5abacfc356e4).

## Installation
```
!pip install git+https://github.com/annakuchko/FiNetwork.git#egg=finetwork
```

## [Documentation & Examples](https://github.com/annakuchko/FiNetwork/blob/main/examples/finetwork_example_sp500.md)


## Main functionality

Select "selection-investment" horizon based on Wyckoff's stages of market cycle using dedicated criterion values:
- "Trading day" criterion
- "Amplitude" criterion
- "and" criterion
- "or" criterion

Build network using one of the provided distance metrics:
- Ultrametric Distance based on Pearson Correlation
- Theil Index based entropic distance
- Atkinson Index based entropic distance

Perform cluster analysis choosing from one of the following clustering algorithms:
- Spectral
- Kmeans
- SpectralKmeans
- Kmedoids
- GaussianMixture
- SpectralGaussianMixture
- Hierarchical

(Optimal) number of clusters is selected based on eigengap.

Check the quality of clustering algorithm based on one of the following metrics:
- Calinski-Harabasz Index
- Silhouette Score
- Davies Bouldin Score

Select optimal (diversified) stocks portfolio and validate the performance using back-testing. Calculate portfolio performance metrics:
- volatility
- sharpe_ratio
- sortino_ratio
- max_drawdown
- calmar_ratio

You can also select portfolio based on industries partition or randomly.

Plot [animations](https://github.com/annakuchko/FiNetwork/blob/main/imgs/pearson_Spectral_clustering_partition.gif) of changing market structure across different stages of market cycle.
