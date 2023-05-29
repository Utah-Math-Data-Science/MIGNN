# Monotone Implicit Graph Neural Networks

Monotone operator theory provides a computationally efficient expressive neural network capable of learning long range dependencies while maintaining peak memory efficiency.
The model is simplified to <a class="anchor" id="eq1" href="#eq1"></a>

$$\begin{equation} Z := \mathbb{A}\mathbb{B}Z \end{equation}$$

where $\mathbb{A}=\mathbb{J}_{\mathbb{C}}=\text{prox}^\alpha_{\partial f}$ and $\mathbb{B}=\text{agg}_W(Z,S)+b(X,S)$.
Written in full we can express the model as

$$\begin{equation} Z := \mathbb{J}_{\mathbb{A}}\left(\text{agg}_W(Z,S) + b(X,S)\right) \end{equation} $$

The input features are given by $X$,
and the edge information is contained in the input $S$. The learned latent features are given by $Z$.

## Installing Dependencies

Best install using `Dockerfile.mignn`, quick install with `requirements.txt`.

## Retrieving Datasets

The `snap_amz` dataset can be found [here](https://drive.google.com/file/d/1cc6ViFbrMu0ws__i2-OLoBVRhM3BV3I2/view?usp=sharing).

The CGS dataset is provided by the [CGS repository](https://github.com/Junyoungpark/CGS).

All other datasets are downloaded at runtime using the [PyG library](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html).

## Running Numerical Simulations

**Benchmark tasks** can be run using
```
CUDA_VISIBLE_DEVICES=<device> python3 /root/workspace/MIGNN/tasks/<task>.py
```

**Porenet task** can be run by placing the cgs_module into the top level [CGS codebase](https://github.com/Junyoungpark/CGS).

## Parameterizations of $W$

To assure fixed point convergence several parametrizations of $W$ are possible and discussed in the MIGNN [paper](https://openreview.net/forum?id=IajGRJuM7D3). An optionally learned parameter $\mu$ may be used to constrain the eigenvalues of $W$.

| **Flag** | **Parametrization** | **MIGNN** |
| --- | --- | --- |
| `cayley`| $\sigma(\mu)(I+B)(I-B)^{-1} D$ | :white_check_mark: |
| `expm`| $\sigma(\mu)\bf{B\exp(B^{-1}C})$ | :white_check_mark: |
| `frob`| $\sigma(\mu)\frac{BB^T}{\lVert BB^T \rVert_F +\varepsilon}$ | :white_check_mark: |
| `proj`| $\lVert W\rVert_F < \lambda_{pf}(A)^{-1}$ | :white_check_mark: |
| `symm`| $\frac{1-e^{\mu}}{2}I-BB^T$ | :white_check_mark: |
| `skew`| $\frac{1-e^{\mu}}{2}I-BB^T-C+C^T$ | :white_check_mark: |


## Accelerated Fixed Point Convergence

Each of the fixed point solving methods can be found in `agg._deq` several of these are operator splitting methods (OSM). Operator splitting methods require the [residual operator](#approximations-of-the-residual-operator). In addition several of the methods can be called with further acceleration schemes.

| **Class** | **OSM** | **Accelerated** |
| --- | --- | --- |
|  `DouglasRachford` | :white_check_mark: | :x: |
|  `DouglasRachfordAnderson` | :white_check_mark: | :white_check_mark: |
|  `DouglasRachfordHalpern` | :white_check_mark: | :white_check_mark: |
| | | 
|  `ForwardBackward` | :white_check_mark: | :x: |
|  `ForwardBackwardAnderson` | :white_check_mark: | :white_check_mark: |
| | | 
|  `PeacemanRachford` | :white_check_mark: | :x: |
|  `PeacemanRachfordAnderson` | :white_check_mark: | :white_check_mark: |
| | | 
|  `PowerMethod` | :x: | :x: |
|  `PowerMethodAnderson` | :x: | :white_check_mark: |


## Approximations of the Residual Operator

The operator splitting methods and their accelerated counterparts require the residual operator 

$$V=(I+\alpha (I-A^T\otimes W))^{-1}.$$

If the graph is very large this direct inverse calculation will be exceptionally expensive. Alternative methods for reducing the order of the model given in `agg._conv` are depicted below.


| **Method** | **Flag** | **Numerical Scheme** | **Complexity** |
| --- | --- | --- | --- |
| Direct Inverse | `direct` | $(I+\alpha (I-A^T\otimes W))^{-1}$ | $\mathcal{O}(m^3n^3)$ |
| Eigen Decomposition | `eig` | $Q_{A^T} Q_S^T(G\circ (Q_{A^T}(\cdot) Q_S^T))Q_S$ | $\mathcal{O}(\text{max}[m^3,n^3])$ |
| Neumann Expansion | `neumann-k` | $I+W(\cdot)A+\dots+W^k(\cdot)A^k$ | $\mathcal{O}(\text{max}[km^2,kn^2])$ |


## Citation

```
@misc{
baker2023stable,
title={Stable, Efficient, and Flexible Monotone Operator Implicit Graph Neural Networks},
author={Justin Baker and Qingsong Wang and Bao Wang},
year={2023},
url={https://openreview.net/forum?id=IajGRJuM7D3}
}