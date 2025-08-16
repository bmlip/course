### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ da8d7fae-6e1b-11f0-38d3-75f571de1b36
md"""
# Factor Graphs

  * **[1]** Consider the following state-space model:

```math
\begin{align*}
z_k &= A z_{k-1} + w_k \\
x_k &= C z_k + v_k 
\end{align*}
```

where ``k=1,2,\ldots,n`` is the time step counter; ``z_k`` is  an *unobserved* state sequence; ``x_k`` is an *observed* sequence; ``w_k \sim \mathcal{N}(0,\Sigma_w)`` and ``v_k \sim \mathcal{N}(0,\Sigma_v)`` are (unobserved) state and observation noise sequences respectively; ``z_0 \sim \mathcal{N}(0,\Sigma_0)`` is the initial state and ``A``, ``C``, ``\Sigma_v``,``\Sigma_w`` and ``\Sigma_0`` are known parameters. The Forney-style factor graph (FFG) for one time step is depicted here:

<img src="./i/ffg-5SSB0-exam-Kalman-filter.png" style="width:500px;">

(a) Rewrite the state-space equations as a set of conditional probability distributions.                 

```math
\begin{align*}
 p(z_k|z_{k-1},A,\Sigma_w) &= \ldots \\
 p(x_k|z_k,C,\Sigma_v) &= \ldots \\
 p(z_0|\Sigma_0) &= \ldots
\end{align*}
```

> This is a linear system with only Gaussian source signals (``w_k`` and ``v_k``), hence the distributions for ``z_k`` and ``x_k`` will also be Gaussian. As a result, we only need to compute the mean and covariance matrix. We begin with the mean for ``p(z_k|z_{k-1},A,\Sigma_w)``:


```math
\begin{align*}
  E[z_k|z_{k-1},A,\Sigma_w] &= E[A z_{k-1} + w_k|z_{k-1},A,\Sigma_w] \\
  &= E[A z_{k-1}|z_{k-1},A] + E[w_k|\Sigma_w] \\
  &= A z_{k-1} + 0
  \end{align*}
```

> And now the variance:


```math
\begin{align*}
  V[z_k|z_{k-1},A,\Sigma_w] &= E[(z_k - E[z_k])(z_k-E[z_k])^T \,|\,z_{k-1},A,\Sigma_w ] \\ &= E[(\overbrace{A z_{k-1} + w_k}^{z_k} - \overbrace{A z_{k-1}}^{E[z_k]})(A z_{k-1} + w_k-A z_{k-1})^T|z_{k-1},A,\Sigma_w] \\
  &= E[w_k w_k^T|\Sigma_w] \\
  &= \Sigma_w
  \end{align*}
```

> You can execute similar computations for the other distributions, leading to


```math
\begin{align*}
 p(z_k|z_{k-1},A,\Sigma_w) &= \mathcal{N}(z_k|A z_{k-1},\Sigma_w) \\
 p(x_k|z_k,C,\Sigma_v) &= \mathcal{N}(x_k|C z_k,\Sigma_v) \\
  p(z_0|\Sigma_0) &= \mathcal{N}(z_0|0,\Sigma_0)
\end{align*}
```

(b) Define ``z^n \triangleq (z_0,z_1,\ldots,z_n)``, ``x^n \triangleq (x_1,\ldots,x_n)`` and ``\theta=\{A,C,\Sigma_w,\Sigma_v\}``. Now write out the generative model ``p(x^n,z^n|\theta)`` as a product of factors.     

```math
\begin{align*}
p(x^n,z^n|\theta) &= p(z_0|\Sigma_0) \prod_{k=1}^n p(x_k|z_k,C,\Sigma_v) \,p(z_k|z_{k-1},A,\Sigma_w) \\
  &= \mathcal{N}(z_0|0,\Sigma_0) \prod_{k=1}^n  \mathcal{N}(x_k|C z_k,\Sigma_v) \,\mathcal{N}(z_k|A z_{k-1},\Sigma_w)
\end{align*}
```

(c) We are interested in estimating ``z_k`` from a given estimate for ``z_{k-1}`` and the current observation ``x_k``, i.e., we are interested in computing ``p(z_k|z_{k-1},x_k,\theta)``. Can ``p(z_k|z_{k-1},x_k,\theta)`` be expressed as a Gaussian distribution? Explain why or why not in one sentence.    

> Yes, since the generative model ``p(x^n,z^n|\theta)`` is (one big) Gaussian.


(d) Copy the graph onto your exam paper and draw the message passing schedule for computing ``p(z_k|z_{k-1},x_k,\theta)`` by drawing arrows in the factor graph. Indicate the order of the messages by assigning numbers to the arrows.      <img src="./i/ffg-5SSB0-exam-Kalman-filter-wMessages-wUncertainSigmaV.png" style="width:500px;">

> Some permutations of this order are also possible. The most important thing here is that you recognize the tree with ``Z_k`` as a root of the tree and pass messages from the terminals (e.g., ``Z_{k-1}``, ``X_k``, etc.) towards the root.


(e) Now assume that our belief about parameter ``\Sigma_v`` is instead given by a distribution ``p(\Sigma_v)`` (rather than a known value). Adapt the factor graph drawing of the previous answer to reflects our belief about ``\Sigma_v``.      

> See drawing in previous answer.


"""

# ╔═╡ da8da4e0-6e1b-11f0-01f4-49c35fc2d3cf
md"""
  * **[2]** Consider an addition node

```math
f_+(x,y,z) = \delta(z-x-y)
```

![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/./i/ffg-addition-node.png?raw=true)

(a) Derive an expression for the outgoing message ``\overrightarrow{\mu}_{Z}(z)`` in terms of the incoming messages ``\overrightarrow{\mu}_{X}(\cdot)`` and ``\overrightarrow{\mu}_{Y}(\cdot)``.   

> We use the sum-product rule to compute


```math
\begin{align*}
  \overrightarrow{\mu}_{Z}(z) &= \iint  \overrightarrow{\mu}_{X}(x) \overrightarrow{\mu}_{Y}(y) \,\delta(z-x-y) \,\mathrm{d}x \mathrm{d}y \\
   &=  \int  \overrightarrow{\mu}_{X}(x) \overrightarrow{\mu}_{Y}(z-x) \,\mathrm{d}x \,, 
  \end{align*}
```

> i.e., ``\overrightarrow{\mu}_{Z}`` is the convolution of the messages ``\overrightarrow{\mu}_{X}`` and ``\overrightarrow{\mu}_{Y}``.


(b) Now assume that both incoming messages are Gaussian, namely ``\overrightarrow{\mu}_{X}(x) \sim \mathcal{N}(\overrightarrow{m}_X,\overrightarrow{V}_X)`` and ``\overrightarrow{\mu}_{Y}(y) \sim \mathcal{N}(\overrightarrow{m}_Y,\overrightarrow{V}_Y)``. Evaluate the outgoing message ``\overrightarrow{\mu}_{Z}(z)``.  You will need the [multiplication rule for Gaussians](https://github.com/bertdv/BMLIP/raw/master/lessons/notebooks/files/Roweis-1999-gaussian-identities.pdf?dl=0).   

> For Gaussian incoming messages, these update rules evaluate to ``\overrightarrow{\mu}_{Z}(z) \sim \mathcal{N}(\overrightarrow{m}_Z,\overrightarrow{V}_Z)`` with


```math
\begin{align*}
  \overrightarrow{m}_Z &= \overrightarrow{m}_X + \overrightarrow{m}_Y \\
  \overrightarrow{V}_z &= \overrightarrow{V}_X + \overrightarrow{V}_Y \,.
\end{align*}
```

  * (c) For the same summation node, work out the SP update rule for the backward message ``\overleftarrow{\mu}_{X}(x)`` as a function of ``\overrightarrow{\mu}_{Y}(y)`` and  ``\overleftarrow{\mu}_{Z}(z)``. And further refine the answer for Gaussian messages.

```math
\begin{align*}
  \overleftarrow{\mu}_{X}(x) &= \iint  \overrightarrow{\mu}_{Y}(y) \overleftarrow{\mu}_{Z}(z) \,\delta(z-x-y) \,\mathrm{d}y \mathrm{d}z \\
   &=  \int  \overrightarrow{\mu}_{Y}(z-x) \overleftarrow{\mu}_{Z}(z) \,\mathrm{d}z  
  \end{align*}
```

> and now further with Gaussian messages,


```math
\begin{align*}
  \overleftarrow{\mu}_{X}(x) &= \int  \mathcal{N}(z-x | m_y,V_y)  \mathcal{N}(z | m_z,V_z)\,\mathrm{d}z  \qquad &&\text{(a)}\\
  &=  \int  \mathcal{N}(z | x+ m_y,V_y)  \mathcal{N}(z | m_z,V_z)\,\mathrm{d}z  \qquad &&\text{(b)}\\
  &=  \int  \mathcal{N}(x+m_y | m_z,V_y+V_z)  \mathcal{N}(z | \cdot,\cdot)\,\mathrm{d}z \qquad &&\text{(c)} \\
  &= \mathcal{N}(x | m_z-m_y, V_y+V_z) \qquad &&\text{(d)}
\end{align*}
```

> where going from (b) to (c) we used the formula for multiplication of Gaussians:


```math
\mathcal{N}(z|m_a,V_a) \mathcal{N}(z|m_b,V_b) = \mathcal{N}(m_a|m_b,V_a+V_b) \mathcal{N}(z|\cdot,\cdot)
```

> where the dots stand for expressions that are not relevant here, since the Gaussian for ``z`` is marginalized away.


"""

# ╔═╡ Cell order:
# ╟─da8d7fae-6e1b-11f0-38d3-75f571de1b36
# ╟─da8da4e0-6e1b-11f0-01f4-49c35fc2d3cf
