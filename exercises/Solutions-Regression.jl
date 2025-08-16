### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 946903d8-6e1b-11f0-0b3c-5f15b6fa5b09
md"""
# Regression

  * **[1]** (#) (a) Write down the generative model for Bayesian linear ordinary regression (i.e., write the likelihood and prior).       (b) State the inference task for the weight parameter in the model.      (c) Why do we call this problem linear?

> (a)


```math
\begin{align*}
\text{likelihood: } p(y_n|x_n,w) &= \mathcal{N}(y_n|w^T\phi(x_n),\beta^{-1}) \\
\text{prior: } p(w|\alpha) &= \mathcal{N}(w|0,\alpha^{-1}I)
\end{align*}
```

> (b) The inference task is to compute


```math
p(w|D) = \frac{p(D|w)p(w)}{p(D)}
```

> (c) The model is linear with respect to ``w``, which is the reason we call it linear.


  * **[2]** (##) Consider a linear regression problem

```math
\begin{align*}
p(y\,|\,\mathbf{X},w,\beta) &= \mathcal{N}(y\,|\,\mathbf{X} w,\beta^{-1} \mathbf{I}) \\
  &= \prod_n \mathcal{N}(y_n\,|\,w^T x_n,\beta^{-1})
\end{align*}
```

with ``y, X`` and ``w`` as defined in the notebook.         (a) Work out the maximum likelihood solution for linear regression by solving

```math
\nabla_{w} \log p(y|X,w) = 0 \,.
```

(b) Work out the MAP solution. How does it relate to the ML solution?

> (a) The gradient of the log-likelihood is


```math
\begin{equation*} \nabla_{w} \log p(y|X,w) = X^T(y-Xw) \end{equation*}
```

> Setting the derivation to zero leads to


```math
\begin{equation*}w_{ML} = (X^TX)^{-1}X^Ty \end{equation*}
```

> (b) We now add a prior ``w \sim \mathcal{N}(0,\alpha^{-1})``, and a similar derivation leads to


```math
\begin{equation*}\nabla_{w} \log p(y,w|X) =-\beta X^T(y-Xw)+\alpha w \end{equation*}
```

> Setting the derivation to zero leads to


```math
\begin{equation*} w_{MAP} = (X^TX+\frac{\alpha}{\beta}I)^{-1}X^Ty  \end{equation*}
```

> The MAP solution weighs both the prior and likelihood. If ``\frac{\alpha}{\beta}`` is close to zero (if the prior is uninformative), then the ML solution and MAP solutions are close to each other.


  * **[3]** (###) Show that the variance of the predictive distribution for linear regression decreases as more data becomes available.

> The variance of the predictive distribution is given by


```math
\begin{align*} \sigma_{N+1}^2(x) &= 1/\beta + \phi(x)^TS_{N+1}\phi(x) \\
S_{N+1} &= (S_N^{-1} + \beta\phi_{N+1}\phi_{N+1}^T)^{-1} \\
&= S_N - \frac{\beta S_N\phi_{N+1}\phi_{N+1}^TS_N}{1+\beta\phi_{N+1}^TS_N\phi_{N+1}}
\end{align*}
```

> where in the last equality, we applied [Woodbury's matrix identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity), which is also listed in [Sam Roweis' matrix notes, eq. 10](https://github.com/bertdv/BMLIP/raw/master/lessons/notebooks/files/Roweis-1999-matrix-identities.pdf?dl=0). Using the recursive relation for ``S_N`` we can write the variance for the next observation as


```math
\begin{align*}
\sigma_{N+1}^2(x) &= \sigma_N^2(x) - \frac{\beta\phi(x)^TS_N\phi_{N+1}\phi_{N+1}^TS_N\phi(x)}{1+\beta\phi_{N+1}^TS_N\phi_{N+1}}.
\end{align*}
```

> Because ``S_N`` is positive definite, the numerator and the denominator of the second term wil be non-negative, hence


```math
\sigma_N^2(x) \geqslant \sigma_{N+1}^2(x)
```

. This shows that the predictive variance decrease as more data becomes available.

  * **[4]** (#) Assume a given data set ``D=\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\}`` with ``x \in \mathbb{R}^M`` and ``y \in \mathbb{R}``. We propose a model given by the following data generating distribution and weight prior functions:

```math
\begin{equation*} p(y_n|x_n,w)\cdot p(w)\,. \end{equation*}
```

(a) Write down Bayes rule for generating the posterior ``p(w|D)`` from a prior and likelihood.       (b) Work out how to compute a distribution for the predicted value ``y_\bullet``, given a new input ``x_\bullet``.    

> (a)


```math
 p(w|D) = \frac{p(w) \prod_{n=1}^N p(y_n|x_n,w)}{\int p(w) \prod_{n=1}^N p(y_n|x_n,w)\mathrm{d}w}
```

> (b)


```math
p(y_\bullet|x_\bullet,D) = \int p(y_\bullet|x_\bullet,w) p(w|D) \mathrm{d}w
```

  * **[5]** (#) In the class we use the following prior for the weights:

```math
\begin{equation*}
p(w|\alpha) = \mathcal{N}\left(w | 0, \alpha^{-1} I \right)
\end{equation*}
```

(a) Give some considerations for choosing a Gaussian prior for the weights.       (b) We could have chosen a prior with full (not diagonal) covariance matrix ``p(w|\alpha) = \mathcal{N}\left(w | 0, \Sigma \right)``. Would that be better? Give your thoughts on that issue.             (c) Generally we choose ``\alpha`` as a small positive number. Give your thoughts on that choice as opposed to choosing a large positive value. How about choosing a negative value for ``\alpha``?

> (a) These considerations can be both computational (eg, Gaussian prior times Gaussian likelihood leads to a Gaussian posterior) or based on available information  (eg, among all distributions with the same variance, the Gaussian distribution has the largest entropy. Roughly this means that the Gaussian makes the least amount of assumptions across all distributions with the same variance).      (b) If you have no prior information about co-variances, why make that assumption? If you do have some prior information, eg based on the physical process, then by all means feel free to add those constraints to the prior. Note that the posterior variance is given by ``S_N = \left( \alpha \mathbf{I} + \beta \mathbf{X}^T\mathbf{X}\right)^{-1}``. Importantly, the term ``\alpha \mathbf{I}`` for small ``\alpha`` makes sure that the matrix is invertible, even for zero observations.      (c) As you can see from the posterior variance (see answer to (b)), for smaller values of ``\alpha``, the data term ``\mathbf{X}^T\mathbf{X}`` gets to play a role after fewer observations. Hence, if you have little prior information, it's better to choose a small value for ``\alpha``.


  * **[6]** Consider an IID data set ``D=\{(x_1,y_1),\ldots,(x_N,y_N)\}``. We will model this data set by a model

```math
y_n =\theta^T  f(x_n) + e_n\,,
```

where ``f(x_n)`` is an ``M``-dimensional feature vector of input ``x_n``; ``y_n`` is a scalar output and ``e_n \sim \mathcal{N}(0,\sigma^2)``.                  (a) Rewrite the model in matrix form by lumping input features in a matrix ``F=[f(x_1),\ldots,f(x_N)]^T``, outputs and noise in the vectors ``y=[y_1,\ldots,y_N]^T`` and ``e=[e_1,\ldots,e_N]^T``, respectively.     ``y = F\theta + e``

(b) Now derive an expression for the log-likelihood ``\log p(y|\,F,\theta,\sigma^2)``. 

```math
\begin{align*}
 \log p(D|\theta,\sigma^2) &= \log \mathcal{N}(y|\,F\theta ,\sigma^2)\\
    &\propto  -\frac{1}{2\sigma^2}\left( {y - F\theta } \right)^T \left( {y - F\theta } \right)
\end{align*}
```

(c) Proof that the maximum likelihood estimate for the parameters is given by

```math
\hat\theta_{\text{ml}} = (F^T F)^{-1}F^Ty \,.
```

> Taking the derivative to ``\theta``


```math
\nabla_\theta \log p(D|\theta) = \frac{1}{\sigma^2} F^T(y-F\theta)
```

> Set derivative to zero for maximum likelihood estimate


```math
 \hat\theta_{\text{ml}} = (F^TF)^{-1}F^Ty
```

(d) What is the predicted output value ``y_\bullet``, given an observation ``x_\bullet`` and the maximum likelihood parameters ``\hat \theta_{\text{ml}}``. Work this expression out in terms of ``F``, ``y`` and ``f(x_\bullet)``.      

> Prediction of new data point: ``\hat y_\bullet = \hat \theta^T f(x_\bullet) = \left((F^TF)^{-1}F^Ty\right)^T  f(x_\bullet)``


(e) Suppose that, before the data set ``D`` was observed, we had reason to assume a prior distribution ``p(\theta)=\mathcal{N}(0,\sigma_0^2)``. Derive the Maximum a posteriori (MAP) estimate ``\hat \theta_{\text{map}}``.(hint: work this out in the ``\log`` domain.)                

```math
\begin{align*}
\log p(\theta|D) &\propto \log p(D|\theta) p(\theta) \\
    &\propto  -\frac{1}{2\sigma^2}\left( {y - F\theta } \right)^T \left( {y - F\theta } \right) - \frac{1}{2 \sigma_0^2}\theta^T \theta
\end{align*}
```

> Derivative ``\nabla_\theta \log p(\theta|D) = (1/\sigma^2)F^T(y-F\theta) - (1/ \sigma_0^2) \theta``      Set derivative to zero for MAP estimate leads to


```math
\hat\theta_{\text{map}} = \left(F^T F + \frac{\sigma^2}{\sigma_0^2} I\right)^{-1}F^Ty
```



"""

# ╔═╡ Cell order:
# ╟─946903d8-6e1b-11f0-0b3c-5f15b6fa5b09
