### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 8fcf38a8-6e1b-11f0-2eb9-efa086c9bc9e
md"""
# Regression

  * **[1]** (#) (a) Write down the generative model for Bayesian linear ordinary regression (i.e., write the likelihood and prior).       (b) State the inference task for the weight parameter in the model.      (c) Why do we call this problem linear?
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

  * **[3]** (###) Show that the variance of the predictive distribution for linear regression decreases as more data becomes available.

  * **[4]** (#) Assume a given data set ``D=\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\}`` with ``x \in \mathbb{R}^M`` and ``y \in \mathbb{R}``. We propose a model given by the following data generating distribution and weight prior functions:

```math
\begin{equation*} p(y_n|x_n,w)\cdot p(w)\,. \end{equation*}
```

(a) Write down Bayes rule for generating the posterior ``p(w|D)`` from a prior and likelihood.       (b) Work out how to compute a distribution for the predicted value ``y_\bullet``, given a new input ``x_\bullet``.   

  * **[5]** (#) In the class we use the following prior for the weights:

```math
\begin{equation*}
p(w|\alpha) = \mathcal{N}\left(w | 0, \alpha^{-1} I \right)
\end{equation*}
```

(a) Give some considerations for choosing a Gaussian prior for the weights.       (b) We could have chosen a prior with full (not diagonal) covariance matrix ``p(w|\alpha) = \mathcal{N}\left(w | 0, \Sigma \right)``. Would that be better? Give your thoughts on that issue.             (c) Generally we choose ``\alpha`` as a small positive number. Give your thoughts on that choice as opposed to choosing a large positive value. How about choosing a negative value for ``\alpha``?

  * **[6]** Consider an IID data set ``D=\{(x_1,y_1),\ldots,(x_N,y_N)\}``. We will model this data set by a model

```math
y_n =\theta^T  f(x_n) + e_n\,,
```

where ``f(x_n)`` is an ``M``-dimensional feature vector of input ``x_n``; ``y_n`` is a scalar output and ``e_n \sim \mathcal{N}(0,\sigma^2)``.                  (a) Rewrite the model in matrix form by lumping input features in a matrix ``F=[f(x_1),\ldots,f(x_N)]^T``, outputs and noise in the vectors ``y=[y_1,\ldots,y_N]^T`` and ``e=[e_1,\ldots,e_N]^T``, respectively.    

(b) Now derive an expression for the log-likelihood ``\log p(y|\,F,\theta,\sigma^2)``. 

(c) Proof that the maximum likelihood estimate for the parameters is given by 

```math
\hat\theta_{\text{ml}} = (F^TF)^{-1}F^Ty
```

(d) What is the predicted output value ``y_\bullet``, given an observation ``x_\bullet`` and the maximum likelihood parameters ``\hat \theta_{\text{ml}}``. Work this expression out in terms of ``F``, ``y`` and ``f(x_\bullet)``.      

(e) Suppose that, before the data set ``D`` was observed, we had reason to assume a prior distribution ``p(\theta)=\mathcal{N}(0,\sigma_0^2)``. Derive the Maximum a posteriori (MAP) estimate ``\hat \theta_{\text{map}}``.(hint: work this out in the ``\log`` domain.)   



"""

# ╔═╡ Cell order:
# ╟─8fcf38a8-6e1b-11f0-2eb9-efa086c9bc9e
