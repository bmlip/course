### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 0630d05e-6e1c-11f0-223b-adae877ed408
md"""
# Dynamic Models

  * **[1]** (##) Given the Markov property

```math
p(x*n|x*{n-1},x*{n-2},\ldots,x*1) = p(x*n|x*{n-1}) \tag{A1}
```

proof that, for any ``n``,

```math
p(x*n,x*{n-1},\ldots,x*{k+1},x*{k-1},\ldots,x*1|x*k) = p(x*n,x*{n-1},\ldots,x*{k+1}|x*k) \cdot p(x*{k-1},x*{k-2},\ldots,x*1|x*k) \tag{A2}
```

In other words, proof that, if the Markov property A1 holds, then, given the "present" (``x_k``), the "future" ``(x_n,x_{n-1},\ldots,x_{k+1})`` is *independent* of the "past" ``(x_{k-1},x_{k-2},\ldots,x_1)``.

> First, we rewrite A2 as


```math
\begin{align}
p(x*n,x*{n-1},\ldots,x*{k+1},x*{k-1},\ldots,x*1|x*k) &= \frac{p(x*n,x*{n-1},\ldots,x*1)}{p(x*k)} \\
&= \frac{p(x*n,x*{n-1},\ldots,x*{k+1}|x*k,\ldots,x*1) \cdot p(x*k,x*{k-1},\ldots,x*1)}{p(x*k)} \\
&= p(x*n,x*{n-1},\ldots,x*{k+1}|x*k,\ldots,x*1) \cdot p(x*{k-1},\ldots,x*1|x_k) \tag{A3}
\end{align}
```

The first term in A3 can be simplified if A1 holds to

```math
\begin{align}
p(x*n,x*{n-1},\ldots,x*{k+1}|x*k,x*{k-1},\ldots,x*1) &= p(x*n|x*{n-1},x*{n-2},\ldots,x*1) \cdot p(x*{n-1}|x*{n-2},x*{n-3},\ldots,x*1) \cdots \\
&\quad \cdots p(x*{k+1}|x*{k},x*{k-2},\ldots,x*1) \\
&= p(x*n|x*{n-1},x*{n-2},\ldots,x*k) \cdot p(x*{n-1}|x*{n-2},x*{n-3},\ldots,x*k) \cdots \\
&\quad \cdots p(x*{k+1}|x*{k}) \\
&= p(x*n,x*{n-1},\ldots,x*{k+1}|x*k) \tag{A4}
\end{align}
```

Substitution of A4 into A3 leads to A2. QED.

  * **[2]** (#)      (a) What's the difference between a hidden Markov model and a linear Dynamical system?    

    > HMM has binary-valued (on-off) states, where the LDS has continuously valued states.


    (b) For the same number of state variables, which of these two models has a larger memory capacity, and why?     

    > The latter holds more capacity because, eg, a 16-bit representation of a continuously-valued variable holds ``2^{16}`` different states.
  * **[3]** (#)

(a) What is the 1st-order Markov assumption?       (b) Derive the joint probability distribution ``p(x_{1:T},z_{0:T})`` (where ``x_t`` and ``z_t`` are observed and latent variables respectively) for the state-space model with transition and observation models ``p(z_t|z_{t-1})`` and ``p(x_t|z_t)``.       (c) What is a Hidden Markov Model (HMM)?        (d) What is a Linear Dynamical System (LDS)?       (e) What is a Kalman Filter?       (f) How does the Kalman Filter relate to the LDS?        (g) Explain the popularity of Kalman filtering and HMMs?        (h) How relates a HMM to a GMM? 

> (a) An auto-regressive model is first-order Markov if


```math
p(x_t|x_{t-1},x_{t-2},\ldots,x_1) = p(x_t|x_{t-1})\,.
```

> (b)


```math
p(x_{1:T},z_{0:T}) = p(z_0)\prod_{t=1}^Tp(z_t|z_{t-1}) \prod_{t=1}^T p(x_t|z_t)
```

> (c)  A HMM is a state-space model (as described in (b)) where the latent variable ``z_t`` is discretely valued. Iow, the HMM has hidden clusters.             (d)  An LDS is a state-space model (also described by the eq in (b)), but now the latent variable ``z_t`` is continuously valued.      (e) A Kalman filter is a recursive solution to the inference problem ``p(z_t|x_t,x_{t-1},\dots,x_1)``, based on a state estimate at the previous time step ``p(z_{t-1}|x_{t-1},x_{t-2},\dots,x_1)``  and a new observation ``x_t``. Basically, it's a recursive filter that updates the optimal Bayesian estimate of the current state ``z_t`` based on all past observations ``x_t,x_{t-1},\dots,x_1``.      (f) The LDS describes a (generative) *model*. The Kalman filter does not describe a model, but rather describes an *inference task* on the LDS model.            (g) The LDS and HMM models are both quite general and flexible generative probabilistic models for time series. There exists very efficient algorithms for executing the latent state inference tasks (Kalman filter for LDS and there is a similar algorithm for the HMM). That makes these models flexible and practical. Hence the popularity of these models.             (h) An HMM can be interpreted as a Gaussian-Mixture-model-over-time.


"""

# ╔═╡ Cell order:
# ╟─0630d05e-6e1c-11f0-223b-adae877ed408
