### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 906a99a6-6e1b-11f0-123f-03237db4fc21
md"""
# Continuous Data and the Gaussian Distribution

  * **[1]** (##) We are given an IID data set ``D = \{x_1,x_2,\ldots,x_N\}``, where ``x_n \in \mathbb{R}^M``. Let's assume that the data were drawn from a multivariate Gaussian (MVG),

```math
\begin{align*}
p(x_n|\theta) = \mathcal{N}(x_n|\,\mu,\Sigma) = \frac{1}{\sqrt{(2 \pi)^{M} |\Sigma|}} \exp\left\{-\frac{1}{2}(x_n-\mu)^T
\Sigma^{-1} (x_n-\mu) \right\}
\end{align*}
```

(a) Derive the log-likelihood of the parameters for these data.          (b) Derive the maximum likelihood estimates for the mean ``\mu`` and variance ``\Sigma`` by setting the derivative of the log-likelihood to zero.

  * **[2]** (#) Shortly explain why the Gaussian distribution is often preferred as a prior distribution over other distributions with the same support?
  * **[3]** (###) We make ``N`` IID observations ``D=\{x_1 \dots x_N\}`` and assume the following model

```math
\begin{aligned}
x_k &= A + \epsilon_k \\
A &\sim \mathcal{N}(m_A,v_A) \\
\epsilon_k &\sim \mathcal{N}(0,\sigma^2) \,.
\end{aligned}
```

We assume that ``\sigma`` has a known value and are interested in deriving an estimator for ``A`` .      (a) Derive the Bayesian (posterior) estimate ``p(A|D)``.        (b) (##) Derive the Maximum Likelihood estimate for ``A``.         (c) Derive the MAP estimates for ``A``.       (d) Now assume that we do not know the variance of the noise term? Describe the procedure for Bayesian estimation of both ``A`` and ``\sigma^2`` (No need to fully work out to closed-form estimates). 

  * **[4]** (##) Proof that a linear transformation ``z=Ax+b`` of a Gaussian variable ``\mathcal{N}(x|\mu,\Sigma)`` is Gaussian distributed as

```math
p(z) = \mathcal{N} \left(z \,|\, A\mu+b, A\Sigma A^T \right) 
```

  * **[5]** (#) Given independent variables

```math
x \sim \mathcal{N}(\mu_x,\sigma_x^2)
```

and ``y \sim \mathcal{N}(\mu_y,\sigma_y^2)``, what is the PDF for ``z = A\cdot(x -y) + b``?    

  * **[6]** (###) Compute

\begin{equation*}         \int_{-\infty}^{\infty} \exp(-x^2)\mathrm{d}x \,.     \end{equation*}

  * **[7]** (##) Show that the system

```math
\begin{align*}
p(x\,|\,\theta) &= \mathcal{N}(x\,|\,\theta,\sigma^2) \\
p(\theta) &= \mathcal{N}(\theta\,|\,\mu_0,\sigma_0^2)
\end{align*}
```

can be written as

```math
p(z) = p\left(\begin{bmatrix} x \\ \theta \end{bmatrix}\right) = \mathcal{N} \left( \begin{bmatrix} x\\ 
  \theta  \end{bmatrix} 
  \,\left|\, \begin{bmatrix} \mu_0\\ 
  \mu_0\end{bmatrix}, 
         \begin{bmatrix} \sigma_0^2+\sigma^2  & \sigma_0^2\\ 
         \sigma_0^2 &\sigma_0^2 
  \end{bmatrix} 
  \right. \right)
```



"""

# ╔═╡ Cell order:
# ╟─906a99a6-6e1b-11f0-123f-03237db4fc21
