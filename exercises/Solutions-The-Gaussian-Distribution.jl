### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 94a4fe74-6e1b-11f0-024b-c576b489a203
md"""
# Continuous Data and the Gaussian Distribution

  * **[1]** (##) We are given an IID data set ``D = \{x_1,x_2,\ldots,x_N\}``, where ``x_n \in \mathbb{R}^M``. Let's assume that the data were drawn from a multivariate Gaussian (MVG),

```math
\begin{align*}
p(x_n|\theta) = \mathcal{N}(x_n|\,\mu,\Sigma) = \frac{1}{\sqrt{(2 \pi)^{M} |\Sigma|}} \exp\left\{-\frac{1}{2}(x_n-\mu)^T
\Sigma^{-1} (x_n-\mu) \right\}
\end{align*}
```

"""

# ╔═╡ 94a50a2c-6e1b-11f0-0cd0-019fa190b3d0
md"""
(a) Derive the log-likelihood of the parameters for these data.  

> (a) Let ``\theta ={\mu,\Sigma}``. Then the log-likelihood can be worked out as


```math
\begin{align*}
\log p(D|\theta) &= \log \prod_n p(x_n|\theta) \\
 &= \log \prod_n \mathcal{N}(x_n|\mu, \Sigma) \\
&= \log \prod_n (2\pi)^{-M/2} |\Sigma|^{-1/2} \exp\left\{ -\frac{1}{2}(x_n-\mu)^T \Sigma^{-1}(x_n-\mu)\right\} \\
&= \sum_n \left( \log (2\pi)^{-M/2} + \log  |\Sigma|^{-1/2} -\frac{1}{2}(x_n-\mu)^T \Sigma^{-1}(x_n-\mu)\right) \\
&\propto \frac{N}{2}\log  |\Sigma|^{-1} - \frac{1}{2}\sum_n (x_n-\mu)^T \Sigma^{-1}(x_n-\mu)
\end{align*}
```

"""

# ╔═╡ 94a5604c-6e1b-11f0-309f-113a8a787af9
md"""
(b) Derive the maximum likelihood estimates for the mean ``\mu`` and variance ``\Sigma`` by setting the derivative of the log-likelihood to zero.

> (b) First we take the derivative with respect to the mean.


```math
\begin{align*}
\nabla_{\mu} \log p(D|\theta) &\propto - \sum_n \nabla_{\mu} \left(x_n-\mu \right)^T\Sigma^{-1}\left(x_n-\mu \right)  \\
&= - \sum_n \nabla_{\mu} \left(-2 \mu^T\Sigma^{-1}x_n + \mu^T \Sigma^{-1}\mu \right) \\
&= - \sum_n \left(-2 \Sigma^{-1}x_n + 2\Sigma^{-1}\mu \right) \\
&= -2 \Sigma^{-1} \sum_n (x_n - \mu)
\end{align*}
```

> Setting the derivative to zeros leads to ``\hat{\mu} = \frac{1}{N}\sum_n x_n``.


The derivative with respect to covariance is a bit more involved. It's actually easier to compute this by taking the derivative to the precision:

```math
\begin{align*}
\nabla_{\Sigma^{-1}} \log p(D|\theta) &= \nabla_{\Sigma^{-1}} \left( \frac{N}{2} \log |\Sigma| ^{-1} -\frac{1}{2}\sum_n (x_n-\mu)^T
\Sigma^{-1} (x_n-\mu)\right)  \\
&= \nabla_{\Sigma^{-1}} \left( \frac{N}{2} \log |\Sigma| ^{-1} - \frac{1}{2}\sum_n \mathrm{Tr}\left[(x_n-\mu)
(x_n-\mu)^T \Sigma^{-1} \right]\right) \\
&=\frac{N}{2}\Sigma - \frac{1}{2}\sum_n (x_n-\mu)
(x_n-\mu)^T
\end{align*}
```

> Setting the derivative to zero leads to :\hat{\Sigma} = \frac{1}{N}\sum*n (x*n-\hat{\mu})


(x_n-\hat{\mu})^T$.

"""

# ╔═╡ 94a570e8-6e1b-11f0-193f-4ba36198fc2e
md"""
  * **[2]** (#) Shortly explain why the Gaussian distribution is often preferred as a prior distribution over other distributions with the same support?

> You can get this answer straight from the lession notebook. Aside from the computational advantages (operations on distributions tends to make them more Gaussian, and Gaussians tends to remain Gaussians in computational manipulations), the Gaussian distribution is also the maximum-entropy distribution among distributions that are defined over real numbers. This means that there is no distribution with the same variance that assumes less information about its argument.


"""

# ╔═╡ 94a5833a-6e1b-11f0-3b02-735fd0fbd170
md"""
  * **[3]** (###) We make ``N`` IID observations ``D=\{x_1 \dots x_N\}`` and assume the following model

```math
\begin{aligned}
x_k &= A + \epsilon_k \\
A &\sim \mathcal{N}(m_A,v_A) \\
\epsilon_k &\sim \mathcal{N}(0,\sigma^2) \,.
\end{aligned}
```

We assume that ``\sigma`` has a known value and are interested in deriving an estimator for ``A`` .

"""

# ╔═╡ 94a5b29c-6e1b-11f0-26cf-e1bec8ba04dd
md"""
(a) Derive the Bayesian (posterior) estimate ``p(A|D)``.   

> Since ``p(D|A) = \prod_k \mathcal{N}(x_k|A,\sigma^2)`` is a Gaussian likelihood and ``p(A)`` is a Gaussian prior, their multiplication is proportional to a Gaussian. We will work this out with the canonical parameterization of the Gaussian since it is easier to multiply Gaussians in that domain. This means the posterior ``p(A|D)`` is


```math
\begin{align*}
   p(A|D) &\propto p(A) p(D|A) \\
   &= \mathcal{N}(A|m_A,v_A) \prod_{k=1}^N \mathcal{N}(x_k|A,\sigma^2) \\
   &= \mathcal{N}(A|m_A,v_A) \prod_{k=1}^N \mathcal{N}(A|x_k,\sigma^2) \\
   &= \mathcal{N}_c\big(A \Bigm|\frac{m_A}{v_A},\frac{1}{v_A}\big)\prod_{k=1}^N \mathcal{N}_c\big(A\Bigm| \frac{x_k}{\sigma^2},\frac{1}{\sigma^2}\big) \\
       &\propto \mathcal{N}_c\big(A \Bigm| \frac{m_A}{v_A} + \frac{1}{\sigma^2} \sum_k x_k , \frac{1}{v_A} + \frac{N}{\sigma^2}  \big)      \,, 
  \end{align*}
```

> where we have made use of the fact that precision-weighted means and precisions add when multiplying Gaussians. In principle this description of the posterior completes the answer.


"""

# ╔═╡ 94a5bb68-6e1b-11f0-3f64-e5e4fbbf6942
md"""
(b) (##) Derive the Maximum Likelihood estimate for ``A``.

> The ML estimate can be found by


```math
\begin{align*}
  \nabla \log p(D|A) &=0\\
  \nabla \sum_k \log \mathcal{N}(x_k|A,\sigma^2) &= 0 \\
  \nabla \frac{-1}{2}\sum_k \frac{(x_k-A)^2}{\sigma^2} &=0\\
  \sum_k(x_k-A) &= 0 \\
  \Rightarrow \hat{A}_{ML} = \frac{1}{N}\sum_{k=1}^N x_k
\end{align*}
```

"""

# ╔═╡ 94a5cfc0-6e1b-11f0-1c62-5befafc104b6
md"""
(c) Derive the MAP estimates for ``A``.  

> The MAP is simply the location where the posterior has its maximum value, which for a Gaussian posterior is its mean value. We computed in (a) the precision-weighted mean, so we need to divide by precision (or multiply by variance) to get the location of the mean:


```math
\begin{align*}   
\hat{A}_{MAP}  &= \left( \frac{m_A}{v_A} + \frac{1}{\sigma^2} \sum_k x_k\right)\cdot \left(  \frac{1}{v_A} + \frac{N}{\sigma^2} \right)^{-1} \\
&= \frac{v_A \sum_k x_k + \sigma^2 m_A}{N v_A + \sigma^2}
\end{align*}
```

"""

# ╔═╡ 94a5dd3a-6e1b-11f0-204e-1dc4585d4ba0
md"""
(d) Now assume that we do not know the variance of the noise term? Describe the procedure for Bayesian estimation of both ``A`` and ``\sigma^2`` (No need to fully work out to closed-form estimates). 

> A Bayesian treatment requires putting a prior on the unknown variance. The variance is constrained to be positive hence the support of the prior distribution needs to be on the positive reals. (In a multivariate case positivity needs to be extended to symmetric positive definiteness.) Choosing a conjugate prior will simplify matters greatly. In this scenerio the inverse Gamma distribution is the conjugate prior for the unknown variance. In the literature this model is called a Normal-Gamma distribution. See https://www.seas.harvard.edu/courses/cs281/papers/murphy-2007.pdf for the analytical treatment.


"""

# ╔═╡ 94a5f73e-6e1b-11f0-2c4d-8f26672965ec
md"""


"""

# ╔═╡ 94a60dc8-6e1b-11f0-204a-6776bd8d9c25
md"""
  * **[4]** (##) Proof that a linear transformation ``z=Ax+b`` of a Gaussian variable ``\mathcal{N}(x|\mu,\Sigma)`` is Gaussian distributed as

```math
p(z) = \mathcal{N} \left(z \,|\, A\mu+b, A\Sigma A^T \right) 
```

> First, we show that a linear transformation of a Gaussian is a Gaussian. In general, the transformed distribution of ``z=g(x)`` is given by


```math
 p_Z(z) = \frac{p_X(g^{-1}(z))}{\mathrm{det}[g(z)]}\,.
```

> Since the transformation is linear, ``\mathrm{det}[g] = \mathrm{det}[A]``, which is independent of ``z``, and consequently ``p_Z(z)`` has the same functional form as ``p_X(x)``, i.e. ``p_Z(z)`` is a also Gaussian. The mean and variance can easily be determined by the calculation that we used in [question 8 of the Probability Theory exercises](https://nbviewer.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Solutions-Probability-Theory-Review.ipynb#distribution-of-sum). This results in


```math
p(z) = \mathcal{N}\left( z \,|\, A\mu+b, A\Sigma A^T \right) \,.
```

"""

# ╔═╡ 94a7726c-6e1b-11f0-2bd0-5be375f8a951
md"""
  * **[5]** (#) Given independent variables

```math
x \sim \mathcal{N}(\mu_x,\sigma_x^2)
```

and ``y \sim \mathcal{N}(\mu_y,\sigma_y^2)``, what is the PDF for ``z = A\cdot(x -y) + b``?    

> ``z`` is also Gaussian with


```math
p_z(z) = \mathcal{N}(z \,|\, A(\mu_x-\mu_y)+b, \, A (\sigma_x^2 + \sigma_y^2) A^T)
```

"""

# ╔═╡ 94a7a106-6e1b-11f0-0a2d-6b5a9bc30046
md"""
  * **[6]** (###) Compute

```math
\begin{equation*}         \int_{-\infty}^{\infty} \exp(-x^2)\mathrm{d}x \,.     \end{equation*}
```

> For a Gaussian with zero mean and varance equal to ``1`` we have


```math
\int \frac{1}{\sqrt{2\pi}}\exp(-\frac{1}{2}x^2) \mathrm{d}x = 1 
```

> Substitution of ``x = \sqrt{2}y`` with ``\mathrm{d}x=\sqrt{2}\mathrm{d}y`` will simply lead you to $ \int_{-\infty}^{\infty} \exp(-y^2)\mathrm{d}y=\sqrt{\pi}$. If you don't want to use the result of the Gaussian integral, you can still do this integral, see [youtube clip](https://www.youtube.com/watch?v=FYNHt4AMxc0).


"""

# ╔═╡ 94a7c488-6e1b-11f0-1544-297113ae040d
md"""
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

> Let's first compute the moments for the marginals ``p(x)`` and ``p(\theta)``:


```math
\begin{align*}
p(x) &= \int p(x|\theta) p(\theta) \mathrm{d}\theta \\
  &= \int \mathcal{N}(x|\theta,\sigma^2) \mathcal{N}(\theta|\mu_0,\sigma_0^2) \mathrm{d}\theta \\
  &= \int \mathcal{N}(\theta|x,\sigma^2) \mathcal{N}(\theta|\mu_0,\sigma_0^2) \mathrm{d}\theta \\
  &= \mathcal{N}(x|\mu_0,\sigma^2+\sigma_0^2) \underbrace{\int \mathcal{N}(\theta| \cdot,\cdot) \mathrm{d}\theta}_{=1} \\
  &= \mathcal{N}(x|\mu_0,\sigma^2+\sigma_0^2)
\end{align*}
```

> and for ``p(\theta)``:


```math
\begin{align*}
p(\theta) &= \int p(x|\theta) p(\theta) \mathrm{d}x \\
  &= \mathcal{N}(\theta|\mu_0,\sigma_0^2) \underbrace{\int \mathcal{N}(x|\theta,\sigma^2)  \mathrm{d}x}_{=1} \\
  &= \mathcal{N}(\theta|\mu_0,\sigma_0^2)
\end{align*}
```

> With this information, we have


```math
p(z) = p\left(\begin{bmatrix} x \\ \theta \end{bmatrix}\right) = \mathcal{N} \left( \begin{bmatrix} x\\ 
  \theta  \end{bmatrix} 
  \,\left|\, \begin{bmatrix} \mu_0\\ 
  \mu_0\end{bmatrix}, 
         \begin{bmatrix} \sigma_0^2+\sigma^2  & \cdot \\ 
         \cdot &\sigma_0^2 
  \end{bmatrix} 
  \right. \right)
```

> So, we only need to compute ``\Sigma_{x\theta} = \Sigma_{\theta x}^T``. It helps here to also write the system as


```math
\begin{align*}
x &= \theta + \epsilon \\
\theta &\sim \mathcal{N}(\mu_0,\sigma_0^2) \\
\epsilon &\sim \mathcal{N}(0,\sigma^2)
\end{align*}
```

> Now we work out ``\Sigma_{x\theta}``:


```math
\begin{align*}
\Sigma_{x\theta} &= E[(x-E[x])(\theta-E[\theta])^T] \\
&= E[(x-\mu_0)(\theta-\mu_0)^T] \\
&= E[x\theta^T] - \mu_0 E[\theta^T] - E[x]\mu_0^T + \mu_0 \mu_0^T \\
&= E[x\theta^T] - \mu_0 \mu_0^T  \\
&= E[(\theta + \epsilon)\theta^T] - \mu_0 \mu_0^T  \\
&= E[\theta \theta^T] + \underbrace{E[\epsilon]}_{=0} E[\theta^T] - \mu_0 \mu_0^T \\
&= Var[\theta] + E[\theta] E[\theta]^T  - \mu_0 \mu_0^T \\
&= \sigma_0^2 + \mu_0 \mu_0^T - \mu_0 \mu_0^T \\
&= \sigma_0^2
\end{align*}
```



"""

# ╔═╡ 94a7d234-6e1b-11f0-3cbd-83ad37e931e9
md"""

"""

# ╔═╡ Cell order:
# ╟─94a4fe74-6e1b-11f0-024b-c576b489a203
# ╟─94a50a2c-6e1b-11f0-0cd0-019fa190b3d0
# ╟─94a5604c-6e1b-11f0-309f-113a8a787af9
# ╟─94a570e8-6e1b-11f0-193f-4ba36198fc2e
# ╟─94a5833a-6e1b-11f0-3b02-735fd0fbd170
# ╟─94a5b29c-6e1b-11f0-26cf-e1bec8ba04dd
# ╟─94a5bb68-6e1b-11f0-3f64-e5e4fbbf6942
# ╟─94a5cfc0-6e1b-11f0-1c62-5befafc104b6
# ╟─94a5dd3a-6e1b-11f0-204e-1dc4585d4ba0
# ╟─94a5f73e-6e1b-11f0-2c4d-8f26672965ec
# ╟─94a60dc8-6e1b-11f0-204a-6776bd8d9c25
# ╟─94a7726c-6e1b-11f0-2bd0-5be375f8a951
# ╟─94a7a106-6e1b-11f0-0a2d-6b5a9bc30046
# ╟─94a7c488-6e1b-11f0-1544-297113ae040d
# ╟─94a7d234-6e1b-11f0-3cbd-83ad37e931e9
