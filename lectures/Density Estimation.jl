### A Pluto.jl notebook ###
# v1.0.3

#> [frontmatter]
#> image = "https://github.com/bmlip/course/blob/v2/assets/figures/fig-linear-system.png?raw=true"
#> description = "Review of information processing with Gaussian distributions in linear systems."
#> 
#>     [[frontmatter.author]]
#>     name = "BMLIP"
#>     url = "https://github.com/bmlip"

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 5638c1d0-db95-49e4-bd80-528f79f2947e
using HCubature, LinearAlgebra# Numerical integration package

# ╔═╡ 03a36e87-2378-4efc-bcac-9c0609b52784
using MarkdownLiteral: @mdx

# ╔═╡ c97c495c-f7fe-4552-90df-e2fb16f81d15
using BmlipTeachingTools

# ╔═╡ 4484429b-5f31-4a8c-89ed-2f67a1ac869e
using Random

# ╔═╡ 3ec821fd-cf6c-4603-839d-8c59bb931fa9
using Distributions, Plots, LaTeXStrings

# ╔═╡ 00482666-0772-4e5d-bb35-df7b6fb67a1b
using SpecialFunctions

# ╔═╡ b9a38e20-d294-11ef-166b-b5597125ed6d
title("Density Estimation for Continuous and Discrete Data")

# ╔═╡ 5e9a51b1-c6e5-4fb5-9df3-9b189f3302e8
PlutoUI.TableOfContents()

# ╔═╡ b9a46c3e-d294-11ef-116f-9b97e0118e5b
md"""
## Preliminaries

##### Goals 

  * Use a Gaussian distribution for continuous data.
  * Use categorical and multinomial distributions for discrete data.
  * Derive maximum-likelihood parameter estimates from observations.
  * Understand conjugate Bayesian updating: Gaussian–Gaussian and Categorical–Dirichlet pairs
  * Predict future observations while accounting for parameter uncertainty.

##### Materials        

  * Mandatory

      * These lecture notes
  * Optional

      * [Bishop PRML book on Gaussian Distributions](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) (2006), pp. 85-93
      * [Bishop PRML book on Bernoulli and Categorial Distributions](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) (2006), pp. 67-70, 74-76, 93-94

      * [MacKay - 2006 - The Humble Gaussian Distribution](https://github.com/bmlip/course/blob/main/assets/files/Mackay-2006-The-humble-Gaussian-distribution.pdf) (highly recommended!)

      * [Ariel Caticha - 2012 - Entropic Inference and the Foundations of Physics](https://github.com/bmlip/course/blob/main/assets/files/Caticha-2012-Entropic-Inference-and-the-Foundations-of-Physics.pdf), pp.30-34, section 2.8, the Gaussian distribution
  * References

      * [E.T. Jaynes - 2003 - The central, Gaussian or normal distribution, ch.7 in: Probability Theory, The Logic of Science](https://github.com/bmlip/course/blob/main/assets/files/Jaynes%20-%202003%20-%20Probability%20theory%20-%20ch-7%20-%20Gaussian%20distribution.pdf) (Very insightful chapter in Jaynes' book on the Gaussian distribution.)
"""

# ╔═╡ 6f35e623-7e53-4fc7-a0be-5e9dc6f378b4
challenge_statement("Gaussian Density Estimation",header_level=1)

# ╔═╡ 7001ad0f-c3bd-424f-91db-d18c5a4c15f9
md"""

Consider a data set as shown in the figure below

"""

# ╔═╡ 61624ee2-948f-464f-a1f7-ea95bcac7881
md"""

##### Setup 

We have a dataset `D` of observations. `D` is a Matrix, where each column is an observation ``\in \mathbb{R}^2``:
"""

# ╔═╡ 5194faba-2db0-4a28-9151-3e6582b1036a
md"""
We now draw an extra observation ``x_\bullet = (a,b)`` from the same data-generating process:
"""

# ╔═╡ b20fa50e-ef09-43fc-93d7-76bf91cce696
md"""
``D`` and ``x_\bullet`` are shown in the plot above.
"""

# ╔═╡ 01c050ed-e29e-431b-bfd6-f2fdcb9687fd
md"""
##### Problem 

What is the probability that ``x_\bullet`` lies within the shaded rectangle 
```math
S = \{ (x,y) \in \mathbb{R}^2 | 0 \leq x \leq 2, 1 \leq y \leq 2 \} \;?
```
"""

# ╔═╡ e12d7cf3-7e78-4472-8efa-528223668661
S = [[0.0, 2.0], [1.0, 2.0]]

# ╔═╡ 6adc446b-1e00-4a98-b68d-3e356682341d
md"""

##### Solution 

- See [later in this lecture](#Challenge-Revisited:-Gaussian-Density-Estimation). 
"""

# ╔═╡ 71f1c8ee-3b65-4ef8-b36f-3822837de410
md"""
# Continuous Data and the Gaussian Distribution
"""

# ╔═╡ b9a4eb62-d294-11ef-06fa-af1f586cbc15
md"""
## The Moment Parameterization 

Consider a random (vector) variable ``x \in \mathbb{R}^M`` that is "normally" (i.e., Gaussian) distributed. The *moment* parameterization of the Gaussian distribution is completely specified by its *mean* ``\mu`` and *variance* ``\Sigma`` parameters, and given by

```math
p(x | \mu, \Sigma) = \mathcal{N}(x|\mu,\Sigma) \triangleq \frac{1}{\sqrt{(2\pi)^M |\Sigma|}} \,\exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) \right)\,,
```

where ``|\Sigma| \triangleq \mathrm{det}(\Sigma)`` is the determinant of ``\Sigma``.  

For a scalar real variable ``x \in \mathbb{R}``, this works out to 

```math
p(x | \mu, \sigma^2) =  \frac{1}{\sqrt{2\pi\sigma^2 }} \,\exp\left(-\frac{(x-\mu)^2}{2 \sigma^2} \right)\,.
```

It is common to write the (scalar) variance parameter as `` \sigma^2 `` to emphasize that the variance is non-negative.

"""

# ╔═╡ b9a50d0c-d294-11ef-0e60-2386cf289478
md"""

## The Canonical (Natural) Parameterization 

Alternatively, the $(HTML("<span id='natural-parameterization'></span>"))*canonical* (a.k.a. *natural*  or *information* ) parameterization of the Gaussian distribution is given by

```math
\begin{equation*}
p(x | \eta, \Lambda) = \mathcal{N}_c(x|\eta,\Lambda)  = \exp\left( a + \eta^T x - \frac{1}{2}x^T \Lambda x \right) \,,
\end{equation*}
```
where
```math
a = -\frac{1}{2} \left( M \log(2 \pi) - \log |\Lambda| + \eta^T \Lambda \eta\right)
```

is the *normalizing* constant that ensures that ``\int p(x)\mathrm{d}x = 1``, and

```math
\Lambda = \Sigma^{-1}
```

is called the *precision* matrix. The parameter

```math
\eta = \Sigma^{-1} \mu
```

is the *natural* mean, or for clarity, often called the *precision-weighted* mean.

The Gaussian distribution can be expressed in both moment and natural parameterizations, which are mathematically equivalent but differ in how the parameters are defined.

"""

# ╔═╡ b9a52b18-d294-11ef-2d42-19c5e3ef3549
md"""
## Why the Gaussian?
"""

# ╔═╡ b9a5589a-d294-11ef-3fc3-0552a69df7b2
md"""

Why is the Gaussian distribution so ubiquitously used in science and engineering? 

1. Operations on probability distributions tend to lead to Gaussian distributions:

    * Any smooth function with a single rounded maximum goes into a Gaussian function, if raised to higher and higher powers. This is particularly useful in sequential Bayesian inference where repeated updates leads to Gaussian posteriors. (See also this [tweet](https://x.com/Almost_Sure/status/1745480056288186768)). 
    * The [Gaussian distribution has higher entropy](https://en.wikipedia.org/wiki/Differential_entropy#Maximization_in_the_normal_distribution) than any other with the same variance. 
        * Therefore, any operation on a probability distribution that discards information but preserves variance gets us closer to a Gaussian.
        * As an example, see [Jaynes, section 7.1.4](https://github.com/bmlip/course/blob/main/assets/files/Jaynes%20-%202003%20-%20Probability%20theory%20-%20ch-7%20-%20Gaussian%20distribution.pdf) for how this leads to the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem), which results from performing convolution operations on distributions.


2. Once the Gaussian has been attained, this form tends to be preserved. e.g.,   

    * The convolution of two Gaussian functions is another Gaussian function (useful in the sum of 2 variables and linear transformations)
    * The product of two Gaussian functions is another Gaussian function (useful in Bayes rule where multiplication of Gaussian Likelihood and prior leads to a Gaussian posterior).
    * The Fourier transform of a Gaussian function is another Gaussian function.

See also [Jaynes, section 7.14](https://github.com/bmlip/course/blob/main/assets/files/Jaynes%20-%202003%20-%20Probability%20theory%20-%20ch-7%20-%20Gaussian%20distribution.pdf), and the whole chapter 7 in his book for more details on why the Gaussian distribution is so useful.

"""

# ╔═╡ 085233ee-f5ad-4731-89bb-84773182bba6
keyconcept("",
md""" 
Why is the Gaussian distribution so ubiquitously used in science and engineering?

  - Operations on probability distributions tend to lead to Gaussian distributions.
  - Once the Gaussian has been attained, this form tends to be preserved. 		   
		   
""")

# ╔═╡ 4632225a-cbfc-4ec9-a978-21037d8253b6
md"""
## A Notational Convention

As an aside, here is a notational convention that you should be precise about (but many authors are not).

If you want to write that a variable ``x`` is distributed as a Gaussian with mean ``\mu`` and covariance matrix ``\Sigma``, you can write this in either of two ways:

```math
\begin{align*} 
p(x) &= \mathcal{N}(x|\mu,\Sigma) \\
x &\sim \mathcal{N}(\mu,\Sigma)
\end{align*}
```

In the second version, the symbol ``\sim`` can be interpreted as "is distributed as" (a Gaussian with parameters ``\mu`` and ``\Sigma``).

Don't write ``p(x) = \mathcal{N}(\mu,\Sigma)`` because ``p(x)`` is a function of ``x`` but ``\mathcal{N}(\mu,\Sigma)`` is not. 

Also, ``x \sim \mathcal{N}(x|\mu,\Sigma)`` is not entirely proper because you already named the argument on the right-hand-site. On the other hand, ``x \sim \mathcal{N}(\cdot|\mu,\Sigma)`` is fine, as is the shorter ``x \sim \mathcal{N}(\mu,\Sigma)``.

This notational convention of course applies in the same way to other distributions.

"""

# ╔═╡ 9501922f-b928-46e2-8f23-8eb9c64f6198
md"""
# Computing with Gaussians
"""

# ╔═╡ b9a5889c-d294-11ef-266e-d90225222e10
md"""
## Linear Transformations of Gaussian Variables

As shown in the [probability theory lecture](https://bmlip.github.io/course/lectures/Probability%20Theory%20Review.html#linear-transformation), under the linear transformation 

```math
z = Ax + b \,,
```
for given ``A`` and ``b``, the mean and covariance of ``z`` are given by ``\mu_z = A\mu_x + b`` and ``\Sigma_z = A\Sigma_x A^\top``, regardless of the distribution of ``x``.

Since a Gaussian distribution is fully specified by its mean and covariance matrix, it follows that a linear transformation ``z=Ax+b`` of a Gaussian variable ``x \sim \mathcal{N}(\mu_x,\Sigma_x)`` is Gaussian distributed as

```math
p(z) = \mathcal{N} \left(z \,|\, A\mu_x+b, A\Sigma_x A^T \right) \,. 
```

In case ``x`` is not Gaussian, higher order moments may be needed to specify the distribution for ``z``. 


"""

# ╔═╡ 56510a09-073c-4fc8-b0b7-17b20dbb95f0
section_outline("Exercise:", "Linear Transformations" , color= "yellow" )

# ╔═╡ a82378ae-d1be-43f9-b63a-2f897767d1fb
md"""
##### The Sum of Gaussian Variables 

A commonly occurring example of a linear transformation is the *sum of two independent Gaussian variables*:

Let ``x \sim \mathcal{N} \left(\mu_x, \sigma_x^2 \right)`` and ``y \sim \mathcal{N} \left(\mu_y, \sigma_y^2 \right)``. Prove that the PDF for ``z=x+y`` is given by

```math
p(z) = \mathcal{N} \left(z\,|\,\mu_x+\mu_y, \sigma_x^2 +\sigma_y^2 \right) 
```


"""

# ╔═╡ 36eff7bc-72f2-4b48-a109-1861af6834aa
hide_proof(
md"""	   
First, recognize that ``z=x+y`` can be written as a linear transformation ``z=A w``, where
```math
A = \begin{bmatrix} 1 & 1\end{bmatrix}
```	
and
```math
w = \begin{bmatrix} x \\ y\end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \mu_x \\ \mu_y\end{bmatrix}, \begin{bmatrix} \sigma_x^2 & 0 \\ 0 & \sigma_y^2\end{bmatrix}\right) \,.
```		

Making use of the above formula for linear transformations, it follows that
```math
\begin{align*}
p(z) &= \mathcal{N}\big(z\,\big|\,A \mu_w, A \Sigma_w A^T \big) \\
  &= \mathcal{N}\bigg(z\, \bigg|\,\begin{bmatrix} 1 & 1 \end{bmatrix}  \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix}, \begin{bmatrix} 1 & 1 \end{bmatrix}  \begin{bmatrix} \sigma_x^2 & 0 \\ 0 & \sigma_y^2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} \bigg) \\
  &= \mathcal{N} \left(z\,|\,\mu_x+\mu_y, \sigma_x^2 +\sigma_y^2 \right) 
\end{align*}
```
"""	   
)

# ╔═╡ 87f400ac-36f2-4778-a3ba-06dd7652e279
md"""
Following the example above, now compute the PDF for ``z`` if ``x`` and ``y`` were *dependent* Gaussian variables?
"""

# ╔═╡ 9c2bf0a2-4bb6-4769-b47b-6a02c4e73044
hide_solution(
md"""	   
In this case, we assume that 
```math
w = \begin{bmatrix} x \\ y\end{bmatrix} \sim \mathcal{N}\Big( \begin{bmatrix} x \\ y\end{bmatrix}, \begin{bmatrix} \sigma_x^2 & \sigma_{xy} \\ \sigma_{xy} & \sigma_y^2\end{bmatrix}\Big) \,.
```
This leads to 		
```math
\begin{align*}
p(z) &= \mathcal{N}\big(z\,\big|\,A \mu_w, A \Sigma_w A^T \big) \\
 
  &= \mathcal{N} \left(z\,|\,\mu_x+\mu_y, \sigma_x^2 +\sigma_y^2 + 2\sigma_{xy} \right) 
\end{align*}
```
"""	   
)

# ╔═╡ 8f7ecb91-d251-4ac9-bb32-0dd7215382e3
md"""

Consequently, the sum of two independent Gaussian random variables remains Gaussian, with its mean given by the sum of the means and its variance given by the sum of the variances.

A common mistake is to confuse the *sum of two Gaussian-distributed variables*, which remains Gaussian-distributed (see above), with the *sum of two Gaussian distributions*, which is typically not a Gaussian distribution.
"""

# ╔═╡ 883e8244-270e-4c6c-874b-b69d8989c24c

md"""

## Gaussian Maximum Likelihood Estimation

We are given an IID data set ``D = \{x_1,x_2,\ldots,x_N\}``, where ``x_n \in \mathbb{R}^M``. Assume that the data were drawn from a multivariate Gaussian (MVG) 

```math 
p(x_n|\theta) = \mathcal{N}(x_n|\,\mu,\Sigma) \,.
```

Let us derive the maximum likelihood estimates for the parameters ``\mu`` and ``\Sigma``.
"""

# ╔═╡ f02aa0b1-2261-4f65-9bd0-3be33230e0d6
md"""

##### Evaluation of log-likelihood function
Let ``\theta =\{\mu,\Sigma\}``. Prove that the log-likelihood (LLH) function ``\log p(D|\theta)`` can be worked out to

```math
\log p(D|\theta) =
 \frac{N}{2}\log  |\Sigma|^{-1} - \frac{1}{2}\sum_n (x_n-\mu)^T \Sigma^{-1}(x_n-\mu) + \mathrm{const.}

```
			
"""

# ╔═╡ f008a742-6900-4e18-ab4e-b5da53fb64a6
hide_proof(
		md""" 
Hint: it may be helpful here to use the matrix calculus rules from the [5SSD0 Formula Sheet](https://github.com/bmlip/course/blob/main/assets/files/5SSD0_formula_sheet.pdf). This sheet will be made available at the written exam.
	
	```math
\begin{align*}
\log p(D|\theta) &= \log \prod_n p(x_n|\theta) \\
 &= \log \prod_n \mathcal{N}(x_n|\mu, \Sigma) \\
&= \log \prod_n (2\pi)^{-M/2} |\Sigma|^{-1/2} \exp\left\{ -\frac{1}{2}(x_n-\mu)^T \Sigma^{-1}(x_n-\mu)\right\} \\
&= \sum_n \left( \log (2\pi)^{-M/2} + \log  |\Sigma|^{-1/2} -\frac{1}{2}(x_n-\mu)^T \Sigma^{-1}(x_n-\mu)\right) \\
&= \frac{N}{2}\log  |\Sigma|^{-1} - \frac{1}{2}\sum_n (x_n-\mu)^T \Sigma^{-1}(x_n-\mu) + \mathrm{const.}
\end{align*}
```
"""	   )

# ╔═╡ 75e35350-af22-42b1-bb55-15e16cb9c375
md"""
##### Maximum likelihood estimate of mean

Prove that the maximum likelihood estimate of the mean is given by
```math
\hat{\mu} = \frac{1}{N}\sum_n x_n \,.
```

"""

# ╔═╡ 8d2732e8-479f-4744-9b1f-d0364f0c6488
hide_proof(	
md""" 
```math
\begin{align*}
\nabla_{\mu} \log p(D|\theta) &\propto - \sum_n \nabla_{\mu} \left(x_n-\mu \right)^T\Sigma^{-1}\left(x_n-\mu \right)  \\
&= - \sum_n \nabla_{\mu} \left(-2 \mu^T\Sigma^{-1}x_n + \mu^T \Sigma^{-1}\mu \right) \\
&= - \sum_n \left(-2 \Sigma^{-1}x_n + 2\Sigma^{-1}\mu \right) \\
&= -2 \Sigma^{-1} \sum_n (x_n - \mu) \\
&= -2 \Sigma^{-1} \Big( \sum_n x_n - N \mu	\Big) 	
\end{align*}
```	

Since the map ``Ax=0`` for invertible ``A`` can only be true if ``x=0``, it follows that setting the gradient to ``0`` leads to 
```math
		\hat{\mu} = \frac{1}{N}\sum_n x_n \,.
```		
		
""")

# ╔═╡ 0f9feb8d-971e-4a94-8c70-3e1f0d284314
md"""
##### Maximum likelihood estimate of variance

The gradient of the LLH with respect to the variance ``\Sigma`` is a bit more involved. It's actually easier to estimate ``\Sigma`` by taking the derivative to the precision. Compute ``\nabla_{\Sigma^{-1}} \log p(D|\theta)``, and show that the maximum likelihood estimate for ``\Sigma`` is given by

```math
\hat{\Sigma} = \frac{1}{N}\sum_n (x_n-\hat{\mu}) (x_n-\hat{\mu})^T
```
"""


# ╔═╡ 2767b364-6f9a-413d-aa9e-88741cd2bbb1
hide_proof(	
md""" 
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

Setting the derivative to zero leads to ``\hat{\Sigma} = \frac{1}{N}\sum_n (x_n-\hat{\mu})
(x_n-\hat{\mu})^T``.
		
""")


# ╔═╡ c6753ff3-7b5e-45b8-8adc-e0bbaa6be7d3
md"""
# Gaussian Bayesian Estimation and Prediction
"""

# ╔═╡ b9a5cbc2-d294-11ef-214a-c71fb1272326
md"""
## Estimation of a Constant

##### Problem

Let's estimate a constant ``\theta`` from one ''noisy'' measurement ``x`` about that constant. 

We assume the following measurement equations (the tilde ``\sim`` means: 'is distributed as'):

```math
\begin{align*}
x &= \theta + \epsilon \\
\epsilon &\sim \mathcal{N}(0,\sigma^2)
\end{align*}
```

Also, let's assume a Gaussian prior for ``\theta``

```math
\begin{align*}
\theta &\sim \mathcal{N}(\mu_0,\sigma_0^2) \\
\end{align*}
```

For simplicity, we will assume that ``\sigma^2``, ``\mu_0`` and ``\sigma_0^2`` are given. 

What is the PDF for the posterior ``p(\theta|x)`` ?
"""

# ╔═╡ b9a5dcc0-d294-11ef-2c85-657a460db5cd
md"""
#### Model specification

Note that you can rewrite these specifications in probabilistic notation as follows:

```math
\begin{align*}
    p(x|\theta) &=  \mathcal{N}(x|\theta,\sigma^2) &&\quad \text{(likelihood)}\\
    p(\theta) &=\mathcal{N}(\theta|\mu_0,\sigma_0^2) &&\quad \text{(prior)}
\end{align*}
```

"""

# ╔═╡ 7b415578-10fa-4eb1-ab1f-ce3ff57dcf45
md"""
#### Inference
"""

# ╔═╡ b9a67d06-d294-11ef-297b-eb9039786ea7
md"""
Let's do Bayes rule for the posterior PDF ``p(\theta|x)``. 

```math
\begin{align*}
p(\theta|x)  &= \frac{p(x|\theta) p(\theta)}{p(x)} \propto p(x|\theta) p(\theta)  \\
    &= \mathcal{N}(x|\theta,\sigma^2) \mathcal{N}(\theta|\mu_0,\sigma_0^2)   \\
    &\propto \exp \left\{   -\frac{(x-\theta)^2}{2\sigma^2} - \frac{(\theta-\mu_0)^2}{2\sigma_0^2} \right\}  \\
    &\propto \exp \left\{ \theta^2 \cdot \left( -\frac{1}{2 \sigma_0^2} - \frac{1}{2\sigma^2}  \right)  + \theta \cdot  \left( \frac{\mu_0}{\sigma_0^2} + \frac{x}{\sigma^2}\right)   \right\} \\
    &= \exp\left\{ -\frac{\sigma_0^2 + \sigma^2}{2 \sigma_0^2 \sigma^2} \left( \theta - \frac{\sigma_0^2 x +  \sigma^2 \mu_0}{\sigma^2 + \sigma_0^2}\right)^2  \right\} 
\end{align*}
```

which we recognize as a Gaussian distribution w.r.t. ``\theta``. 

"""

# ╔═╡ b9a68d3a-d294-11ef-2335-093a39648007
md"""
(Just as an aside,) this computational 'trick' for multiplying two Gaussians is called **completing the square**. The procedure makes use of the equality 

```math
ax^2+bx+c_1 = a\left(x+\frac{b}{2a}\right)^2+c_2
```

"""

# ╔═╡ b9a697fa-d294-11ef-3a57-7b7ba1f4fd70
md"""
In particular, it follows that the posterior for ``\theta`` is

```math
\begin{equation*}
    p(\theta|x) = \mathcal{N} (\theta |\, \mu_1, \sigma_1^2)
\end{equation*}
```

where

```math
\begin{align*}
  \frac{1}{\sigma_1^2}  &= \frac{\sigma_0^2 + \sigma^2}{\sigma^2 \sigma_0^2} = \frac{1}{\sigma_0^2} + \frac{1}{\sigma^2}  \\
  \mu_1   &= \frac{\sigma_0^2 x +  \sigma^2 \mu_0}{\sigma^2 + \sigma_0^2} = \sigma_1^2 \, \left(  \frac{1}{\sigma_0^2} \mu_0 + \frac{1}{\sigma^2} x \right) 
\end{align*}
```

So, multiplication of two Gaussian distributions yields another (unnormalized) Gaussian with

  * posterior precision equals **sum of prior precisions**
  * posterior precision-weighted mean equals **sum of prior precision-weighted means**


"""

# ╔═╡ b9a6b7b2-d294-11ef-06dc-4de5ef25c1fd
md"""

## Conjugate Distributions

As we just saw, a Gaussian prior, combined with a Gaussian likelihood, makes Bayesian inference analytically solvable in closed-form (!), since 

```math
\begin{equation*}
\underbrace{\text{Gaussian}}_{\text{posterior}}
 \propto \underbrace{\text{Gaussian}}_{\text{likelihood}} \times \underbrace{\text{Gaussian}}_{\text{prior}} \,.
\end{equation*}
```


"""

# ╔═╡ 702e7b10-14a4-42da-a192-f7c02a3d470a
md"""
When applying Bayes rule, if the posterior distribution belongs to the same family as the prior (e.g., both are Gaussian distributions), we say that the prior is a [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior) to the likelihood. 
"""

# ╔═╡ 51d81901-213f-42ce-b77e-10f7ca4a4145

keyconcept("", md"In Bayesian inference, a Gaussian prior distribution is **conjugate** to a Gaussian likelihood (when the variance is known), which ensures that the posterior distribution remains Gaussian. This conjugacy greatly simplifies calculation of Bayes rule.")


# ╔═╡ b9a6c7b6-d294-11ef-0446-c372aa610df8
md"""

## (Multivariate) Gaussian Multiplication


$(HTML("<span id='Gaussian-multiplication'></span>")) In general, the multiplication of two multi-variate Gaussians over ``x`` yields an (unnormalized) Gaussian over ``x``:

```math
\begin{equation*}
\mathcal{N}(x|\mu_a,\Sigma_a) \cdot \mathcal{N}(x|\mu_b,\Sigma_b) = \underbrace{\mathcal{N}(\mu_a|\, \mu_b, \Sigma_a + \Sigma_b)}_{\text{normalization constant}} \cdot \mathcal{N}(x|\mu_c,\Sigma_c) 
\end{equation*}
```

where

```math
\begin{align*}
\Sigma_c^{-1} &= \Sigma_a^{-1} + \Sigma_b^{-1} \\
\Sigma_c^{-1} \mu_c &= \Sigma_a^{-1}\mu_a + \Sigma_b^{-1}\mu_b
\end{align*}
```

"""

# ╔═╡ b9a6ecd2-d294-11ef-02af-37c977f2814b
md"""
Check out that normalization constant ``\mathcal{N}(\mu_a|\, \mu_b, \Sigma_a + \Sigma_b)``. Amazingly, this constant can also be expressed by a Gaussian!

"""

# ╔═╡ b9a6f916-d294-11ef-38cb-b78c0c448550
md"""

Also note that Bayesian inference is trivial in the [*canonical* parameterization of the Gaussian](#natural-parameterization), where we would get

```math
\begin{align*}
 \Lambda_c &= \Lambda_a + \Lambda_b  \quad &&\text{(precisions add)}\\
 \eta_c &= \eta_a + \eta_b \quad &&\text{(precision-weighted means add)}
\end{align*}
```

This property is an important reason why the canonical parameterization of the Gaussian distribution is useful in Bayesian data processing. 

"""

# ╔═╡ d2bedf5f-a0ea-4604-b5da-adf9f11e80be
md"""
It is important to distinguish between two concepts: the *product of Gaussian distributions*, which results in a (possibly unnormalized) Gaussian distribution, and the *product of Gaussian-distributed variables*, which generally does not yield a Gaussian-distributed variable. See the [optional slides below](#OPTIONAL-SLIDES) for further discussion.
"""

# ╔═╡ b9a7073a-d294-11ef-2330-49ffa7faff21
md"""
$(code_example("Product of Two Gaussian PDFs"))

Let's plot the exact product of two Gaussian PDFs as well as the normalized product according to the above derivation.
"""

# ╔═╡ 45c2fb37-a078-4284-9e04-176156cffb1e
begin
	d1 = Normal(0.0, 1); # μ=0, σ^2=1
	d2 = Normal(2.5, 2); # μ=2.5, σ^2=4
	s2_prod = (d1.σ^-2 + d2.σ^-2)^-1
	m_prod = s2_prod * ((d1.σ^-2)*d1.μ + (d2.σ^-2)*d2.μ)
	d_prod = Normal(m_prod, sqrt(s2_prod)) # (Note that we neglect the normalization constant.)
end;

# ╔═╡ df8867ed-0eff-4a52-8f5e-2472467e1aa2
let
	x = range(-4, stop=8, length=100)
	fill = (0, 0.1)
	
	# Plot the first Gaussian
	plot(x, pdf.(d1,x); label=L"\mathcal{N}(0,1)", fill)
	
	# Plot the second Gaussian
	plot!(x, pdf.(d2,x); label=L"\mathcal{N}(3,4)", fill)
	
	#  Plot the exact product
	plot!(x, pdf.(d1,x) .* pdf.(d2,x); label=L"\mathcal{N}(0,1) \mathcal{N}(3,4)", fill)
	
	# Plot the normalized Gaussian product
	plot!(x, pdf.(d_prod,x); label=L"Z^{-1} \mathcal{N}(0,1) \mathcal{N}(3,4)", fill)
end

# ╔═╡ 3a0f7324-0955-4c1c-8acc-0d33ebd16f78
md"""
Check out this mini lecture to learn more about this topic!
"""

# ╔═╡ db730ca7-4850-49c7-a93d-746d393b509b
NotebookCard("https://bmlip.github.io/course/minis/Sum%20and%20product%20of%20Gaussians.html")

# ╔═╡ 4dfa6aa8-2ba1-4ee8-8488-0e11d4132891
section_outline("Exercise:", "Inference with Multiple Observations" , color= "yellow" )

# ╔═╡ b9a80522-d294-11ef-39d8-53a536d66bf9

md"""
Now consider estimation of a constant value ``\theta`` after observing a data set with multiple observations, given by ``D = \{x_1, x_2, \ldots, x_N\}``. Assume the model

```math
\begin{aligned}
x_n &= \theta + \epsilon_n \\
\epsilon_n &\sim \mathcal{N}(0,\sigma^2) \\
\theta &\sim \mathcal{N}(\mu_0,\sigma_0^2)
\end{aligned}
```

Proof that the posterior distribution ``p(\theta|D)`` can be written as

```math
p(\theta|D) = \mathcal{N} (\theta |\, \mu_N, \sigma_N^2)
```

where 

```math
\begin{align*}
  \frac{1}{\sigma_N^2}  &= \frac{1}{\sigma_0^2} + \sum_n \frac{1}{\sigma^2}  \tag{B-2.142} \\
  \mu_N   &= \sigma_N^2 \, \left( \frac{1}{\sigma_0^2} \mu_0 + \sum_n \frac{1}{\sigma^2} x_n  \right) \,.\tag{B-2.141}
\end{align*}
```
"""

# ╔═╡ 87bce1d2-e473-4eb8-a876-1475bf841d70
hide_solution(
md"""
The posterior ``p(\theta|D)`` can be written as a simple extension of the result for estimation with a single data point:  

```math
\begin{align*}
p(\theta|D) \propto  \underbrace{\mathcal{N}(\theta|\mu_0,\sigma_0^2)}_{\text{prior}} \cdot \underbrace{\prod_{n=1}^N \mathcal{N}(x_n|\theta,\sigma^2)}_{\text{likelihood}} \,.
\end{align*}
```

Since ``\mathcal{N}(x_n|\theta,\sigma^2) = \mathcal{N}(\theta | x_n,\sigma^2)``, the posterior ``p(\theta|D)`` is formed by multiplying ``N+1`` Gaussian distributions in ``\theta``, and the result is also Gaussian in ``\theta``, due to the closure of the Gaussian family under multiplication (up to a normalization constant).   

Using the property that precisions and precision-weighted means add when Gaussians are multiplied, the above results (Eqs. B-2.142 and B-2.141) follow immediately.
"""    
)

# ╔═╡ 6efa81ff-6445-4da6-a297-03d7b23f0c4b
md"""
With the posterior over the model parameters in hand, we can now evaluate the posterior predictive distribution for the next sample ``x_{N+1}``. Proof for yourself that

```math
\begin{align*}
 p(x_{N+1}|D) &= \int p(x_{N+1}|\theta) p(\theta|D)\mathrm{d}\theta \\
  &=\mathcal{N}(x_{N+1}|\mu_N, \sigma^2_N +\sigma^2 )
\end{align*}
``` 
"""

# ╔═╡ 922f0eb6-9e29-4b6c-9701-cb7b2f07bb7a
hide_solution(
md"""
```math
\begin{align*}
 p(x_{N+1}|D) &= \int p(x_{N+1}|\theta) p(\theta|D)\mathrm{d}\theta \\
  &= \int \mathcal{N}(x_{N+1}|\theta,\sigma^2) \mathcal{N}(\theta|\mu_N,\sigma^2_N) \mathrm{d}\theta \\
  &\stackrel{1}{=} \int \mathcal{N}(\theta|x_{N+1},\sigma^2) \mathcal{N}(\theta|\mu_N,\sigma^2_N) \mathrm{d}\theta \\
  &\stackrel{2}{=} \int  \mathcal{N}(x_{N+1}|\mu_N, \sigma^2_N +\sigma^2 ) \mathcal{N}(\theta|\cdot,\cdot)\mathrm{d}\theta  \\
  &= \mathcal{N}(x_{N+1}|\mu_N, \sigma^2_N +\sigma^2 ) \underbrace{\int \mathcal{N}(\theta|\cdot,\cdot)\mathrm{d}\theta}_{=1} \\
  &=\mathcal{N}(x_{N+1}|\mu_N, \sigma^2_N +\sigma^2 )
\end{align*}
```

To follow the above derivation of ``p(x_{N+1}|D)``, note that transition ``1`` relies on the identity
```math
\mathcal{N}(x|\mu,\Sigma) = \mathcal{N}(\mu|x,\Sigma)
```
and transition ``2`` derives from using the multiplication rule for Gaussians.
""")

# ╔═╡ ffdfe355-1c08-4b94-97f3-f9df2b8325e5
md"""
Note that uncertainty about the next observation ``x_{N+1}`` involves both uncertainty about the parameter (``\sigma_N^2``) and observation noise ``\sigma^2``.
"""

# ╔═╡ f4735ea4-b2b8-4021-a66a-976ff6639653
md"""
# Discrete Data: Categorical and Multinomial Distributions
"""

# ╔═╡ 8cc63234-0553-471e-9ad8-f44efefc5f3a
md"""
## Discrete Data: the 1-of-K Coding Scheme

Consider a coin-tossing experiment with outcomes ``x \in\{0,1\}`` (tail and head, respectively) and let ``0\leq \mu \leq 1`` represent the probability of heads. The data-generating distribution for this model can be written as a [**Bernoulli distribution**](https://en.wikipedia.org/wiki/Bernoulli_distribution):

```math
 
p(x|\mu) = \mu^{x}(1-\mu)^{1-x}
```

Note that the variable ``x`` acts as a (binary) **selector** for the tail or head probabilities. Think of this as an 'if'-statement in programming.

"""

# ╔═╡ fd2a4003-1362-43f6-9bbe-ce667e4e4611
md"""
Now consider a ``K``-sided coin (e.g., a six-faced *die* (pl.: dice)). How should we encode outcomes? Two natural options present themselves:

##### Option 1: label encoding 

```math
x \in \{1,2,\ldots,K\} \,.
```
  - E.g., for ``K=6``, if the die lands on the 3rd face, then ``x=3``.
  - This coding scheme is called **label** (or **index**) encoding. 

##### Option 2: one-hot encoding

```math
x = (x_1,\ldots,x_K)^T 
```
where ``x_k`` are **binary selection variables**, given by
```math
x_k = \begin{cases} 1 & \text{if die landed on $k$th face}\\
0 & \text{otherwise} \end{cases}
```
  - For instance, for ``K=6``, if the die lands on the ``3``-rd face, then ``x=(0,0,1,0,0,0)^T``.

  - This coding scheme is called a **1-of-K** or **one-hot** coding scheme.

It turns out that the one-hot coding scheme is mathematically more convenient!

"""

# ╔═╡ 83c12d79-9fe3-4b5e-87b7-31174d0849cf
keyconcept("", "Discrete event outcomes are typically represented via one-hot encoding, in which each outcome corresponds to a unique binary indicator vector.")

# ╔═╡ bf52bd62-44f2-4bcf-acc9-e4180c0dbc28
md"""
## The Categorical Distribution

Consider a toss with a ``K``-sided die. We use a one-hot coding scheme, i.e., the outcome is encoded as 
```math
x_{k} = \begin{cases} 1 & \text{if the throw landed on $k$-th face}\\
0 & \text{otherwise} \end{cases} \,.
```

Assume the probabilities


```math 
p(x_{k}=1) = \mu_k \quad \text{with } \mu_k \geq 0 \text{ and }\sum_k \mu_k  = 1 \,.
```
The data generating distribution for one-hot encoded outcome ``x = (x_{1},x_{2},\ldots,x_{K})`` (and ``\mu = (\mu_1,\mu_2,\dots,\mu_k)^T``) is then given by 

```math
p(x|\mu) = \mu_1^{x_1} \mu_2^{x_2} \cdots \mu_K^{x_K}=\prod_{k=1}^K \mu_k^{x_k} \tag{B-2.26}
```

This generalized Bernoulli distribution is called the [**categorical distribution**](https://en.wikipedia.org/wiki/Categorical_distribution).

"""

# ╔═╡ 21e9f9bd-71df-4fc0-adca-d4dc5835584a
md"""
# Bayesian Density Estimation for a Loaded Die

Now let's proceed with learning the parameters for a model for ``N`` independent-and-identically-distributed (IID) rolls of a ``K``-sided die, based on observed data set ``D=\{x_1,\ldots,x_N\}``. 

"""

# ╔═╡ 52cd2c58-62a6-43ee-9188-2e23ea4037ed
md"""
## Model specification

#### likelihood function

The outcomes ``x_n`` are encoded as
```math
x_{nk} = \begin{cases} 1 & \text{if the $n$-th throw landed on $k$-th face}\\
0 & \text{otherwise} \end{cases}
```

and the likelihood function for ``\mu`` is now

```math
p(D|\mu) = \prod_n \prod_k \mu_k^{x_{nk}} = \prod_k \mu_k^{\sum_n x_{nk}} = \prod_k \mu_k^{m_k} \tag{B-2.29}
```

where ``m_k= \sum_n x_{nk}`` is the total number of occurrences that the outcome landed on face ``k``. The vector ``m = (m_1,m_2, \ldots, m_K)^T`` is known as the **count vector**. Note that ``\sum_k m_k = N``.

This distribution depends on the observations **only** through the ''observed'' counts ``\{m_k\}``. For given counts ``\{m_k\}``, ``p(D|\mu)`` can be interpreted as a likelihood function for ``\mu``.

"""

# ╔═╡ d0751756-8550-4d49-9cd7-884c2fbcd77b
md"""

#### prior distribution

Next, we need a prior for the parameters ``\mu = (\mu_1,\mu_2,\ldots,\mu_K)^T``. 

In the [binary coin toss example](https://bmlip.github.io/course/lectures/Bayesian%20Machine%20Learning.html#beta-prior), we used a [beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) that was conjugate with the binomial and forced us to choose prior pseudo-counts. 

The generalization of the beta prior to ``K`` parameters ``\{\mu_k\}`` is the [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution):

```math
p(\mu|\alpha) = \mathrm{Dir}(\mu|\alpha) = \frac{\Gamma\left(\sum_k \alpha_k\right)}{\Gamma(\alpha_1)\cdots \Gamma(\alpha_K)} \prod_{k=1}^K \mu_k^{\alpha_k-1} 
```

where ``\Gamma(\cdot)`` is the [Gamma function](https://en.wikipedia.org/wiki/Gamma_function). 

  - The Gamma function can be interpreted as a generalization of the factorial function to the real (``\mathbb{R}``) numbers. If ``n`` is a natural number (``1,2,3, \ldots $), then $\Gamma(n) = (n-1)!``, where ``(n-1)! = (n-1)\cdot (n-2) \cdot 1``.

As before for the Beta distribution in the coin toss experiment, you can interpret ``\alpha_k`` as the prior number of (pseudo-)observations that the die landed on the  ``k``-th face.

"""

# ╔═╡ 7908e67b-f390-4636-9d9d-2f8b73dc4b3a
md"""
## Inference for ``\{\mu_k\}``

The posterior for  ``\{\mu_k\}`` can be obtained through Bayes rule:

```math
\begin{align*}
p(\mu|D,\alpha) &\propto p(D|\mu) \cdot p(\mu|\alpha) \\
  &\propto  \prod_k \mu_k^{m_k} \cdot \prod_k \mu_k^{\alpha_k-1} \\
  &= \prod_k \mu_k^{\alpha_k + m_k -1}\\
  &\propto \mathrm{Dir}\left(\mu\,|\,\alpha + m \right) \tag{B-2.41} \\
  &= \frac{\Gamma\left(\sum_k (\alpha_k + m_k) \right)}{\Gamma(\alpha_1+m_1) \Gamma(\alpha_2+m_2) \cdots \Gamma(\alpha_K + m_K)} \prod_{k=1}^K \mu_k^{\alpha_k + m_k -1}
\end{align*}
```

where ``m = (m_1,m_2,\ldots,m_K)^T`` is the count vector.

"""

# ╔═╡ dfde69ed-6ad3-4d09-bb59-76e7aebb9868
md"""
We recognize the ``(\alpha_k)``'s as prior pseudo-counts and the Dirichlet distribution shows to be a [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior) to the categorical/multinomial:

```math
\begin{align*}
\underbrace{\text{Dirichlet}}_{\text{posterior}} &\propto \underbrace{\text{categorical}}_{\text{likelihood}} \cdot \underbrace{\text{Dirichlet}}_{\text{prior}}
\end{align*}
```

"""

# ╔═╡ 33484269-19b0-4f01-ab83-a94fc68eace3
md"""
This is actually a generalization of the conjugate relation that we found for the binary coin toss: 

```math
\begin{align*}
\underbrace{\text{beta}}_{\text{posterior}} &\propto \underbrace{\text{binomial}}_{\text{likelihood}} \cdot \underbrace{\text{beta}}_{\text{prior}}
\end{align*}
```

"""

# ╔═╡ 2fc84689-4f84-4a1d-b5a1-eeb6adf3adeb
md"""
## $(HTML("<span id='prediction-loaded-die'>Prediction of next toss for the loaded die</span>"))

Let's apply what we have learned about the loaded die to compute the probability that we throw the ``k``-th face at the next toss. 

```math
\begin{align*}
p(x_{\bullet,k}=1|D)  &= \int p(x_{\bullet,k}=1|\mu)\,p(\mu|D) \,\mathrm{d}\mu \\
  &= \int_0^1 \mu_k \times  \mathcal{Dir}(\mu|\,\alpha+m) \,\mathrm{d}\mu  \\
  &= \mathrm{E}\left[ \mu_k | D\right] \\
  &= \frac{m_k + \alpha_k }{ N+ \sum_k \alpha_k}
\end{align*}
```

(You can find the mean of the Dirichlet distribution ``\mathrm{E}\left[ \mu_k \right]`` at its [Wikipedia site](https://en.wikipedia.org/wiki/Dirichlet_distribution)). 

This result is simply a generalization of [**Laplace's rule of succession**](https://en.wikipedia.org/wiki/Rule_of_succession).

"""

# ╔═╡ 53a1e971-d774-46fb-b328-4cd71585ee75
md"""
## Categorical, Multinomial and Related Distributions

In the above derivation, we noticed that the data generating distribution for ``N`` die tosses with data outcomes ``D=\{x_1,\ldots,x_N\}`` only depends on the **counts** ``m_k``:

```math
p(D|\mu) = \prod_n \underbrace{\prod_k \mu_k^{x_{nk}}}_{\text{categorical dist.}} = \prod_k \mu_k^{\sum_n x_{nk}} = \prod_k \mu_k^{m_k} \tag{B-2.29}
```

"""

# ╔═╡ 55c101f3-90f3-4a1a-9e65-72a35db254a9
md"""
A related distribution is the distribution over count observations ``D_m=\{m_1,\ldots,m_K\}``, which is called the **multinomial distribution**,

```math
p(D_m|\mu) =\frac{N!}{m_1! m_2!\ldots m_K!} \,\prod_k \mu_k^{m_k}\,.
```

"""

# ╔═╡ f7ae238e-8a9f-4510-85f7-e7cdae100f99
md"""
(We insert this slide only to alert you to the difference between using one-hot encoded outcomes ``D=\{x_1,x_2,\ldots,x_N\}`` as the data, versus using counts ``D_m = \{m_1,m_2,\ldots,m_K\}`` as the data. When used as a likelihood function for ``\mu``, it makes no difference whether you use ``p(D|\mu)`` or ``p(D_m|\mu)``.)

"""

# ╔═╡ f8b57f11-f014-4a4a-aa41-8217b1a5b21d
md"""
## Multinomial Maximum Likelihood Estimation

#### Maximum likelihood as a special case of Bayesian estimation

We can obtain the maximum likelihood estimate for ``\mu_k`` based on ``N`` throws of a ``K``-sided die within the Bayesian framework by letting the prior for ``\mu`` approach a uniform distribution. For a Dirichlet prior ``\mathrm{Dir}(\mu | \alpha)``, this corresponds to setting
``\alpha \rightarrow (1, 1, \dots, 1)``.


Prove for yourself that 

```math
\begin{align*}
\hat{\mu}_k &= \arg\max_{\mu_k} p(D|\mu) = \frac{m_k}{N}\,.
\end{align*}
```

"""

# ╔═╡ bed4962d-cd5f-4bff-bf25-68e524b20183
hide_proof(
md"""
```math
\begin{align*}
\hat{\mu}_k &= \arg\max_{\mu_k} p(D|\mu) \\
&= \arg\max_{\mu_k} p(D|\mu) \cdot \underbrace{\left.\mathrm{Dir}(\mu|\alpha)\right|_{\alpha=(1,1,\ldots,1)}}_{\text{uniform distr.}} \\
&= \arg\max_{\mu_k} \left.p(\mu|D,\alpha)\right|_{\alpha=(1,1,\ldots,1)}  \\
&= \arg\max_{\mu_k} \left.\mathrm{Dir}\left( \mu | m + \alpha \right)\right|_{\alpha=(1,1,\ldots,1)} \\
&= \frac{m_k}{\sum_k m_k} = \frac{m_k}{N}
\end{align*}
```

where we used the fact that the [maximum of the Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution#Mode) ``\mathrm{Dir}(\{\alpha_1,\ldots,\alpha_K\})`` is obtained at  ``(\alpha_k-1)/(\sum_k\alpha_k - K)``.

		""")

# ╔═╡ bed7072c-53b0-47f7-bf77-32deed55b3e5
md"""
In practice, maximum likelihood estimation (MLE) is, of course, not executed by first doing full Bayesian estimation and then considering MLE as a special case. In the [optional slide below](#Multinomial-Maximum-Likelihood-Estimation-by-Optimizing-a-Constrained-Log-likelihood), you can verify that direct minimization of the likelihood function leads to the same answer.
"""

# ╔═╡ 71c09b1e-de5c-404d-8099-965433e85481
challenge_solution("Gaussian Density Estimation", header_level=1)

# ╔═╡ eb901f4d-2ecd-44fb-baf2-80e65489186e
md"""

Let's solve the challenge from the beginning of the lecture. We apply maximum likelihood estimation to fit a 2-dimensional Gaussian model (``m``) to data set ``D``. Next, we evaluate ``p(x_\bullet \in S | m)`` by (numerical) integration of the Gaussian pdf over ``S``: ``p(x_\bullet \in S | m) = \int_S p(x|m) \mathrm{d}x``.

"""

# ╔═╡ b89360b8-39fa-46e9-96c8-7eece50fcb90
md"""
# Summary
"""

# ╔═╡ a439c0a7-afa1-4d9a-8737-58d341744016
keyconceptsummary()

# ╔═╡ 79a99a22-3bb5-431b-bf84-5dce5cccfe25
exercises(header_level=1)

# ╔═╡ 14b3edcc-0d16-4055-9b1c-7f324514a0a9
md"""
#### Gaussian Message Passing (**)

This exercise is a continuation of the [exercise on message passing for an addition node](https://bmlip.github.io/course/lectures/Factor%20Graphs.html#Messages-for-the-Addition-Node-(*)).
"""

# ╔═╡ dd7786e2-d6ac-4dba-abca-3686242c067d
TwoColumn(
md"""
Consider an addition node

```math
f_+(x,y,z) = \delta(z-x-y)
```
Assume that both incoming messages are Gaussian, namely ``\overrightarrow{\mu}_{X}(x) \sim \mathcal{N}(\overrightarrow{m}_X,\overrightarrow{V}_X)`` and ``\overrightarrow{\mu}_{Y}(y) \sim \mathcal{N}(\overrightarrow{m}_Y,\overrightarrow{V}_Y)``. 

""", 
	
@htl """

<img src="https://github.com/bmlip/course/blob/main/assets/figures/ffg-addition-node.png?raw=true" alt=" " style="display: block; width: 100%; margin: 0 auto;">

""")

# ╔═╡ b7a810a3-dc38-4e72-ab10-2ad2f064bdbb
md"""

- (a) Evaluate the outgoing message ``\overrightarrow{\mu}_{Z}(z)``. 

- (b) For the same summation node, work out the SP update rule for the backward message ``\overleftarrow{\mu}_{X}(x)`` as a function of ``\overrightarrow{\mu}_{Y}(y)`` and  ``\overleftarrow{\mu}_{Z}(z)``. And further refine the answer for Gaussian messages.


"""

# ╔═╡ f711b053-dccf-4bf1-b285-e8da94a48b68
hide_solution(
md"""

- (a) Evaluate the outgoing message ``\overrightarrow{\mu}_{Z}(z)``. 

In the [exercise on message passing for an addition node](https://bmlip.github.io/course/lectures/Factor%20Graphs.html#Messages-for-the-Addition-Node-(*)), we found that the outgoing message is given by

```math
\begin{align*}
  \overrightarrow{\mu}_{Z}(z) &= \iint  \overrightarrow{\mu}_{X}(x) \overrightarrow{\mu}_{Y}(y) \,\delta(z-x-y) \,\mathrm{d}x \mathrm{d}y \\
   &=  \int  \overrightarrow{\mu}_{X}(x) \overrightarrow{\mu}_{Y}(z-x) \,\mathrm{d}x \,, 
  \end{align*}
```


For Gaussian incoming messages, these update rules evaluate to ``\overrightarrow{\mu}_{Z}(z) \sim \mathcal{N}(\overrightarrow{m}_Z,\overrightarrow{V}_Z)`` with


```math
\begin{align*}
  \overrightarrow{m}_Z &= \overrightarrow{m}_X + \overrightarrow{m}_Y \\
  \overrightarrow{V}_z &= \overrightarrow{V}_X + \overrightarrow{V}_Y \,.
\end{align*}
```

- (b) For the same summation node, work out the SP update rule for the backward message ``\overleftarrow{\mu}_{X}(x)`` as a function of ``\overrightarrow{\mu}_{Y}(y)`` and  ``\overleftarrow{\mu}_{Z}(z)``. And further refine the answer for Gaussian messages.

```math
\begin{align*}
  \overleftarrow{\mu}_{X}(x) &= \iint  \overrightarrow{\mu}_{Y}(y) \overleftarrow{\mu}_{Z}(z) \,\delta(z-x-y) \,\mathrm{d}y \mathrm{d}z \\
   &=  \int  \overrightarrow{\mu}_{Y}(z-x) \overleftarrow{\mu}_{Z}(z) \,\mathrm{d}z  
  \end{align*}
```

and now further with Gaussian messages,


```math
\begin{align*}
  \overleftarrow{\mu}_{X}(x) &= \int  \mathcal{N}(z-x | m_y,V_y)  \mathcal{N}(z | m_z,V_z)\,\mathrm{d}z \\
  &=  \int  \mathcal{N}(z | x+ m_y,V_y)  \mathcal{N}(z | m_z,V_z)\,\mathrm{d}z  \\
  &=  \int  \mathcal{N}(x+m_y | m_z,V_y+V_z)  \mathcal{N}(z | \cdot,\cdot)\,\mathrm{d}z  \\
  &= \mathcal{N}(x | m_z-m_y, V_y+V_z) 
\end{align*}
```


""")

# ╔═╡ 1df7a10d-c4f6-40d6-8f5a-cbd79ef1d415
TwoColumn(
md"""
#### Gaussian Signals in a Linear System (**)
	
Given independent variables ``x \sim \mathcal{N}(\mu_x, \sigma_x^2)`` and ``y \sim \mathcal{N}(\mu_y, \sigma_x^y)``, what is the PDF for

```math
z =a \cdot (x-y) + b \,\text{?}
```

""", 
@htl """

<img src="https://github.com/bmlip/course/blob/v2/assets/figures/fig-linear-system.png?raw=true" alt=" " style="display: block; width: 100%; margin: 0 auto;">

""")

# ╔═╡ 673360e8-27ed-471c-a866-15af550df5e7
hide_solution(
md"""

		
Let ``z \sim \mathcal{N}(\mu_z, \sigma_z^2)``. We proceed by working out the mean and variance for ``z`` explicitly, yielding


```math
\begin{align}
\mu_z &= \mathrm{E}\left[ z\right] \\
&= \mathrm{E}\left[ a\cdot(x -y) + b\right] \\ 
&= a\cdot\mathrm{E}\left[ (x -y)\right] + b \\ 
&= a\cdot(\mu_x -\mu_y) + b
\end{align}
```
and
```math
\begin{align}
\sigma_z^2 &= \mathrm{E}\left[ (z-\mu_z)(z-\mu_z)^T\right] \\
&= \mathrm{E}\left[ a\cdot \big( (x - \mu_x) - (y - \mu_y) \big) \big( (x - \mu_x) - (y - \mu_y) \big)^T \cdot a^T\right] \\ 
&= a\cdot(\sigma_x^2 - 2 \underbrace{\sigma_{xy}}_{=0} + \sigma_y^2) \cdot a^T \\ 
&= a^2\cdot(\sigma_x^2 + \sigma_y^2)
\end{align}
```
"""		
)

# ╔═╡ 22539cfe-3694-4100-8120-ca6ac1e66b31
md"""
#### Estimation of a Constant (**)

We make ``N`` IID observations ``D=\{x_1 \dots x_N\}`` and assume the following model

```math
\begin{align}
x_k &= A + \epsilon_k \\
A &\sim \mathcal{N}(m_A,v_A) \\
\epsilon_k &\sim \mathcal{N}(0,\sigma^2) \,.
\end{align}
```

We assume that ``\sigma`` has a known value and are interested in deriving an estimator for ``A``.

- (a) Derive the Bayesian (posterior) estimate ``p(A|D)``.   

- (b) Derive the Maximum Likelihood estimate for ``A``.

- (c) Derive the MAP estimates for ``A``.  

- (d) Now assume that we do not know the variance of the noise term? Describe the procedure for Bayesian estimation of both ``A`` and ``\sigma^2`` (No need to fully work out to closed-form estimates). 

"""

# ╔═╡ fa197526-6706-47ce-b84b-5675eee00610
hide_solution(
md"""
- (a) Derive the Bayesian (posterior) estimate ``p(A|D)``.   

Since ``p(D|A) = \prod_k \mathcal{N}(x_k|A,\sigma^2)`` is a Gaussian likelihood and ``p(A)`` is a Gaussian prior, their multiplication is proportional to a Gaussian. We will work this out with the canonical parameterization of the Gaussian since it is easier to multiply Gaussians in that domain. This means the posterior ``p(A|D)`` is


```math
\begin{align*}
   p(A|D) &\propto p(A) p(D|A) \\
   &= \mathcal{N}(A|m_A,v_A) \prod_{k=1}^N \mathcal{N}(x_k|A,\sigma^2) \\
   &= \mathcal{N}(A|m_A,v_A) \prod_{k=1}^N \mathcal{N}(A|x_k,\sigma^2) \\
   &= \mathcal{N}_c\big(A \Bigm|\frac{m_A}{v_A},\frac{1}{v_A}\big)\prod_{k=1}^N \mathcal{N}_c\big(A\Bigm| \frac{x_k}{\sigma^2},\frac{1}{\sigma^2}\big) \\
       &\propto \mathcal{N}_c\big(A \Bigm| \frac{m_A}{v_A} + \frac{1}{\sigma^2} \sum_k x_k , \frac{1}{v_A} + \frac{N}{\sigma^2}  \big)      \,, 
  \end{align*}
```

where we have made use of the fact that precision-weighted means and precisions add when multiplying Gaussians. In principle, this description of the posterior completes the answer.

- (b) Derive the Maximum Likelihood estimate for ``A``.

The ML estimate can be found by


```math
\begin{align*}
  \nabla \log p(D|A) &=0\\
  \nabla \sum_k \log \mathcal{N}(x_k|A,\sigma^2) &= 0 \\
  \nabla \frac{-1}{2}\sum_k \frac{(x_k-A)^2}{\sigma^2} &=0\\
  \sum_k(x_k-A) &= 0 \\
  \Rightarrow \hat{A}_{ML} = \frac{1}{N}\sum_{k=1}^N x_k
\end{align*}
```

- (c) Derive the MAP estimates for ``A``.  

The MAP is simply the location where the posterior has its maximum value, which for a Gaussian posterior is its mean value. We computed in (a) the precision-weighted mean, so we need to divide by precision (or multiply by variance) to get the location of the mean:


```math
\begin{align*}   
\hat{A}_{MAP}  &= \left( \frac{m_A}{v_A} + \frac{1}{\sigma^2} \sum_k x_k\right)\cdot \left(  \frac{1}{v_A} + \frac{N}{\sigma^2} \right)^{-1} \\
&= \frac{v_A \sum_k x_k + \sigma^2 m_A}{N v_A + \sigma^2}
\end{align*}
```

- (d) Now assume that we do not know the variance of the noise term? Describe the procedure for Bayesian estimation of both ``A`` and ``\sigma^2`` (No need to fully work out to closed-form estimates). 

A Bayesian treatment requires putting a prior on the unknown variance. The variance is constrained to be positive; hence the support of the prior distribution needs to be on the positive reals. (In a multivariate case, positivity needs to be extended to symmetric positive definiteness.) Choosing a conjugate prior will simplify matters greatly. In this scenerio, the inverse Gamma distribution is the conjugate prior for the unknown variance. In the literature, this model is called a Normal-Gamma distribution. See [Murphy (2007)](https://www.seas.harvard.edu/courses/cs281/papers/murphy-2007.pdf) for the analytical treatment.
""")

# ╔═╡ 645308ac-c9e3-4d6f-bcff-82327fbb8edf
md"""
####  Conversion to Joint Distribution (**)

Show that the system

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

# ╔═╡ 03c399e1-d0d8-493a-9f95-4209918d132a
hide_solution(
md"""
Let's first compute the moments for the marginals ``p(x)`` and ``p(\theta)``:


```math
\begin{align*}
p(x) &= \int p(x|\theta) p(\theta) \mathrm{d}\theta \\
  &= \int \mathcal{N}(x|\theta,\sigma^2) \mathcal{N}(\theta|\mu_0,\sigma_0^2) \mathrm{d}\theta \\
  &= \int \mathcal{N}(\theta|x,\sigma^2) \mathcal{N}(\theta|\mu_0,\sigma_0^2) \mathrm{d}\theta \\
  &= \mathcal{N}(x|\mu_0,\sigma^2+\sigma_0^2) \underbrace{\int \mathcal{N}(\theta| \cdot,\cdot) \mathrm{d}\theta}_{=1} \\
  &= \mathcal{N}(x|\mu_0,\sigma^2+\sigma_0^2)
\end{align*}
```

and for ``p(\theta)``:


```math
\begin{align*}
p(\theta) &= \int p(x|\theta) p(\theta) \mathrm{d}x \\
  &= \mathcal{N}(\theta|\mu_0,\sigma_0^2) \underbrace{\int \mathcal{N}(x|\theta,\sigma^2)  \mathrm{d}x}_{=1} \\
  &= \mathcal{N}(\theta|\mu_0,\sigma_0^2)
\end{align*}
```

With this information, we have


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

So, we only need to compute ``\Sigma_{x\theta} = \Sigma_{\theta x}^T``. It helps here to write the system as


```math
\begin{align*}
x &= \theta + \epsilon \\
\theta &\sim \mathcal{N}(\mu_0,\sigma_0^2) \\
\epsilon &\sim \mathcal{N}(0,\sigma^2)
\end{align*}
```

Now we work out ``\Sigma_{x\theta}``:


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
( I am sure one of you can do it simpler and faster. Let me know:)

		
""")

# ╔═╡ 32a7da22-7dba-41eb-8125-a9c8409a968e
md"""

#### Laplace's Generalized Rule of Succession (**) 

Show that Laplace's generalized rule of succession can be worked out to a prediction that is composed of a prior prediction and data-based correction term.


"""

# ╔═╡ a0a96322-d71e-4e5c-95a4-fcc01c0db542
hide_solution(
md"""

```math
\begin{align*}
p(&x_{\bullet,k}=1|D) = \frac{m_k + \alpha_k }{ N+ \sum_k \alpha_k} \\
&= \frac{m_k}{N+\sum_k \alpha_k}  + \frac{\alpha_k}{N+\sum_k \alpha_k}\\
&= \frac{m_k}{N+\sum_k \alpha_k} \cdot \frac{N}{N} + \frac{\alpha_k}{N+\sum_k \alpha_k}\cdot \frac{\sum_k \alpha_k}{\sum_k\alpha_k} \\
&= \frac{N}{N+\sum_k \alpha_k} \cdot \frac{m_k}{N} + \frac{\sum_k \alpha_k}{N+\sum_k \alpha_k} \cdot \frac{\alpha_k}{\sum_k\alpha_k} \\
&= \frac{N}{N+\sum_k \alpha_k} \cdot \frac{m_k}{N} + \bigg( \frac{\sum_k \alpha_k}{N+\sum_k \alpha_k} + \underbrace{\frac{N}{N+\sum_k \alpha_k} - \frac{N}{N+\sum_k \alpha_k}}_{0}\bigg) \cdot \frac{\alpha_k}{\sum_k\alpha_k} \\
&= \frac{N}{N+\sum_k \alpha_k} \cdot \frac{m_k}{N} + \bigg( 1 - \frac{N}{N+\sum_k \alpha_k}\bigg) \cdot \frac{\alpha_k}{\sum_k\alpha_k} \\
&= \underbrace{\frac{\alpha_k}{\sum_k\alpha_k}}_{\text{prior prediction}} + \underbrace{\frac{N}{N+\sum_k \alpha_k} \cdot \underbrace{\left(\frac{m_k}{N} - \frac{\alpha_k}{\sum_k\alpha_k}\right)}_{\text{prediction error}}}_{\text{data-based correction}}
\end{align*}
```

(If you know how to do it shorter and more elegantly, please post in Piazza.)

This decomposition is the natural consequence of doing Bayesian estimation, which always involves a prior-based prediction term and a likelihood-based (or data-based) correction term that can be interpreted as a (precision-weighted) prediction error. 
		
		""")

# ╔═╡ ff9e3293-0a49-4c51-b118-cf7af45e4bcb
md"""

#### Evidence for the Multinomial-Dirichlet model (**) 

As above, consider the following model assumptions for $N$ tosses with a $K$-sided die with parameters $\mu = (\mu_1,\mu_2, \ldots,\mu_K)$.  

```math
\begin{align}
p(D|\mu) &= \prod_{n=1}^N \mathrm{Cat}(x_n|\mu) = \prod_{k=1}^{K} \mu_k^{m_k} \tag{likelihood}\\
p(\mu|\alpha) &= \mathrm{Dir}(\mu|\alpha) = \frac{1}{B(\alpha)} \prod_{k=1}^{K} \mu_k^{\alpha_k -1}   \tag{prior}
\end{align}
```
where $B(\alpha) = \frac{\prod_k \Gamma(\alpha_k)}{\Gamma(\sum_k \alpha_k)}$ is known as the [Beta function](https://en.wikipedia.org/wiki/Beta_function).

Work out both the model evidence and the posterior distribution for $\mu$.
"""

# ╔═╡ fded7c8c-feed-4630-a50d-5a854536cab3
hide_solution(
	md"""

	```math
	\begin{align}
	\overbrace{\prod_{k=1}^{K} \mu_k^{m_k}}^{\text{likelihood }p(D|\mu)} \cdot \overbrace{\frac{1}{B(\alpha)} \prod_{k=1}^{K} \mu_k^{\alpha_k -1}}^{\text{prior }p(\mu|\alpha)}  
	&= \frac{1}{B(\alpha)} \prod_{k=1}^{K} \mu_k^{m_k + \alpha_k -1} \\
	&= \frac{B(m+\alpha)}{B(\alpha)} \frac{1}{B(m+\alpha)}\prod_{k=1}^{K} \mu_k^{m_k + \alpha_k -1} \\
	&= \underbrace{\frac{B(m+\alpha)}{B(\alpha)}}_{\text{evidence }p(D|\alpha)} \,\underbrace{\mathrm{Dir}(\mu|m+\alpha)}_{\text{posterior }p(\mu|D,\alpha)} 
	\end{align} 
	```

	This equation is the equivalent of the [Gaussian multiplication formula](https://bmlip.github.io/course/lectures/The%20Gaussian%20Distribution.html#(Multivariate)-Gaussian-Multiplication) for discrete data. Note that the evidence is a scalar normalizer for given observations $m$ and pseudo-observations ("prior" observations) $\alpha$.
	""")

# ╔═╡ 6dfc31a0-d0d7-4901-a876-890df9ab4258
md"""
# Optional Slides
"""

# ╔═╡ b9a885a8-d294-11ef-079e-411d3f1cda03
md"""
## Conditioning and Marginalization of a Gaussian

Let ``z = \begin{bmatrix} x \\ y \end{bmatrix}`` be jointly normal distributed as

```math
\begin{align*}
p(z) &= \mathcal{N}(z | \mu, \Sigma) 
  =\mathcal{N} \left( \begin{bmatrix} x \\ y \end{bmatrix} \left| \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix}, 
  \begin{bmatrix} \Sigma_x & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_y \end{bmatrix} \right. \right)
\end{align*}
```

Since covariance matrices are by definition symmetric, it follows that ``\Sigma_x`` and ``\Sigma_y`` are symmetric and ``\Sigma_{xy} = \Sigma_{yx}^T``.

Let's factorize ``p(z) = p(x,y)`` as ``p(x,y) = p(y|x) p(x)`` through conditioning and marginalization.

##### conditioning
```math
\begin{equation*}
p(y|x) = \mathcal{N}\left(y\,|\,\mu_y + \Sigma_{yx}\Sigma_x^{-1}(x-\mu_x),\, \Sigma_y - \Sigma_{yx}\Sigma_x^{-1}\Sigma_{xy} \right)
\end{equation*}
```

##### marginalization
```math
\begin{equation*}
 p(x) = \mathcal{N}\left( x|\mu_x, \Sigma_x \right)
\end{equation*}
```

**proof**: in [Bishop](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) pp.87-89

Hence, conditioning and marginalization in Gaussians lead to Gaussians again. This is very useful for applications in Bayesian inference in jointly Gaussian systems.

With a natural parameterization of the Gaussian ``p(z) = \mathcal{N}_c(z|\eta,\Lambda)`` with precision matrix ``\Lambda = \Sigma^{-1} = \begin{bmatrix} \Lambda_x & \Lambda_{xy} \\ \Lambda_{yx} & \Lambda_y \end{bmatrix}``,  the conditioning operation results in a simpler result, see Bishop pg.90, eqs. 2.96 and 2.97. 

As an exercise, interpret the formula for the conditional mean (``\mathbb{E}[y|x]=\mu_y + \Sigma_{yx}\Sigma_x^{-1}(x-\mu_x)``) as a prediction-correction operation.

"""

# ╔═╡ b9a9565c-d294-11ef-1b67-83d1ab18035b
md"""
$(code_example("Joint, Marginal, and Conditional Gaussian Distributions"))

Let's plot the joint, marginal, and conditional distributions for some Gaussians.

"""

# ╔═╡ 59599e04-3e81-4518-b232-3264d9bde4f7
let

	# up_or_down_one_order_of_magnitude = 10 .^ (-1.0:0.1:1.0)
	range = [(0:.1:1)..., (1.2:.2:2)..., (2.5:.5:10)...]

	Σb11 = @bind example_Σ_11 Scrubbable(range; default=0.3)
	Σb12 = @bind example_Σ_12 Scrubbable(range; default=0.7)
	Σb22 = @bind example_Σ_22 Scrubbable(range; default=2.0)


	μb1 = @bind example_μ_1 Scrubbable(range; default=1.0)
	μb2 = @bind example_μ_2 Scrubbable(range; default=2.0)

	

	grid2(xs...) = @htl """<div style="display: inline-grid; grid-template-columns: auto auto;">$(xs)</div>"""
	
	
	a(s) = @htl """<span style="color: var(--cm-color-variable) !important; font-weight: 700;">$s</span>"""


	
	@htl """
	<code style="
		display: flex; 
		white-space: pre;
		align-items: center;
	"
	>$(a(:μ)) = $(
		grid2(μb1, μb2)
	), $(a(:Σ)) = $(
		grid2(Σb11, Σb12, Σb12, Σb22)
	)</code>
	
	
	
	"""
end

# ╔═╡ b9a99fcc-d294-11ef-3de4-5369d9796de7
let
	# Define the joint distribution p(x,y)
	μ = [example_μ_1, example_μ_2]
	Σ = [
		example_Σ_11 example_Σ_12
		example_Σ_12 example_Σ_22
	]

	ohno = [:😔, :😤, :😖, :🥶]

	try 
		
		
		joint = MvNormal(μ,Σ)
		
		# Define the marginal distribution p(x)
		marginal_x = Normal(μ[1], sqrt(Σ[1,1]))
		
		# Plot p(x,y)
		x_range = y_range = range(-2,stop=5,length=100)
		
		joint_pdf = [ pdf(joint, [x_range[i];y_range[j]]) for  j=1:length(y_range), i=1:length(x_range)]
		plot_1 = heatmap(x_range, y_range, joint_pdf, title = L"p(x, y)")
		
		# Plot p(x)
		plot_2 = plot(range(-2,stop=5,length=1000), pdf.(marginal_x, range(-2,stop=5,length=1000)), title = L"p(x)", label="", fill=(0, 0.1))
		
		# Plot p(y|x = 0.1)
		x = 0.1
		conditional_y_m = μ[2]+Σ[2,1]*inv(Σ[1,1])*(x-μ[1])
		conditional_y_s2 = Σ[2,2] - Σ[2,1]*inv(Σ[1,1])*Σ[1,2]
		conditional_y = Normal(conditional_y_m, sqrt.(conditional_y_s2))
		plot_3 = plot(range(-2,stop=5,length=1000), pdf.(conditional_y, range(-2,stop=5,length=1000)), title = L"p(y|x = %$x)", label="", fill=(0, 0.1))
	
		# Combined
		plot(plot_1, plot_2, plot_3, layout=(1,3), size=(1200,300))
		
	catch e
		str = sprint(showerror, e)
		Text("$str $(rand(ohno))")
	end

end

# ╔═╡ b9a9b8e0-d294-11ef-348d-c197c4ce2b8c
md"""
As is clear from the plots, the conditional distribution is a renormalized slice from the joint distribution.

"""

# ╔═╡ b9a9dca8-d294-11ef-04ec-a9202c319f89
md"""
## Gaussian Conditioning Revisited

Consider (again) the system 

```math
\begin{align*}
p(x\,|\,\theta) &= \mathcal{N}(x\,|\,\theta,\sigma^2) \\
p(\theta) &= \mathcal{N}(\theta\,|\,\mu_0,\sigma_0^2)
\end{align*}
```

"""

# ╔═╡ b9a9f98e-d294-11ef-193a-0dbdbfffa86f
md"""
Let ``z = \begin{bmatrix} x \\ \theta \end{bmatrix}``. The distribution for ``z`` is then given by (see [exercise below](#Conversion-to-Joint-Distribution-(**)))

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

# ╔═╡ b9aa27da-d294-11ef-0780-af9d89f9f599
md"""
Direct substitution of the rule for Gaussian conditioning leads to the $(HTML("<span id='precision-weighted-update'>posterior</span>")) (derivation as an Exercise):

```math
\begin{align*}
p(\theta|x) &= \mathcal{N} \left( \theta\,|\,\mu_1, \sigma_1^2 \right)\,,
\end{align*}
```

with

```math
\begin{align*}
K &= \frac{\sigma_0^2}{\sigma_0^2+\sigma^2} \qquad \text{($K$ is called: Kalman gain)}\\
\mu_1 &= \mu_0 + K \cdot (x-\mu_0)\\
\sigma_1^2 &= \left( 1-K \right) \sigma_0^2  
\end{align*}
```

Hence, for jointly Gaussian systems, inference can be performed in a single step using closed-form expressions for conditioning and marginalization of (multivariate) Gaussian distributions.
"""

# ╔═╡ b9acd5d4-d294-11ef-1ae5-ed4e13d238ef
md"""
## $(HTML("<span id='inference-for-precision'>Inference for the Precision Parameter of the Gaussian</span>"))



"""



# ╔═╡ b9acf7a8-d294-11ef-13d9-81758355cb1e
md"""

#### Problem



Consider again a Gaussian data-generating (measurement) model

```math
\mathcal{N}\left(x_n \,|\, \mu, \lambda^{-1} \right) \,.
```

(We express here the variance as the inverse of a precision parameter ``\lambda``, rather than using ``\sigma^2``, since this simplifies the subsequent Bayesian computations.)

Earlier in this lecture, we discussed Bayesian inference from a data set for the mean ``\mu``, when the variance ``\lambda^{-1}`` was given. 

We now derive the posterior distribution over the precision parameter ``\lambda``, assuming that the mean ``\mu`` is known. We omit the more general case in which both ``\mu`` and ``\lambda`` are treated as unknowns, since the resulting calculations are considerably more involved (but still result in a closed-form solution).


"""

# ╔═╡ b9ad0842-d294-11ef-2035-31bceab4ace1
md"""
#### model specification

The likelihood for the precision parameter is 

```math
\begin{align*}
p(D|\lambda) &= \prod_{n=1}^N \mathcal{N}\left(x_n \,|\, \mu, \lambda^{-1} \right) \\
  &\propto \lambda^{N/2} \exp\left\{ -\frac{\lambda}{2}\sum_{n=1}^N \left(x_n - \mu \right)^2\right\} \tag{B-2.145}
\end{align*}
```

"""

# ╔═╡ b9ad1b70-d294-11ef-3931-d1dcd2343ac9
md"""
The conjugate distribution for this function of ``\lambda`` is the [*Gamma* distribution](https://en.wikipedia.org/wiki/Gamma_distribution), given by

```math
p(\lambda\,|\,a,b) = \mathrm{Gam}\left( \lambda\,|\,a,b \right) \triangleq \frac{1}{\Gamma(a)} b^{a} \lambda^{a-1} \exp\left\{ -b \lambda\right\}\,, \tag{B-2.146}
```

where ``a>0`` and ``b>0`` are known as the *shape* and *rate* parameters, respectively. 

![](https://github.com/bmlip/course/blob/v2/assets/figures/B-fig-2.13.png?raw=true)

(Bishop fig.2.13). Plots of the Gamma distribution ``\mathrm{Gam}\left( \lambda\,|\,a,b \right)`` for different values of ``a`` and ``b``.

"""

# ╔═╡ b9ad299e-d294-11ef-36d7-2f73d3cd1fa7
md"""
The mean and variance of the Gamma distribution evaluate to ``\mathrm{E}\left( \lambda\right) = \frac{a}{b}`` and ``\mathrm{var}\left[\lambda\right] = \frac{a}{b^2}``. 

For this example, we consider a prior 
```math
p(\lambda) = \mathrm{Gam}\left( \lambda\,|\,a_0, b_0\right) \,. 
```

"""

# ╔═╡ b9ad5100-d294-11ef-0e8b-3f67ddb2d86d
md"""
#### inference

The posterior is given by Bayes rule, 

```math
\begin{align*}
p(\lambda\,|\,D) &\propto \underbrace{\lambda^{N/2} \exp\left\{ -\frac{\lambda}{2}\sum_{n=1}^N \left(x_n - \mu \right)^2\right\} }_{\text{likelihood}} \cdot \underbrace{\frac{1}{\Gamma(a_0)} b_0^{a_0} \lambda^{a_0-1} \exp\left\{ -b_0 \lambda\right\}}_{\text{prior}} \\
  &\propto \mathrm{Gam}\left( \lambda\,|\,a_N,b_N \right) 
\end{align*}
```

with

```math
\begin{align*}
a_N &= a_0 + \frac{N}{2} \qquad &&\text{(B-2.150)} \\
b_N &= b_0 + \frac{1}{2}\sum_n \left( x_n-\mu\right)^2 \qquad &&\text{(B-2.151)}
\end{align*}
```

"""

# ╔═╡ b9ad6238-d294-11ef-3fed-bbcc7d7443ee
md"""
Hence the **posterior is again a Gamma distribution**. By inspection of B-2.150 and B-2.151, we deduce that we can interpret ``2a_0`` as the number of a priori (pseudo-)observations. 

"""

# ╔═╡ b9ad71a6-d294-11ef-185f-f1f6e6ac4464
md"""
Since the most uninformative prior is given by ``a_0=b_0 \rightarrow 0``, we can derive the **maximum likelihood estimate** for the precision as

```math
\lambda_{\text{ML}} = \left.\mathrm{E}\left[ \lambda\right]\right\vert_{a_0=b_0\rightarrow 0} = \left. \frac{a_N}{b_N}\right\vert_{a_0=b_0\rightarrow 0} = \frac{N}{\sum_{n=1}^N \left(x_n-\mu \right)^2}
```

"""

# ╔═╡ b9ad85a4-d294-11ef-2af2-953ac0ab8927
md"""
In short, if we do density estimation with a Gaussian distribution ``\mathcal{N}\left(x_n\,|\,\mu,\sigma^2 \right)`` for an observed data set ``D = \{x_1, x_2, \ldots, x_N\}``, the $(HTML("<span id='ML-for-Gaussian'>maximum likelihood estimates</span>")) for ``\mu`` and ``\sigma^2`` are given by

```math
\begin{align*}
\mu_{\text{ML}} &= \frac{1}{N} \sum_{n=1}^N x_n \qquad &&\text{(B-2.121)} \\
\sigma^2_{\text{ML}} &= \frac{1}{N} \sum_{n=1}^N \left(x_n - \mu_{\text{ML}} \right)^2 \qquad &&\text{(B-2.122)}
\end{align*}
```

These estimates are also known as the *sample mean* and *sample variance* respectively. 

"""

# ╔═╡ b9abadce-d294-11ef-14a6-9131c5b1b802
md"""
## $(HTML("<span id='product-of-gaussians'>Product of Normally Distributed Variables</span>"))

(We've seen that) the sum of two Gausssian-distributed variables is also Gaussian distributed.

Has the *product* of two Gaussian distributed variables also a Gaussian distribution?

**No**! In general, this is a difficult computation. As an example, let's compute ``p(z)`` for ``Z=XY`` for the special case that ``X\sim \mathcal{N}(0,1)`` and ``Y\sim \mathcal{N}(0,1)``.

```math
\begin{align*}
p(z) &= \int_{X,Y} p(z|x,y)\,p(x,y)\,\mathrm{d}x\mathrm{d}y \\
  &= \frac{1}{2 \pi}\int  \delta(z-xy) \, e^{-(x^2+y^2)/2} \, \mathrm{d}x\mathrm{d}y \\
  &=  \frac{1}{\pi} \int_0^\infty \frac{1}{x} e^{-(x^2+z^2/x^2)/2} \, \mathrm{d}x \\
  &= \frac{1}{\pi} \mathrm{K}_0( \lvert z\rvert )\,.
\end{align*}
```

where  ``\mathrm{K}_n(z)`` is a [modified Bessel function of the second kind](http://mathworld.wolfram.com/ModifiedBesselFunctionoftheSecondKind.html).

"""

# ╔═╡ b9abdc7e-d294-11ef-394a-a708c96c86fc
md"""
$(code_example("Product of Gaussian Distributions"))


We plot ``p(Z=XY)`` and ``p(X)p(Y)`` for ``X\sim\mathcal{N}(0,1)`` and ``Y \sim \mathcal{N}(0,1)`` to give an idea of how these distributions differ.

"""


# ╔═╡ b9abf984-d294-11ef-1eaa-3358379f8b44
let
	X = Normal(0, 1)
	Y = Normal(0, 1)
	pdf_product_std_normals(z::Real) = besselk(0, abs(z))/π
	
	range1 = range(-4,stop=4,length=100)
	plot(range1, t -> pdf(X, t); label=L"p(X)=p(Y)=\mathcal{N}(0,1)", fill=(0, 0.1))
	plot!(range1, t -> pdf(X,t)*pdf(Y,t); label=L"p(X)*p(Y)", fill=(0, 0.1))
	plot!(range1, pdf_product_std_normals; label=L"p(Z=X*Y)", fill=(0, 0.1))
end

# ╔═╡ b9ac09c4-d294-11ef-2cb8-270289d01f25
md"""
In short, Gaussian-distributed variables remain Gaussian in linear systems, but this is not the case in non-linear systems. 

"""

# ╔═╡ f07d505a-5fc3-45fb-9a55-b8d397d1280e
md"""
## Multinomial Maximum Likelihood Estimation by Optimizing a Constrained Log-likelihood
"""

# ╔═╡ d66ab614-f672-490f-896c-565b70a21e48
md"""

The log-likelihood for the multinomial distribution is given by

```math
\begin{align*}
\mathrm{L}(\mu) &\triangleq \log p(D_m|\mu) \propto \log \prod_k \mu_k^{m_k} =  \sum_k m_k \log \mu_k 
\end{align*}
```

"""

# ╔═╡ bd4b4266-82c4-4355-afba-302d39f7c30b
md"""
When doing ML estimation, we must obey the constraint ``\sum_k \mu_k  = 1``, which can be accomplished by a [Lagrange multiplier](https://en.wikipedia.org/wiki/Lagrange_multiplier). The **constrained log-likelihood** with Lagrange multiplier is then

```math
\tilde{\mathrm{L}}(\mu) = \sum_k m_k \log \mu_k  + \lambda \cdot \big(1 - \sum_k \mu_k \big)
```

The method of Lagrange multipliers is a mathematical method for transforming a constrained optimization problem to an unconstrained optimization problem (see [Bishop App.E](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf#page=727)). Unconstrained optimization problems can be solved by setting the derivative to zero. 

"""

# ╔═╡ bc16a308-c04e-4416-9a62-c741bc4e9911
md"""
Setting the derivative of ``\tilde{\mathrm{L}}(\mu)`` to zero yields the **sample proportion** for ``\mu_k`` 

```math
\begin{equation*}
\nabla_{\mu_k}   \tilde{\mathrm{L}}(\mu) = \frac{m_k }
{\hat\mu_k } - \lambda  \overset{!}{=} 0 \; \Rightarrow \; \hat\mu_k = \frac{m_k }{N}
\end{equation*}
```

where we get ``\lambda`` from the constraint 

```math
\begin{equation*}
\sum_k \hat \mu_k = \sum_k \frac{m_k}
{\lambda} = \frac{N}{\lambda} \overset{!}{=}  1
\end{equation*}
```



"""

# ╔═╡ c3a886ee-4a10-485c-8b4c-2c205ed87cfb
navigate_prev_next(
	"https://bmlip.github.io/course/lectures/Factor%20Graphs.html",
	"https://bmlip.github.io/course/lectures/Regression.html",
)

# ╔═╡ f78bc1f5-cf7b-493f-9c5c-c2fbd6788616
md"""
# Code
"""

# ╔═╡ 026da6b9-dee1-485e-af00-3b9e35f71b6b
md"""
#### Introduction
"""

# ╔═╡ 6ffabd68-4c38-4024-a21b-1d6fa7c3a6d7
d(x; kwargs...) = PlutoUI.ExperimentalLayout.Div([x]; kwargs...)

# ╔═╡ 69ee7a67-84a8-4d90-9020-f0a76a7c5d58
begin
	intro_bonds = PlutoUI.ExperimentalLayout.Div([
	d(@bindname(N, Slider(3:100; default=90, show_value=true)); style="flex: 1 0 max-content"),
	d(@bind(redraw_button_clicked_count, CounterButton("Redraw x.")); style="flex: 1 1 50%")
];
	style="display: flex; flex-drection: row; flex-wrap: wrap;
    ")
end

# ╔═╡ 654909ff-2b56-4f81-97e3-a74e9028ada1
intro_bonds

# ╔═╡ ce16666b-aa90-42ae-b3a7-690e71301024
# macro StableRandom(x=nothing)
# 	:(MersenneTwister($(hash(__source__)) + hash($(esc(x)))))
# end

# ╔═╡ 724cac08-a54d-4dea-8416-0bce33c75405
stable_rand(args...; seed=nothing) = rand(MersenneTwister(543432 + hash(seed)), args...)

# ╔═╡ 92efa7c1-dde6-4b21-bf3b-0fa91931620c
secret_generative_dist = MvNormal([0,1.], [0.8 0.5; 0.5 1.0]);

# ╔═╡ ecac6e86-03f4-485e-b2ce-c502dec52cf5
D = stable_rand(secret_generative_dist, N)

# ╔═╡ 6e17e1e4-065e-47c9-bfd4-273d021b9ce0
x_dot = stable_rand(secret_generative_dist; seed=redraw_button_clicked_count)

# ╔═╡ 3d0f7af2-082d-4305-a271-349d41fcd166
md"""
#### Challenge solution
"""

# ╔═╡ eaf6794e-66a1-45f0-95ff-7d13983aafa2
baseplot() = plot(; xlim=(-3,3), ylim=(-2,3))

# ╔═╡ 5f19da92-9fe2-4607-89ae-3a0a98169bdc
let
	baseplot()
	scatter!(D[1,:], D[2,:], marker=:x, markerstrokewidth=3, label=L"D")
	scatter!([x_dot[1]], [x_dot[2]], label=L"x_\bullet")
	plot!(S[1], fill(S[2][1], 2), fillrange=S[2][2], alpha=0.4, color=:gray,label=L"S")
end

# ╔═╡ bfab8dd0-b69e-4078-abb7-868e5f923a79
let
	baseplot()
	
	# Maximum likelihood estimation of 2D Gaussian
	μ = 1/N * sum(D,dims=2)[:,1]
	D_min_μ = D - repeat(μ, 1, N)
	Σ = Hermitian(1/N * D_min_μ*D_min_μ')
	global m = MvNormal(μ, convert(Matrix, Σ));
	
	contour!(range(-3, 4, length=100), range(-3, 4, length=100), (x, y) -> pdf(m, [x, y]))
	scatter!(D[1,:], D[2,:]; marker=:x, markerstrokewidth=3, label=L"D")
	scatter!([x_dot[1]], [x_dot[2]]; label=L"x_\bullet")
	plot!(range(0, 2), [1., 1., 1.]; fillrange=2, alpha=0.4, color=:gray, label=L"S")
end

# ╔═╡ eadb40d9-1f89-4047-9d50-978603589925
let
	# We can use HCubature.jl to numerically evaluate the integral and get a good approximation.

	(val,err) = hcubature(
		(x)->pdf(m,x), # function to integrate
		first.(S), last.(S), # start and end coordinates
	)
	
	@mdx "Answer: ``p(x_⋅ ∈ S | m) ≈ $(round(val; digits=4))``"
end

# ╔═╡ bc7a875f-e4fa-43fd-b001-cec6aadea3bc
md"""
#### Packages
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BmlipTeachingTools = "656a7065-6f73-6c65-7465-6e646e617262"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HCubature = "19dc6840-f33b-545b-b366-655c7e3ffd49"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MarkdownLiteral = "736d6165-7244-6769-4267-6b50796e6954"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[compat]
BmlipTeachingTools = "~1.4.1"
Distributions = "~0.25.127"
HCubature = "~1.8.0"
LaTeXStrings = "~1.4.0"
MarkdownLiteral = "~0.1.5"
Plots = "~1.41.6"
SpecialFunctions = "~2.8.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.7"
manifest_format = "2.0"
project_hash = "3220e8fff80e0d34fd1cf8d4c692796b1ec07cbe"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "bbe1079eecf9c9fbb52765193ad2bae27ae09bc8"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.10"

[[deps.BmlipTeachingTools]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "PlutoTeachingTools", "PlutoUI", "Reexport"]
git-tree-sha1 = "721865ca80c702e053b7d3958c5de5295ad84eca"
uuid = "656a7065-6f73-6c65-7465-6e646e617262"
version = "1.4.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Combinatorics]]
git-tree-sha1 = "c761b00e7755700f9cdf5b02039939d1359330e1"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.1.0"

[[deps.CommonMark]]
deps = ["PrecompileTools"]
git-tree-sha1 = "731edf630eb23140a3c7a2e74d4186f562996d76"
uuid = "a80b9123-70ca-4bc0-993e-6e3bcb318db6"
version = "1.0.2"

    [deps.CommonMark.extensions]
    CommonMarkMarkdownASTExt = "MarkdownAST"
    CommonMarkMarkdownExt = "Markdown"

    [deps.CommonMark.weakdeps]
    Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
    MarkdownAST = "d0879d2d-cac2-40c8-9cee-1863dc0c7391"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.1+2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "21d088c496ea22914fe80906eb5bce65755e5ec8"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.1"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6fb53a69613a0b2b68a0d12671717d307ab8b24e"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.5"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3c8a0a9a6d4a10bdfb6b751bd2b6051ed3e25fd4"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.127"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c307cd83373868391f3ac30b41530bc5d5d05d08"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.1+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "9e0fb9e54594c47f278d75063980e43066e26e20"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "f954322d5de03ec630d177cda203dcd92b6be399"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.26"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "6fada551286ab6ea4ca1628cb2de9f166a2ec966"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.26+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HCubature]]
deps = ["Combinatorics", "DataStructures", "LinearAlgebra", "QuadGK", "StaticArrays"]
git-tree-sha1 = "8ee627fb73ecba0b5254158b04d4745611b404a1"
uuid = "19dc6840-f33b-545b-b366-655c7e3ffd49"
version = "1.8.0"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c89d196f5ffb64bfbf80985b699ea913b0d2c211"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "bba2d9aa057d8f126415de240573e86a8f39d2a1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.1"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MarkdownLiteral]]
deps = ["CommonMark", "HypertextLiteral"]
git-tree-sha1 = "e88f9af659a0cc9326fa464427f71ae6c9a83381"
uuid = "736d6165-7244-6769-4267-6b50796e6954"
version = "0.1.5"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "a680e58292816791211582a4d8e44353835e991f"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.38"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "32a4e09c5f29402573d673901778a0e03b0807b9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.6"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "cb20a4eacda080e517e4deb9cfb6c7c518131265"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.6"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "90b41ced6bacd8c01bd05da8aed35c5458891749"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "6547cbdd8ce32efba0d21c5a40fa96d1a3548f9f"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.8.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "e4d7a1a0edc20af42689ea6f4f3587a2175d50ee"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.12"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "770240df9a3b8888065046948f7a09b4e0f997d5"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "ed349d26affcacafbc7fc2941ace1fb98f71e715"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+1"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╟─b9a38e20-d294-11ef-166b-b5597125ed6d
# ╟─5e9a51b1-c6e5-4fb5-9df3-9b189f3302e8
# ╟─b9a46c3e-d294-11ef-116f-9b97e0118e5b
# ╟─6f35e623-7e53-4fc7-a0be-5e9dc6f378b4
# ╟─7001ad0f-c3bd-424f-91db-d18c5a4c15f9
# ╟─69ee7a67-84a8-4d90-9020-f0a76a7c5d58
# ╟─5f19da92-9fe2-4607-89ae-3a0a98169bdc
# ╟─61624ee2-948f-464f-a1f7-ea95bcac7881
# ╠═ecac6e86-03f4-485e-b2ce-c502dec52cf5
# ╟─5194faba-2db0-4a28-9151-3e6582b1036a
# ╠═6e17e1e4-065e-47c9-bfd4-273d021b9ce0
# ╟─b20fa50e-ef09-43fc-93d7-76bf91cce696
# ╟─01c050ed-e29e-431b-bfd6-f2fdcb9687fd
# ╠═e12d7cf3-7e78-4472-8efa-528223668661
# ╟─6adc446b-1e00-4a98-b68d-3e356682341d
# ╟─71f1c8ee-3b65-4ef8-b36f-3822837de410
# ╟─b9a4eb62-d294-11ef-06fa-af1f586cbc15
# ╟─b9a50d0c-d294-11ef-0e60-2386cf289478
# ╟─b9a52b18-d294-11ef-2d42-19c5e3ef3549
# ╟─b9a5589a-d294-11ef-3fc3-0552a69df7b2
# ╟─085233ee-f5ad-4731-89bb-84773182bba6
# ╟─4632225a-cbfc-4ec9-a978-21037d8253b6
# ╟─9501922f-b928-46e2-8f23-8eb9c64f6198
# ╟─b9a5889c-d294-11ef-266e-d90225222e10
# ╟─56510a09-073c-4fc8-b0b7-17b20dbb95f0
# ╟─a82378ae-d1be-43f9-b63a-2f897767d1fb
# ╟─36eff7bc-72f2-4b48-a109-1861af6834aa
# ╟─87f400ac-36f2-4778-a3ba-06dd7652e279
# ╟─9c2bf0a2-4bb6-4769-b47b-6a02c4e73044
# ╟─8f7ecb91-d251-4ac9-bb32-0dd7215382e3
# ╟─883e8244-270e-4c6c-874b-b69d8989c24c
# ╟─f02aa0b1-2261-4f65-9bd0-3be33230e0d6
# ╟─f008a742-6900-4e18-ab4e-b5da53fb64a6
# ╟─75e35350-af22-42b1-bb55-15e16cb9c375
# ╟─8d2732e8-479f-4744-9b1f-d0364f0c6488
# ╟─0f9feb8d-971e-4a94-8c70-3e1f0d284314
# ╟─2767b364-6f9a-413d-aa9e-88741cd2bbb1
# ╟─c6753ff3-7b5e-45b8-8adc-e0bbaa6be7d3
# ╟─b9a5cbc2-d294-11ef-214a-c71fb1272326
# ╟─b9a5dcc0-d294-11ef-2c85-657a460db5cd
# ╟─7b415578-10fa-4eb1-ab1f-ce3ff57dcf45
# ╟─b9a67d06-d294-11ef-297b-eb9039786ea7
# ╟─b9a68d3a-d294-11ef-2335-093a39648007
# ╟─b9a697fa-d294-11ef-3a57-7b7ba1f4fd70
# ╟─b9a6b7b2-d294-11ef-06dc-4de5ef25c1fd
# ╟─702e7b10-14a4-42da-a192-f7c02a3d470a
# ╟─51d81901-213f-42ce-b77e-10f7ca4a4145
# ╟─b9a6c7b6-d294-11ef-0446-c372aa610df8
# ╟─b9a6ecd2-d294-11ef-02af-37c977f2814b
# ╟─b9a6f916-d294-11ef-38cb-b78c0c448550
# ╟─d2bedf5f-a0ea-4604-b5da-adf9f11e80be
# ╟─b9a7073a-d294-11ef-2330-49ffa7faff21
# ╟─45c2fb37-a078-4284-9e04-176156cffb1e
# ╟─df8867ed-0eff-4a52-8f5e-2472467e1aa2
# ╟─3a0f7324-0955-4c1c-8acc-0d33ebd16f78
# ╟─db730ca7-4850-49c7-a93d-746d393b509b
# ╟─4dfa6aa8-2ba1-4ee8-8488-0e11d4132891
# ╟─b9a80522-d294-11ef-39d8-53a536d66bf9
# ╟─87bce1d2-e473-4eb8-a876-1475bf841d70
# ╟─6efa81ff-6445-4da6-a297-03d7b23f0c4b
# ╟─922f0eb6-9e29-4b6c-9701-cb7b2f07bb7a
# ╟─ffdfe355-1c08-4b94-97f3-f9df2b8325e5
# ╟─f4735ea4-b2b8-4021-a66a-976ff6639653
# ╟─8cc63234-0553-471e-9ad8-f44efefc5f3a
# ╟─fd2a4003-1362-43f6-9bbe-ce667e4e4611
# ╟─83c12d79-9fe3-4b5e-87b7-31174d0849cf
# ╟─bf52bd62-44f2-4bcf-acc9-e4180c0dbc28
# ╟─21e9f9bd-71df-4fc0-adca-d4dc5835584a
# ╟─52cd2c58-62a6-43ee-9188-2e23ea4037ed
# ╟─d0751756-8550-4d49-9cd7-884c2fbcd77b
# ╟─7908e67b-f390-4636-9d9d-2f8b73dc4b3a
# ╟─dfde69ed-6ad3-4d09-bb59-76e7aebb9868
# ╟─33484269-19b0-4f01-ab83-a94fc68eace3
# ╟─2fc84689-4f84-4a1d-b5a1-eeb6adf3adeb
# ╟─53a1e971-d774-46fb-b328-4cd71585ee75
# ╟─55c101f3-90f3-4a1a-9e65-72a35db254a9
# ╟─f7ae238e-8a9f-4510-85f7-e7cdae100f99
# ╟─f8b57f11-f014-4a4a-aa41-8217b1a5b21d
# ╟─bed4962d-cd5f-4bff-bf25-68e524b20183
# ╟─bed7072c-53b0-47f7-bf77-32deed55b3e5
# ╟─71c09b1e-de5c-404d-8099-965433e85481
# ╟─eb901f4d-2ecd-44fb-baf2-80e65489186e
# ╟─654909ff-2b56-4f81-97e3-a74e9028ada1
# ╟─bfab8dd0-b69e-4078-abb7-868e5f923a79
# ╟─eadb40d9-1f89-4047-9d50-978603589925
# ╟─b89360b8-39fa-46e9-96c8-7eece50fcb90
# ╟─a439c0a7-afa1-4d9a-8737-58d341744016
# ╟─79a99a22-3bb5-431b-bf84-5dce5cccfe25
# ╟─14b3edcc-0d16-4055-9b1c-7f324514a0a9
# ╟─dd7786e2-d6ac-4dba-abca-3686242c067d
# ╟─b7a810a3-dc38-4e72-ab10-2ad2f064bdbb
# ╟─f711b053-dccf-4bf1-b285-e8da94a48b68
# ╟─1df7a10d-c4f6-40d6-8f5a-cbd79ef1d415
# ╟─673360e8-27ed-471c-a866-15af550df5e7
# ╟─22539cfe-3694-4100-8120-ca6ac1e66b31
# ╟─fa197526-6706-47ce-b84b-5675eee00610
# ╟─645308ac-c9e3-4d6f-bcff-82327fbb8edf
# ╟─03c399e1-d0d8-493a-9f95-4209918d132a
# ╟─32a7da22-7dba-41eb-8125-a9c8409a968e
# ╟─a0a96322-d71e-4e5c-95a4-fcc01c0db542
# ╟─ff9e3293-0a49-4c51-b118-cf7af45e4bcb
# ╟─fded7c8c-feed-4630-a50d-5a854536cab3
# ╟─6dfc31a0-d0d7-4901-a876-890df9ab4258
# ╟─b9a885a8-d294-11ef-079e-411d3f1cda03
# ╟─b9a9565c-d294-11ef-1b67-83d1ab18035b
# ╟─59599e04-3e81-4518-b232-3264d9bde4f7
# ╟─b9a99fcc-d294-11ef-3de4-5369d9796de7
# ╟─b9a9b8e0-d294-11ef-348d-c197c4ce2b8c
# ╟─b9a9dca8-d294-11ef-04ec-a9202c319f89
# ╟─b9a9f98e-d294-11ef-193a-0dbdbfffa86f
# ╟─b9aa27da-d294-11ef-0780-af9d89f9f599
# ╟─b9acd5d4-d294-11ef-1ae5-ed4e13d238ef
# ╟─b9acf7a8-d294-11ef-13d9-81758355cb1e
# ╟─b9ad0842-d294-11ef-2035-31bceab4ace1
# ╟─b9ad1b70-d294-11ef-3931-d1dcd2343ac9
# ╟─b9ad299e-d294-11ef-36d7-2f73d3cd1fa7
# ╟─b9ad5100-d294-11ef-0e8b-3f67ddb2d86d
# ╟─b9ad6238-d294-11ef-3fed-bbcc7d7443ee
# ╟─b9ad71a6-d294-11ef-185f-f1f6e6ac4464
# ╟─b9ad85a4-d294-11ef-2af2-953ac0ab8927
# ╟─b9abadce-d294-11ef-14a6-9131c5b1b802
# ╟─b9abdc7e-d294-11ef-394a-a708c96c86fc
# ╟─b9abf984-d294-11ef-1eaa-3358379f8b44
# ╟─b9ac09c4-d294-11ef-2cb8-270289d01f25
# ╟─f07d505a-5fc3-45fb-9a55-b8d397d1280e
# ╟─d66ab614-f672-490f-896c-565b70a21e48
# ╟─bd4b4266-82c4-4355-afba-302d39f7c30b
# ╟─bc16a308-c04e-4416-9a62-c741bc4e9911
# ╟─c3a886ee-4a10-485c-8b4c-2c205ed87cfb
# ╟─f78bc1f5-cf7b-493f-9c5c-c2fbd6788616
# ╟─026da6b9-dee1-485e-af00-3b9e35f71b6b
# ╠═6ffabd68-4c38-4024-a21b-1d6fa7c3a6d7
# ╠═ce16666b-aa90-42ae-b3a7-690e71301024
# ╠═724cac08-a54d-4dea-8416-0bce33c75405
# ╠═92efa7c1-dde6-4b21-bf3b-0fa91931620c
# ╟─3d0f7af2-082d-4305-a271-349d41fcd166
# ╠═5638c1d0-db95-49e4-bd80-528f79f2947e
# ╠═eaf6794e-66a1-45f0-95ff-7d13983aafa2
# ╠═03a36e87-2378-4efc-bcac-9c0609b52784
# ╟─bc7a875f-e4fa-43fd-b001-cec6aadea3bc
# ╠═c97c495c-f7fe-4552-90df-e2fb16f81d15
# ╠═4484429b-5f31-4a8c-89ed-2f67a1ac869e
# ╠═3ec821fd-cf6c-4603-839d-8c59bb931fa9
# ╠═00482666-0772-4e5d-bb35-df7b6fb67a1b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
