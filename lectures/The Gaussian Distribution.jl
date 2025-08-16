### A Pluto.jl notebook ###
# v0.20.13

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

# ╔═╡ 9edd80d4-d088-4b2f-8843-abaa7a5d9c5e
using Random

# ╔═╡ c97c495c-f7fe-4552-90df-e2fb16f81d15
using PlutoUI, PlutoTeachingTools

# ╔═╡ 3ec821fd-cf6c-4603-839d-8c59bb931fa9
using Distributions, Plots, LaTeXStrings

# ╔═╡ b9abf984-d294-11ef-1eaa-3358379f8b44
begin
  using SpecialFunctions
  let
	X = Normal(0, 1)
	Y = Normal(0, 1)
	pdf_product_std_normals(z::Real) = besselk(0, abs(z))/π
	
	range1 = range(-4,stop=4,length=100)
	plot(range1, t -> pdf(X, t); label=L"p(X)=p(Y)=\mathcal{N}(0,1)", fill=(0, 0.1))
	plot!(range1, t -> pdf(X,t)*pdf(Y,t); label=L"p(X)*p(Y)", fill=(0, 0.1))
	plot!(range1, pdf_product_std_normals; label=L"p(Z=X*Y)", fill=(0, 0.1))
  end
end

# ╔═╡ 00482666-0772-4e5d-bb35-df7b6fb67a1b
using HypertextLiteral

# ╔═╡ b9a38e20-d294-11ef-166b-b5597125ed6d
md"""
# Continuous Data and the Gaussian Distribution

"""

# ╔═╡ 5e9a51b1-c6e5-4fb5-9df3-9b189f3302e8
PlutoUI.TableOfContents()

# ╔═╡ b9a46c3e-d294-11ef-116f-9b97e0118e5b
md"""
## Preliminaries

##### Goal 

  * Review of information processing with Gaussian distributions in linear systems

##### Materials        

  * Mandatory

      * These lecture notes
  * Optional

      * [Bishop PRML book](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) (2006), pp. 85-93

      * [MacKay - 2006 - The Humble Gaussian Distribution](https://github.com/bmlip/course/blob/main/assets/files/Mackay-2006-The-humble-Gaussian-distribution.pdf) (highly recommended!)
      * [Ariel Caticha - 2012 - Entropic Inference and the Foundations of Physics](https://github.com/bmlip/course/blob/main/assets/files/Caticha-2012-Entropic-Inference-and-the-Foundations-of-Physics.pdf), pp.30-34, section 2.8, the Gaussian distribution
  * References

      * [E.T. Jaynes - 2003 - Probability Theory, The Logic of Science](http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf) (best book available on the Bayesian view on probability theory)

"""

# ╔═╡ 82025c2f-a21f-4080-b301-3ffe3715442d
section_outline("Challenge:", "Classify a Gaussian Sample" , color= "red" )

# ╔═╡ b9a48c60-d294-11ef-3b90-03053fcd82fb
md"""

Consider a data set as shown in the figure below

"""


# ╔═╡ ba57ecbb-b64e-4dd8-8398-a90af1ac71f3
begin
	N = 100;
	generative_dist = MvNormal([0,1.], [0.8 0.5; 0.5 1.0]);
	D = rand(generative_dist, N);
	x_dot = rand(generative_dist);
	
	let
		scatter(D[1,:], D[2,:], marker=:x, markerstrokewidth=3, label=L"D")
		scatter!([x_dot[1]], [x_dot[2]], label=L"x_\bullet")
		plot!(range(0, 2), [1., 1., 1.], fillrange=2, alpha=0.4, color=:gray,label=L"S")
	end
end

# ╔═╡ 02853a5c-f6aa-4af8-8a25-bfffd4b96afc
md"""

##### Problem 

- Consider a set of observations ``D=\{x_1,…,x_N\}`` in the 2-dimensional plane (see Figure). All observations were generated using the same process. We now draw an extra observation ``x_\bullet = (a,b)`` from the same data-generating process. What is the probability that ``x_\bullet`` lies within the shaded rectangle ``S = \{ (x,y) \in \mathbb{R}^2 | 0 \leq x \leq 2, 1 \leq y \leq 2 \} ``?


##### Solution 

- See later in this lecture. 
"""

# ╔═╡ 71f1c8ee-3b65-4ef8-b36f-3822837de410
md"""
# The Gaussian Distribution
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
        * As an example, see [Jaynes, section 7.1.4](http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf#page=250) for how this leads to the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem), which results from performing convolution operations on distributions.


2. Once the Gaussian has been attained, this form tends to be preserved. e.g.,   

    * The convolution of two Gaussian functions is another Gaussian function (useful in the sum of 2 variables and linear transformations)
    * The product of two Gaussian functions is another Gaussian function (useful in Bayes rule).
    * The Fourier transform of a Gaussian function is another Gaussian function.

See also [Jaynes, section 7.14](http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf#page=250), and the whole chapter 7 in his book for more details on why the Gaussian distribution is so useful.

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
p(z) = \mathcal{N} \left(z \,|\, A\mu_x+b, A\Sigma_x A^T \right) \,. \tag{SRG-4a}
```

In case ``x`` is not Gaussian, higher order moments may be needed to specify the distribution for ``z``. 


"""

# ╔═╡ 56510a09-073c-4fc8-b0b7-17b20dbb95f0
section_outline("Exercises:", "Linear Transformations" , color= "yellow" )

# ╔═╡ a82378ae-d1be-43f9-b63a-2f897767d1fb
md"""
##### The Sum of Gaussian Variables 

A commonly occurring example of a linear transformation is the *sum of two independent Gaussian variables*:

Let ``x \sim \mathcal{N} \left(\mu_x, \sigma_x^2 \right)`` and ``y \sim \mathcal{N} \left(\mu_y, \sigma_y^2 \right)``. Proof that the PDF for ``z=x+y`` is given by

```math
p(z) = \mathcal{N} \left(z\,|\,\mu_x+\mu_y, \sigma_x^2 +\sigma_y^2 \right) \tag{SRG-8}
```


"""

# ╔═╡ 36eff7bc-72f2-4b48-a109-1861af6834aa
details("Click for proof",
md"""	   
First, recognize that ``z=x+y`` can be written as a linear transformation ``z=A w``, where
```math
A = \begin{bmatrix} 1 & 1\end{bmatrix}
```	
and
```math
w = \begin{bmatrix} x \\ y\end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} x \\ y\end{bmatrix}, \begin{bmatrix} \sigma_x^2 & 0 \\ 0 & \sigma_y^2\end{bmatrix}\right) \,.
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
details("Click for answer",
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
details("click for answer",
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
&= a\cdot(\sigma_x^2 - 2 \underbrace{\sigma_{xy}}_{-0} + \sigma_y^2) \cdot a^T \\ 
&= a^2\cdot(\sigma_x^2 + \sigma_y^2)
\end{align}
```

		
		
		"""		
	   )

# ╔═╡ 9eb3e920-fab5-4a6a-8fe1-5734ebc6b25c
md"""
# Maximum Likelihood Estimation
"""

# ╔═╡ 883e8244-270e-4c6c-874b-b69d8989c24c

md"""

## MLE for a Gaussian

We are given an IID data set ``D = \{x_1,x_2,\ldots,x_N\}``, where ``x_n \in \mathbb{R}^M``. Assume that the data were drawn from a multivariate Gaussian (MVG) 

```math 
p(x_n|\theta) = \mathcal{N}(x_n|\,\mu,\Sigma) \,.
```

Let us derive the maximum likelihood estimates for the parameters ``\mu`` and ``\Sigma``.
"""

# ╔═╡ f02aa0b1-2261-4f65-9bd0-3be33230e0d6
md"""

##### Evaluation of log-likelihood function
Let ``\theta =\{\mu,\Sigma\}``. Proof that the log-likelihood (LLH) function ``\log p(D|\theta)`` can be worked out to

```math
\log p(D|\theta) =
 \frac{N}{2}\log  |\Sigma|^{-1} - \frac{1}{2}\sum_n (x_n-\mu)^T \Sigma^{-1}(x_n-\mu)

```
			
"""

# ╔═╡ f008a742-6900-4e18-ab4e-b5da53fb64a6
details("click to see proof",
		
		md" ```math
\begin{align*}
\log p(D|\theta) &= \log \prod_n p(x_n|\theta) \\
 &= \log \prod_n \mathcal{N}(x_n|\mu, \Sigma) \\
&= \log \prod_n (2\pi)^{-M/2} |\Sigma|^{-1/2} \exp\left\{ -\frac{1}{2}(x_n-\mu)^T \Sigma^{-1}(x_n-\mu)\right\} \\
&= \sum_n \left( \log (2\pi)^{-M/2} + \log  |\Sigma|^{-1/2} -\frac{1}{2}(x_n-\mu)^T \Sigma^{-1}(x_n-\mu)\right) \\
&\propto \frac{N}{2}\log  |\Sigma|^{-1} - \frac{1}{2}\sum_n (x_n-\mu)^T \Sigma^{-1}(x_n-\mu)
\end{align*}
```
"	   )

# ╔═╡ 75e35350-af22-42b1-bb55-15e16cb9c375
md"""
##### Maximum likelihood estimate of mean

Proof that the maximum likelihood estimate of the mean is given by
```math
\hat{\mu} = \frac{1}{N}\sum_n x_n \,.
```

"""

# ╔═╡ 8d2732e8-479f-4744-9b1f-d0364f0c6488
details("click to see proof",		
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

Since the map ``Ax=0`` for general ``A`` can only be true if ``x=0``, it follows that setting the gradient to ``0`` leads to 
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
details("click to see proof",		
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
# Simple Bayesian Inference
"""

# ╔═╡ b9a5cbc2-d294-11ef-214a-c71fb1272326
md"""
## Bayesian Inference for Estimation of a Constant

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
    p(x|\theta) &=  \mathcal{N}(x|\theta,\sigma^2) \\
    p(\theta) &=\mathcal{N}(\theta|\mu_0,\sigma_0^2)
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

As we just saw, a Gaussian prior, combined with a Gaussian likelihood, makes Bayesian inference analytically solvable (!), since 

```math
\begin{equation*}
\underbrace{\text{Gaussian}}_{\text{posterior}}
 \propto \underbrace{\text{Gaussian}}_{\text{likelihood}} \times \underbrace{\text{Gaussian}}_{\text{prior}} \,.
\end{equation*}
```


"""

# ╔═╡ 702e7b10-14a4-42da-a192-f7c02a3d470a
md"""
When applying Bayes rule, if the posterior distribution belongs to the same family as the prior (e.g., both are Gaussian distributions), we say that the prior and the likelihood form a conjugate pair.
"""

# ╔═╡ 51d81901-213f-42ce-b77e-10f7ca4a4145

keyconcept("", md"In Bayesian inference, a Gaussian prior distribution is **conjugate** to a Gaussian likelihood (when the variance is known), which ensures that the posterior distribution remains Gaussian. This conjugacy greatly simplifies calculation of Bayes rule.")


# ╔═╡ b9a6c7b6-d294-11ef-0446-c372aa610df8
md"""

## (Multivariate) Gaussian Multiplication


$(HTML("<span id='Gaussian-multiplication'></span>")) In general, the multiplication of two multi-variate Gaussians over ``x`` yields an (unnormalized) Gaussian over ``x``:

```math
\begin{equation*}
\mathcal{N}(x|\mu_a,\Sigma_a) \cdot \mathcal{N}(x|\mu_b,\Sigma_b) = \underbrace{\mathcal{N}(\mu_a|\, \mu_b, \Sigma_a + \Sigma_b)}_{\text{normalization constant}} \cdot \mathcal{N}(x|\mu_c,\Sigma_c) \tag{SRG-6}
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

# ╔═╡ 93361b31-022f-46c0-b80d-b34f3ed61d5f
md"""
## Gaussian Distributions in Julia
Take a look at this mini lecture to see some simple examples of using distributions in Julia:
"""

# ╔═╡ bbf3a1e7-9f25-434c-95c7-898648b5bc90
NotebookCard("https://bmlip.github.io/course/minis/Distributions%20in%20Julia.html")

# ╔═╡ b9a7073a-d294-11ef-2330-49ffa7faff21
md"""
$(section_outline("Code Example:", "Product of Two Gaussian PDFs"))

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
$(section_outline("Code Example:", "Joint, Marginal, and Conditional Gaussian Distributions"))

Let's plot the joint, marginal, and conditional distributions for some Gaussians.

"""

# ╔═╡ b9a99fcc-d294-11ef-3de4-5369d9796de7
let
	# Define the joint distribution p(x,y)
	μ = [1.0; 2.0]
	Σ = [0.3 0.7;
	     0.7 2.0]
	joint = MvNormal(μ,Σ)
	
	# Define the marginal distribution p(x)
	marginal_x = Normal(μ[1], sqrt(Σ[1,1]))
	
	# Plot p(x,y)
	x_range = y_range = range(-2,stop=5,length=1000)
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

"""

# ╔═╡ b9aa3950-d294-11ef-373f-d5d330694bfd

keyconcept("", md"For jointly Gaussian systems, inference can be performed in a single step using closed-form expressions for conditioning and marginalization of (multivariate) Gaussian distributions.")


# ╔═╡ b426f9c8-4506-43ef-92fa-2ee30be621ca
md"""
# Inference with Multiple Observations
"""


# ╔═╡ b9a80522-d294-11ef-39d8-53a536d66bf9

md"""

## Estimation of a Constant

#### model specification

Now consider that we measure a data set ``D = \{x_1, x_2, \ldots, x_N\}``, with measurements

```math
\begin{aligned}
x_n &= \theta + \epsilon_n \\
\epsilon_n &\sim \mathcal{N}(0,\sigma^2) \,,
\end{aligned}
```

and the same prior for ``\theta``:

```math
\theta \sim \mathcal{N}(\mu_0,\sigma_0^2) \\
```

Let's derive the predictive distribution ``p(x_{N+1}|D)`` for the next sample. 


#### inference

First, we derive the posterior for ``\theta``:

```math
\begin{align*}
p(\theta|D) \propto  \underbrace{\mathcal{N}(\theta|\mu_0,\sigma_0^2)}_{\text{prior}} \cdot \underbrace{\prod_{n=1}^N \mathcal{N}(x_n|\theta,\sigma^2)}_{\text{likelihood}} \,.
\end{align*}
```

Since the posterior is formed by multiplying ``N+1`` Gaussian distributions in ``\theta``, the result is also Gaussian in ``\theta``, due to the closure of the Gaussian family under multiplication (up to a normalization constant).

Using the property that precisions and precision-weighted means add when Gaussians are multiplied, we can immediately write the posterior as

```math
p(\theta|D) = \mathcal{N} (\theta |\, \mu_N, \sigma_N^2)
```

where 

```math
\begin{align*}
  \frac{1}{\sigma_N^2}  &= \frac{1}{\sigma_0^2} + \sum_n \frac{1}{\sigma^2}  \tag{B-2.142} \\
  \mu_N   &= \sigma_N^2 \, \left( \frac{1}{\sigma_0^2} \mu_0 + \sum_n \frac{1}{\sigma^2} x_n  \right) \tag{B-2.141}
\end{align*}
```


"""

# ╔═╡ 364cd002-92ee-4fb6-b89a-3251eff7502c
md"""
#### application: prediction of future sample

With the posterior over the model parameters in hand, we can now evaluate the posterior predictive distribution for the next sample ``x_{N+1}``. Proof for yourself that

```math
\begin{align*}
 p(x_{N+1}|D) &= \int p(x_{N+1}|\theta) p(\theta|D)\mathrm{d}\theta \\
  &=\mathcal{N}(x_{N+1}|\mu_N, \sigma^2_N +\sigma^2 )
\end{align*}
```

Note that uncertainty about ``x_{N+1}`` involves both uncertainty about the parameter (``\sigma_N^2``) and observation noise ``\sigma^2``.

"""

# ╔═╡ 922f0eb6-9e29-4b6c-9701-cb7b2f07bb7a
details("Click for solution",
md"""
```math
\begin{align*}
 p(x_{N+1}|D) &= \int p(x_{N+1}|\theta) p(\theta|D)\mathrm{d}\theta \\
  &= \int \mathcal{N}(x_{N+1}|\theta,\sigma^2) \mathcal{N}(\theta|\mu_N,\sigma^2_N) \mathrm{d}\theta \\
  &\stackrel{1}{=} \int \mathcal{N}(\theta|x_{N+1},\sigma^2) \mathcal{N}(\theta|\mu_N,\sigma^2_N) \mathrm{d}\theta \\
  &\stackrel{2}{=} \int  \mathcal{N}(x_{N+1}|\mu_N, \sigma^2_N +\sigma^2 ) \mathcal{N}(\theta|\cdot,\cdot)\mathrm{d}\theta \tag{use SRG-6} \\
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

# ╔═╡ 9bd38e28-73d4-4c6c-a1fe-35c7a0e750b3
section_outline("Challenge Revisited:", "Classify a Gaussian Sample", header_level=2, color="red")

# ╔═╡ b9ac2d3c-d294-11ef-0d37-65a65525ad28
md"""

Let's solve the challenge from the beginning of the lecture. We apply maximum likelihood estimation to fit a 2-dimensional Gaussian model (``m``) to data set ``D``. Next, we evaluate ``p(x_\bullet \in S | m)`` by (numerical) integration of the Gaussian pdf over ``S``: ``p(x_\bullet \in S | m) = \int_S p(x|m) \mathrm{d}x``.

"""

# ╔═╡ b9ac5190-d294-11ef-0a99-a9d369b34045
let
	# Maximum likelihood estimation of 2D Gaussian
	N = length(sum(D,dims=1))
	μ = 1/N * sum(D,dims=2)[:,1]
	D_min_μ = D - repeat(μ, 1, N)
	Σ = Hermitian(1/N * D_min_μ*D_min_μ')
	m = MvNormal(μ, convert(Matrix, Σ));
	
	contour(range(-3, 4, length=100), range(-3, 4, length=100), (x, y) -> pdf(m, [x, y]))
	
	# Numerical integration of p(x|m) over S:
	(val,err) = hcubature((x)->pdf(m,x), [0., 1.], [2., 2.])
	@debug("p(x⋅∈S|m) ≈ $(val)")
	
	scatter!(D[1,:], D[2,:]; marker=:x, markerstrokewidth=3, label=L"D")
	scatter!([x_dot[1]], [x_dot[2]]; label=L"x_\bullet")
	plot!(range(0, 2), [1., 1., 1.]; fillrange=2, alpha=0.4, color=:gray, label=L"S")
end

# ╔═╡ b9a85716-d294-11ef-10e0-a7b08b800a98
md"""
## Maximum Likelihood Estimation (MLE) Revisited

##### MLE as a special case of Bayesian Inference

To determine the MLE of ``\mu`` as a special case of Bayesian inference, we let ``\sigma_0^2 \rightarrow \infty`` in the Bayesian posterior for ``\mu`` (Eq. B-2.141) to get a uniform prior for ``\mu``. This yields

```math
\begin{align}
 \mu_{\text{ML}} = \left.\mu_N\right\vert_{\sigma_0^2 \rightarrow \infty} = \frac{1}{N} \sum_{n=1}^N x_n 
\end{align}
```


"""

# ╔═╡ 0d303dba-51d4-4413-8001-73ed98bf74df
details("Click for proof",
md"""
```math
\begin{align}
 \mu_{\text{ML}} &= \left.\mu_N\right\vert_{\sigma_0^2 \rightarrow \infty} = \Bigg.  \underbrace{\left(\frac{1}{\sigma_0^2} + \sum_n \frac{1}{\sigma^2}\right)^{-1}}_{\text{Eq. B-2.142}} \cdot \underbrace{\left( \frac{1}{\sigma_0^2} \mu_0 + \sum_n \frac{1}{\sigma^2} x_n  \right)}_{\text{Eq. B-2.141 }} \Bigg\vert_{\sigma_0^2 \rightarrow \infty}  \\
&=  \left(\sum_n \frac{1}{\sigma^2}\right)^{-1} \cdot \left( \sum_n \frac{1}{\sigma^2} x_n  \right)  \\
&= \left(\frac{N}{\sigma^2}\right)^{-1} \cdot \left( \frac{1}{\sigma^2} \sum_n  x_n  \right) \\
&= \frac{1}{N} \sum_{n=1}^N x_n 
\end{align}
```
		""")

# ╔═╡ 4a2cd378-0960-4089-81ad-87bf1be9a3b2
md"""
This is a reassuring result: it matches the maximum likelihood estimate for ``\mu`` that we [previously derived by setting the gradient of the log-likelihood function to zero](#Maximum-Likelihood-Estimation).

Of course, in practical applications, the maximum likelihood estimate is not obtained by first computing the full Bayesian posterior and then applying simplifications. This derivation (see proof) is included solely to illuminate the connection between Bayesian inference and maximum likelihood estimation.

"""

# ╔═╡ 50d90759-8e7f-4da5-a741-89b997eae40b
md"""
##### A prediction-correction decomposition 

Having an expression for the maximum likelihood estimate, it is now possible to rewrite the (Bayesian) posterior mean for ``\mu`` as the combination of a prior-based prediction and likelihood-based (data-based) correction. 

Proof that 

```math
\underbrace{\mu_N}_{\substack{\text{posterior} \\ \text{mean}}}= \overbrace{\underbrace{\mu_0}_{\substack{\text{prior} \\ \text{mean}}}}^{\substack{\text{prior-based} \\ \text{prediction}}} + \overbrace{\underbrace{\frac{N \sigma_0^2}{N \sigma_0^2 + \sigma^2}}_{\text{gain}}\cdot \underbrace{\left(\mu_{\text{ML}} - \mu_0 \right)}_{\text{prediction error}}}^{\text{data-based correction}}\tag{B-2.141}
```


"""

# ╔═╡ d05975bb-c5cc-470a-a6f3-60bc43c51e89
details("Click for proof", 
md"""		
```math
\begin{align*}
\mu_N  &= \sigma_N^2 \, \left( \frac{1}{\sigma_0^2} \mu_0 + \sum_n \frac{1}{\sigma^2} x_n  \right) \tag{B-2.141 } \\
  &= \frac{\sigma_0^2 \sigma^2}{N\sigma_0^2 + \sigma^2} \, \left( \frac{1}{\sigma_0^2} \mu_0 + \sum_n \frac{1}{\sigma^2} x_n  \right) \tag{used B-2.142}\\
  &= \frac{ \sigma^2}{N\sigma_0^2 + \sigma^2}   \mu_0 + \frac{N \sigma_0^2}{N\sigma_0^2 + \sigma^2} \mu_{\text{ML}}   \\
  &= \mu_0 + \frac{N \sigma_0^2}{N \sigma_0^2 + \sigma^2}\cdot \left(\mu_{\text{ML}} - \mu_0 \right)
\end{align*}
```
""")		

# ╔═╡ e8e26e57-ae94-478a-8bb2-2868de5d99e0
md"""

Hence, the posterior mean always lies somewhere between the prior mean ``\mu_0`` and the maximum likelihood estimate (the "data" mean) ``\mu_{\text{ML}}``.

"""

# ╔═╡ cfa0d29a-ffd8-4e14-b3fd-03c824db395f
md"""
# Recursive Bayesian Inference
"""

# ╔═╡ b9aa930a-d294-11ef-37ec-8d17be226c74
md"""
## Kalman Filtering (simple case)

##### Problem

Consider a signal 

```math
x_t=\theta+\epsilon_t \, \text{,    with    } \epsilon_t \sim \mathcal{N}(0,\sigma^2)\,,
```
where ``D_t= \left\{x_1,\ldots,x_t\right\}`` is observed *sequentially* (over time). Derive a **recursive** algorithm for 
```math
p(\theta|D_t) \,,
```
i.e., an update rule for (posterior) ``p(\theta|D_t)``, based on (prior) ``p(\theta|D_{t-1})`` and (a new observation) ``x_t``.

"""

# ╔═╡ b9aabe9a-d294-11ef-2489-e9fc0dbb760a
md"""
#### Model specification

The data-generating distribution is given as
```math
p(x_t|\theta) = \mathcal{N}(x_t\,|\, \theta,\sigma^2)\,.
```

For a given new measurement ``x_t`` and given ``\sigma^2``, this equation can also be read as a likelihood function for $\theta$. 

We now need a prior for $\theta$. Let's define the estimate for $\theta$ after ``t`` observations (i.e., our *solution* ) as ``p(\theta|D_t) = \mathcal{N}(\theta\,|\,\mu_t,\sigma_t^2)``. The prior is then given by

```math
p(\theta|D_{t-1}) = \mathcal{N}(\theta\,|\,\mu_{t-1},\sigma_{t-1}^2)\,.
```

"""

# ╔═╡ b9aad50e-d294-11ef-23d2-8d2bb3b47574
md"""
#### Inference

Use Bayes rule,

```math
\begin{align*}
p(\theta|D_t) &= p(\theta|x_t,D_{t-1}) \\
  &\propto p(x_t,\theta | D_{t-1}) \\
  &= p(x_t|\theta) \, p(\theta|D_{t-1}) \\
  &= \mathcal{N}(x_t|\theta,\sigma^2) \, \mathcal{N}(\theta\,|\,\mu_{t-1},\sigma_{t-1}^2) \\
  &= \mathcal{N}(\theta|x_t,\sigma^2) \, \mathcal{N}(\theta\,|\,\mu_{t-1},\sigma_{t-1}^2) \;\;\text{(note this trick)}\\
  &= \mathcal{N}(\theta|\mu_t,\sigma_t^2) \;\;\text{(use Gaussian multiplication formula SRG-6)}
\end{align*}
```

with

```math
\begin{align*}
K_t &= \frac{\sigma_{t-1}^2}{\sigma_{t-1}^2+\sigma^2} \qquad \text{(Kalman gain)}\\
\mu_t &= \mu_{t-1} + K_t \cdot (x_t-\mu_{t-1})\\
\sigma_t^2 &= \left( 1-K_t \right) \sigma_{t-1}^2 
\end{align*}
```

"""

# ╔═╡ b9aaee4a-d294-11ef-2ed7-0dcb360d8bb7
md"""
This *online* (recursive) estimator of mean and variance in Gaussian observations is called a **Kalman Filter**.

 

"""

# ╔═╡ b9aafc6e-d294-11ef-1b1a-df718c1f1a58
md"""
Note that the so-called Kalman gain ``K_t`` serves as a "learning rate" (step size) in the update equation for the posterior mean ``\mu_t``.

"""

# ╔═╡ e2fc4945-4f88-4520-b56c-c7208b62c29d
keyconcept("", md"Bayesian inference does not require manual tuning of a learning rate; instead, it adapts its own effective learning rate via balancing prior beliefs with incoming evidence.")
 

# ╔═╡ b9ab0b46-d294-11ef-13c5-8314655f7867
md"""
Note that the uncertainty about ``\theta`` decreases over time (since ``0<(1-K_t)<1``). If we assume that the statistics of the system do not change (stationarity), each new sample provides new information about the process, so the uncertainty decreases. 

"""

# ╔═╡ b9ab1dd4-d294-11ef-2e86-31c4a4389475
md"""
Recursive Bayesian estimation as discussed here is the basis for **adaptive signal processing** algorithms such as the [Least Mean Squares](https://en.wikipedia.org/wiki/Least_mean_squares_filter) (LMS) filter and the [Recursive Least Squares](https://en.wikipedia.org/wiki/Recursive_least_squares_filter) (RLS) filter. Both RLS and LMS are special cases of Recursive Bayesian estimation.

"""

# ╔═╡ b9ab2e32-d294-11ef-2ccc-9760ead59972
md"""
$(section_outline("Code Example:", "Kalman Filtering"))

Let's implement the Kalman filter described above. We'll use it to recursively estimate the value of ``\theta`` based on noisy observations.

"""

# ╔═╡ 3a53f67c-f291-4530-a2ba-f95a97b27960
@bindname N_data_kalman Slider(1:100; default=100, show_value=true)

# ╔═╡ b9ab9e28-d294-11ef-3a73-1f5cefdab3d8
md"""
The shaded area represents 2 standard deviations of posterior ``p(\theta|D)``. The variance of the posterior is guaranteed to decrease monotonically for the standard Kalman filter.

"""

# ╔═╡ ffa570a9-ceda-4a21-80a7-a193de12fa2c
md"""
### Implementation
Here is the implementation, but feel free to skip this part.
"""

# ╔═╡ 85b15f0a-650f-44be-97ab-55d52cb817ed
begin
	n = N_data_kalman  # number of observations
	θ = 2.0            # true value of the parameter we would like to estimate
	noise_σ2 = 0.3     # variance of observation noise
	observations = noise_σ2 * randn(MersenneTwister(1), n) .+ θ	
end;

# ╔═╡ 115eabf2-c476-40f8-8d7b-868a7359c1b6
function perform_kalman_step(prior :: Normal, x :: Float64, noise_σ2 :: Float64)
    K = prior.σ / (noise_σ2 + prior.σ)          # compute the Kalman gain
    posterior_μ = prior.μ + K*(x - prior.μ)     # update the posterior mean
    posterior_σ = prior.σ * (1.0 - K)           # update the posterior standard deviation
    return Normal(posterior_μ, posterior_σ)     # return the posterior
end;

# ╔═╡ 61764e4a-e5ef-4744-8c71-598b2155f4d9
begin
	post_μ = fill!(Vector{Float64}(undef,n + 1), NaN)     # means of p(θ|D) over time
	post_σ2 = fill!(Vector{Float64}(undef,n + 1), NaN)    # variances of p(θ|D) over time

	# specify the prior distribution (you can play with the parameterization of this to get a feeling of how the Kalman filter converges)
	prior = Normal(0, 1)

	# save prior mean and variance to show these in plot
	post_μ[1] = prior.μ
	post_σ2[1] = prior.σ
	
	
	# note that this loop demonstrates Bayesian learning on streaming data; we update the prior distribution using observation(s), after which this posterior becomes the new prior for future observations
	for (i, x) in enumerate(observations)
		# compute the posterior distribution given the observation
	    posterior = perform_kalman_step(prior, x, noise_σ2)
		# save the mean of the posterior distribution
	    post_μ[i + 1] = posterior.μ
		# save the variance of the posterior distribution
	    post_σ2[i + 1] = posterior.σ
		# the posterior becomes the prior for future observations
	    prior = posterior
	end
end

# ╔═╡ 661082eb-f0c9-49a9-b046-8705f4342b37
let
	obs_scale = collect(2:n+1)
	# scatter the observations
	scatter(obs_scale, observations, label=L"D", )  
	post_scale = collect(1:n+1)
	# lineplot our estimated means of intermediate posterior distributions
	plot!(post_scale, post_μ, ribbon=sqrt.(post_σ2), linewidth=3, label=L"p(θ | D_t)")
	# plot the true value of θ
	plot!(post_scale, θ*ones(n + 1), linewidth=2, label=L"θ")
end

# ╔═╡ b9ac7486-d294-11ef-13e5-29b7ffb440bc
md"""
# Summary

A **linear transformation** ``z=Ax+b`` of a Gaussian variable ``x \sim \mathcal{N}(\mu_x,\Sigma_x)`` is Gaussian distributed as

```math
p(z) = \mathcal{N} \left(z \,|\, A\mu_x+b, A\Sigma_x A^T \right) 
```

Bayesian inference with a Gaussian prior and Gaussian likelihood leads to an analytically computable Gaussian posterior, because of the **multiplication rule for Gaussians**:

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

**Conditioning and marginalization** of a multivariate Gaussian distribution yields Gaussian distributions. In particular, the joint distribution

```math
\mathcal{N} \left( \begin{bmatrix} x \\ y \end{bmatrix} \left| \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix}, 
  \begin{bmatrix} \Sigma_x & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_y \end{bmatrix} \right. \right)
```

can be decomposed as

```math
\begin{align*}
 p(y|x) &= \mathcal{N}\left(y\,|\,\mu_y + \Sigma_{yx}\Sigma_x^{-1}(x-\mu_x),\, \Sigma_y - \Sigma_{yx}\Sigma_x^{-1}\Sigma_{xy} \right) \\
p(x) &= \mathcal{N}\left( x|\mu_x, \Sigma_x \right)
\end{align*}
```

Here's a nice [summary of Gaussian calculations](https://github.com/bertdv/AIP-5SSB0/raw/master/lessons/notebooks/files/RoweisS-gaussian_formulas.pdf) by Sam Roweis. 

"""

# ╔═╡ 79a99a22-3bb5-431b-bf84-5dce5cccfe25
md"""
# Exercises

"""

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
details("Click for solution",
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
details("Click for solution",
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
details("Click for solution",
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

# ╔═╡ 6dfc31a0-d0d7-4901-a876-890df9ab4258
md"""
# Optional
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
$(section_outline("Code Example:", "Product of Gaussian Distributions"))


We plot ``p(Z=XY)`` and ``p(X)p(Y)`` for ``X\sim\mathcal{N}(0,1)`` and ``Y \sim \mathcal{N}(0,1)`` to give an idea of how these distributions differ.

"""


# ╔═╡ b9ac09c4-d294-11ef-2cb8-270289d01f25
md"""
In short, Gaussian-distributed variables remain Gaussian in linear systems, but this is not the case in non-linear systems. 

"""

# ╔═╡ f78bc1f5-cf7b-493f-9c5c-c2fbd6788616
md"""
# Code
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HCubature = "19dc6840-f33b-545b-b366-655c7e3ffd49"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[compat]
Distributions = "~0.25.120"
HCubature = "~1.7.0"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
Plots = "~1.40.17"
PlutoTeachingTools = "~0.4.4"
PlutoUI = "~0.7.68"
SpecialFunctions = "~2.5.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "3af7b7a1e24ddf3753ac4474e9b19716a9f90072"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

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
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "a656525c8b46aa6a1c76891552ed5381bb32ae7b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.30.0"

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
git-tree-sha1 = "8010b6bb3388abe68d95743dcbea77650bb2eddf"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.3"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "0037835448781bb46feb39866934e243886d756a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

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
git-tree-sha1 = "3e6d038b77f22791b8e3472b7c633acea1ecac06"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.120"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

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
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "83dc665d0312b41367b7263e8a4d172eac1897f4"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.4"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3a948313e7a41eb1db7a1e733e6335f17b4ab3c4"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "7.1.1+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "1828eb7275491981fa5f1752a5e126e8f26f8741"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.17"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "27299071cc29e409488ada41ec7643e0ab19091f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.17+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "35fbd0cefb04a516104b8e183ce0df11b70a3f1a"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.84.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HCubature]]
deps = ["Combinatorics", "DataStructures", "LinearAlgebra", "QuadGK", "StaticArrays"]
git-tree-sha1 = "19ef9f0cb324eed957b7fe7257ac84e8ed8a48ec"
uuid = "19dc6840-f33b-545b-b366-655c7e3ffd49"
version = "1.7.0"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "ed5e9c58612c4e081aecdb6e1a479e18462e041e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.17"

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
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "52e1296ebbde0db845b356abbbe67fb82a0a116c"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.9"

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
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

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
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

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
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

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
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+4"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2ae7d4ddec2e13ad3bddf5c0796f7547cf682391"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.2+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "275a9a6d85dc86c24d03d1837a0010226a96f540"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.3+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
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
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "9a9216c0cf706cb2cc58fd194878180e3e51e8c0"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.18"

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
git-tree-sha1 = "85778cdf2bed372008e6646c64340460764a5b85"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "2d7662f95eafd3b6c346acdbfc11a762a2256375"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.69"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "eb38d376097f47316fe089fc62cb7c6d85383a52"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "e1d5e16d0f65762396f9ca4644a5f4ddab8d452b"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
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
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

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
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "cbea8a6bd7bed51b1619658dec70035e07b8502f"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.14"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

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
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2c962245732371acd51700dbb268af311bddd719"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.6"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

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
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

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

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "6258d453843c466d84c17a58732dda5deeb8d3af"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.24.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "af305cc62419f9bd61b6644d19170a4d258c7967"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.7.0"

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
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

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
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

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
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "9caba99d38404b285db8801d5c45ef4f4f425a6d"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.1+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "c5bf2dad6a03dfef57ea0a170a1fe493601603f2"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.5+0"

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
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

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
git-tree-sha1 = "4bba74fa59ab0755167ad24f98800fe5d727175b"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.12.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

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
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

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
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

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
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
"""

# ╔═╡ Cell order:
# ╟─b9a38e20-d294-11ef-166b-b5597125ed6d
# ╟─5e9a51b1-c6e5-4fb5-9df3-9b189f3302e8
# ╟─b9a46c3e-d294-11ef-116f-9b97e0118e5b
# ╟─82025c2f-a21f-4080-b301-3ffe3715442d
# ╟─b9a48c60-d294-11ef-3b90-03053fcd82fb
# ╟─ba57ecbb-b64e-4dd8-8398-a90af1ac71f3
# ╟─02853a5c-f6aa-4af8-8a25-bfffd4b96afc
# ╟─71f1c8ee-3b65-4ef8-b36f-3822837de410
# ╟─b9a4eb62-d294-11ef-06fa-af1f586cbc15
# ╟─b9a50d0c-d294-11ef-0e60-2386cf289478
# ╟─b9a52b18-d294-11ef-2d42-19c5e3ef3549
# ╟─b9a5589a-d294-11ef-3fc3-0552a69df7b2
# ╟─9501922f-b928-46e2-8f23-8eb9c64f6198
# ╟─b9a5889c-d294-11ef-266e-d90225222e10
# ╟─56510a09-073c-4fc8-b0b7-17b20dbb95f0
# ╟─a82378ae-d1be-43f9-b63a-2f897767d1fb
# ╟─36eff7bc-72f2-4b48-a109-1861af6834aa
# ╟─87f400ac-36f2-4778-a3ba-06dd7652e279
# ╟─9c2bf0a2-4bb6-4769-b47b-6a02c4e73044
# ╟─8f7ecb91-d251-4ac9-bb32-0dd7215382e3
# ╟─1df7a10d-c4f6-40d6-8f5a-cbd79ef1d415
# ╟─673360e8-27ed-471c-a866-15af550df5e7
# ╟─9eb3e920-fab5-4a6a-8fe1-5734ebc6b25c
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
# ╟─93361b31-022f-46c0-b80d-b34f3ed61d5f
# ╟─bbf3a1e7-9f25-434c-95c7-898648b5bc90
# ╟─b9a7073a-d294-11ef-2330-49ffa7faff21
# ╟─45c2fb37-a078-4284-9e04-176156cffb1e
# ╟─df8867ed-0eff-4a52-8f5e-2472467e1aa2
# ╟─3a0f7324-0955-4c1c-8acc-0d33ebd16f78
# ╟─db730ca7-4850-49c7-a93d-746d393b509b
# ╟─b9a885a8-d294-11ef-079e-411d3f1cda03
# ╟─b9a9565c-d294-11ef-1b67-83d1ab18035b
# ╟─b9a99fcc-d294-11ef-3de4-5369d9796de7
# ╟─b9a9b8e0-d294-11ef-348d-c197c4ce2b8c
# ╟─b9a9dca8-d294-11ef-04ec-a9202c319f89
# ╟─b9a9f98e-d294-11ef-193a-0dbdbfffa86f
# ╟─b9aa27da-d294-11ef-0780-af9d89f9f599
# ╟─b9aa3950-d294-11ef-373f-d5d330694bfd
# ╟─b426f9c8-4506-43ef-92fa-2ee30be621ca
# ╟─b9a80522-d294-11ef-39d8-53a536d66bf9
# ╟─364cd002-92ee-4fb6-b89a-3251eff7502c
# ╟─922f0eb6-9e29-4b6c-9701-cb7b2f07bb7a
# ╟─9bd38e28-73d4-4c6c-a1fe-35c7a0e750b3
# ╟─b9ac2d3c-d294-11ef-0d37-65a65525ad28
# ╠═5638c1d0-db95-49e4-bd80-528f79f2947e
# ╟─b9ac5190-d294-11ef-0a99-a9d369b34045
# ╟─b9a85716-d294-11ef-10e0-a7b08b800a98
# ╟─0d303dba-51d4-4413-8001-73ed98bf74df
# ╟─4a2cd378-0960-4089-81ad-87bf1be9a3b2
# ╟─50d90759-8e7f-4da5-a741-89b997eae40b
# ╟─d05975bb-c5cc-470a-a6f3-60bc43c51e89
# ╟─e8e26e57-ae94-478a-8bb2-2868de5d99e0
# ╟─cfa0d29a-ffd8-4e14-b3fd-03c824db395f
# ╟─b9aa930a-d294-11ef-37ec-8d17be226c74
# ╟─b9aabe9a-d294-11ef-2489-e9fc0dbb760a
# ╟─b9aad50e-d294-11ef-23d2-8d2bb3b47574
# ╟─b9aaee4a-d294-11ef-2ed7-0dcb360d8bb7
# ╟─b9aafc6e-d294-11ef-1b1a-df718c1f1a58
# ╟─e2fc4945-4f88-4520-b56c-c7208b62c29d
# ╟─b9ab0b46-d294-11ef-13c5-8314655f7867
# ╟─b9ab1dd4-d294-11ef-2e86-31c4a4389475
# ╟─b9ab2e32-d294-11ef-2ccc-9760ead59972
# ╟─3a53f67c-f291-4530-a2ba-f95a97b27960
# ╟─661082eb-f0c9-49a9-b046-8705f4342b37
# ╟─b9ab9e28-d294-11ef-3a73-1f5cefdab3d8
# ╟─ffa570a9-ceda-4a21-80a7-a193de12fa2c
# ╠═9edd80d4-d088-4b2f-8843-abaa7a5d9c5e
# ╠═85b15f0a-650f-44be-97ab-55d52cb817ed
# ╠═115eabf2-c476-40f8-8d7b-868a7359c1b6
# ╠═61764e4a-e5ef-4744-8c71-598b2155f4d9
# ╟─b9ac7486-d294-11ef-13e5-29b7ffb440bc
# ╟─79a99a22-3bb5-431b-bf84-5dce5cccfe25
# ╟─14b3edcc-0d16-4055-9b1c-7f324514a0a9
# ╟─dd7786e2-d6ac-4dba-abca-3686242c067d
# ╟─b7a810a3-dc38-4e72-ab10-2ad2f064bdbb
# ╟─f711b053-dccf-4bf1-b285-e8da94a48b68
# ╟─22539cfe-3694-4100-8120-ca6ac1e66b31
# ╟─fa197526-6706-47ce-b84b-5675eee00610
# ╟─645308ac-c9e3-4d6f-bcff-82327fbb8edf
# ╟─03c399e1-d0d8-493a-9f95-4209918d132a
# ╟─6dfc31a0-d0d7-4901-a876-890df9ab4258
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
# ╟─f78bc1f5-cf7b-493f-9c5c-c2fbd6788616
# ╠═c97c495c-f7fe-4552-90df-e2fb16f81d15
# ╠═3ec821fd-cf6c-4603-839d-8c59bb931fa9
# ╠═00482666-0772-4e5d-bb35-df7b6fb67a1b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
