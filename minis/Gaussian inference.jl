### A Pluto.jl notebook ###
# v0.20.10

#> [frontmatter]
#> image = "https://imgur.com/v6BLTI0.png"
#> title = "Mini: Gaussian Inference example"
#> description = "A simple example of Gaussian inference in practice, using an interactive dataset."

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

# ╔═╡ db1a7392-825a-454c-a777-954c05a6a310
using RxInfer

# ╔═╡ f4e55044-ff65-4c21-aeb3-2d8be2b97cf4
using Random

# ╔═╡ 452300fb-2cf2-4fab-8923-b847ed4b0c97
using Distributions, Plots, LaTeXStrings, PlutoUI

# ╔═╡ 636389e3-f169-4255-9b4a-8d3426d042b0
using Integrals

# ╔═╡ 5062b49b-34d0-4364-a9b3-7d29119ed599
using PlutoTeachingTools

# ╔═╡ 3eeabdc9-1906-4578-a2d0-cdf28246a24d
html"""
<style>
	pluto-output h2:first-child,
	pluto-output h2
	{
		margin-block-start: 8rem;
	}
	
	pluto-output h3:first-child,
	pluto-output h3
	{
		margin-block-start: 5rem;
	}
</style>
"""

# ╔═╡ 246dc845-2a07-4109-b6df-72722cc3e8a2
md"""
# Simple inference example

Let's walk through a simple example of **(Bayesian) inference**: given a dataset, we want to infer the parameters to a distribution that were used to generate the data.
"""

# ╔═╡ ce5cf62a-bd84-4abe-8cef-07fde003624d
md"""
## Sampling from the Gaussian distribution

In Julia, we can sample from a Normal distribution using the `rand` function and the Distributions.jl package:
"""

# ╔═╡ 41bc3e62-e454-4f14-9d3c-c725a5a93aa8
sample_gaussian_data = rand(Normal(5.0, 0.1), 100)

# ╔═╡ 24125436-1b30-4d05-8502-3c8ede5cc00f
md"""
This gives us a Vector of **`100` numbers**, sampled from the distribution **`Normal(5.0, 0.1)`**.

Let's look at these numbers in a histogram:
"""

# ╔═╡ 29f9c097-8bca-4339-a927-28d101cb3a22
histogram(
	sample_gaussian_data;
	bins=3.975:.05:6,
	size=(650,90), legend=nothing, color="black"
)

# ╔═╡ 7fad63f9-c986-438f-99ec-bc8b8bf70ce5
md"""

## Inference



In the code above, we were **given** the numbers ``(\mu, \sigma)``, and we **generated** 100 numbers.


The **inference** task is the opposite: **given** 100 numbers sampled from a distribution, we try to **infer** the parameters ``(\mu, \sigma)`` were used to generate that data.

Let's look at two methods of inference:
- **ML estimator** gives the most likely **value** of ``(\mu, \sigma)``.
- *(later)* **Bayesian inference** gives **posterior distributions** for ``\mu`` and ``\sigma``.


### ML estimator

We can compute the statistical mean and standard deviation of our data:
"""

# ╔═╡ cc17e180-89f1-48b7-a9dd-fa1e7814ceb7
mean(sample_gaussian_data)

# ╔═╡ b9aa17ac-ff44-4899-b15a-e2136eaadbed
std(sample_gaussian_data)

# ╔═╡ fa5e50f5-56b8-48c0-83a0-031fade560d6
md"""

In this simple Gaussian case, the statistical mean and standard deviation are TODO WHAT IS ML


These values give our **best guess** for the parameters that were used to generate the data. But **how sure are we about these estimates?** Can we give a margin of error?


"""

# ╔═╡ 108ba4f6-9292-45a5-a4f3-bb560b43d31b


# ╔═╡ 3a99575c-2773-4b86-814c-465a74d297cf
md"""
## Bayesian Inference

In the example below, we take a subset from our dataset. We can then use Bayesian Inference to infer the parameters (mean, variance) from the dataset.
"""

# ╔═╡ 89c630f6-e7ef-4284-86cd-5eaa3383735e
@bindname data_size Slider(
	[1,10,20,50,100,200,500,1000,2000,5000,10_000,20_000]; 
	show_value=true,
	default=10,
)

# ╔═╡ 3271e5a2-d749-4a7c-801f-f1cb692c74ba


# ╔═╡ 21f842fe-e37a-4154-8561-ef814ccd3e54
md"""
With Bayesian machine learning, we can use this data to **infer** the generating distribution.


As more data is provided, our inference gives a more precise result!
"""

# ╔═╡ 57df5ab6-a9ae-4c8f-8147-78b7e16988d4


# ╔═╡ 9e0d5434-6953-4e7c-884b-9ef607441e1b


# ╔═╡ 7384f4ee-a889-44db-842b-df30ed1e6a49
full_data = rand(
	# 🤫 these parameters are secret! we will try to infer them from the data
	Normal(10, 7),
	20_000
)

# ╔═╡ adac696f-1b5e-4af4-bd45-4a13e8606332
data = full_data[1:data_size]

# ╔═╡ b3f91134-c64c-49cc-b9a6-4da1fe2a37ad
histogram(data; bins=-30.5:30, xlim=(-30,30), size=(650,90), legend=nothing, color="black")

# ╔═╡ bc13ef80-d16f-48c2-b699-3c1043927c8e
md"""
### Three variances

In each of the three charts, take a look at the **variance**. What does the variance mean in this chart?

#### Chart 1 & 2: **data variance**
In the first chart, you can see the variance of our observed variable, which is σ² = 7². 

In the second chart, you see our inference results! We tried to **infer** the standard deviation from our data. 


#### Chart 3: **posterior variance**
In the third chart, you see another variance, the **variance of the posterior for μ**! A small variance of the posterior means **high precision**. Indeed, as we get more data, this variance gets very small – we are very sure about our inference 

"""

# ╔═╡ eed3b2da-8eb6-4462-a852-d39e95fdba48


# ╔═╡ 80612314-bf45-4bb8-9316-a910716ca459
@model function simple_data_model(y, a)
	μ ~ Normal(mean=0, var=a)
	σ = 7
	
    for i in eachindex(y)
        y[i] ~ Normal(mean=μ, var=σ^2)
    end
end

# ╔═╡ 25032652-bf9b-4165-9c2f-15a6298145f4
result = infer(
	model = simple_data_model(a = 1e9),
	data = (y = data,),
)

# ╔═╡ 1c432cb9-bcae-44aa-a32e-1fd142a938b1
result.posteriors

# ╔═╡ 75c3d336-f4dc-43d4-a9fe-d026d37dac66
posterior = convert(Normal, result.posteriors[:μ])

# ╔═╡ 4ffd7bad-3d46-43f5-a0f3-d8a642b98f4c
let
	p = plot(; xlim=(-30,30), ylim=(0,0.06), size=(650,90), legend=nothing, title="Some possible distributions")
	μ_samples = rand(posterior, 20)
	σ_samples = rand(Normal(7.0, 10/data_size), 20)

	# @info "hujh" σ_samples

	for (μ,σ) in zip(μ_samples, σ_samples)
		plot!(p, x -> Distributions.pdf(Normal(μ, max(2.0,σ)), x); color="black", opacity=.3)
		vline!(p, [μ]; color="red", opacity=.3)
	end

	p
end
	
	

# ╔═╡ 0501b18c-cdc9-4772-917f-195c75f05118
plot(x -> Distributions.pdf(posterior, x); xlim=(-30,30), ylim=(0,1), size=(650,90), legend=nothing, color="red", normalize=:pdf, title="Posterior of μ")

# ╔═╡ 05938560-9159-4528-af37-c0451fd4aca0
md"""
## Bzzzzzz everything below you can ignore
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
"""

# ╔═╡ bd9e3aa4-e6a2-4b4e-bd45-ed9a80374cb4
md"""
## Multivariate Gaussian distribution


"""

# ╔═╡ 0dd817c0-b7c6-11ef-1f8b-ff0f59f7a8ce
md"""
### Preliminaries

Goal 

  * Review of information processing with Gaussian distributions in linear systems

Materials        

  * Mandatory

      * These lecture notes
  * Optional

      * Bishop pp. 85-93
      * [MacKay - 2006 - The Humble Gaussian Distribution](https://github.com/bmlip/course/blob/main/assets/files/Mackay-2006-The-humble-Gaussian-distribution.pdf) (highly recommended!)
      * [Ariel Caticha - 2012 - Entropic Inference and the Foundations of Physics](https://github.com/bmlip/course/blob/main/assets/files/Caticha-2012-Entropic-Inference-and-the-Foundations-of-Physics.pdf), pp.30-34, section 2.8, the Gaussian distribution
  * References

      * [E.T. Jaynes - 2003 - Probability Theory, The Logic of Science](http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf) (best book available on the Bayesian view on probability theory)

"""

# ╔═╡ 06cad4f9-b40c-428f-b2b9-686274749dc9
rand_deterministic(a...) = rand(MersenneTwister(1), a...)

# ╔═╡ 0dd82814-b7c6-11ef-3927-b3ec0b632c31
md"""
### Example Problem

Consider a set of observations ``D=(x_1,…,x_N)`` in ``\mathbb{R}^2`` (see Figure). All observations were generated by the same process. 

We now draw an extra observation ``x_\bullet ∈ \mathbb{R}^2`` from the same data generating process. What is the probability that ``x_\bullet`` lies within the shaded rectangle ``S \subseteq \mathbb{R}^2``?

"""

# ╔═╡ a3638872-847f-47a2-941d-94b9c7d2218b
N = 100

# ╔═╡ cd6cc0a8-eb41-4366-82e4-2b6ee92c7a37
S = ([0,1.],[2,2.])

# ╔═╡ 0aa00f04-a53e-4c2b-9b54-a9f93977c481
in_rectangle(x, rectangle=S) = all(rectangle[1] .<= x .<= rectangle[2]) 

# ╔═╡ 07b6e6be-eed0-4a89-bcdb-52f3ed79f100
@bind try_again CounterButton()

# ╔═╡ ff222c45-8909-489d-b4a2-ec7d9914ede4
generative_dist = let
	# reference 
	try_again

	# the distribution
	MvNormal(
		[0.0, 1.0], 
		[
			0.8 0.5
			0.5 1.0
		]
	)
end

# ╔═╡ 5c9b660b-d44a-402f-a1f8-738911a183b3
D = rand(generative_dist, N)

# ╔═╡ d34233a3-60fe-4ac6-9822-b6097749206b
x_dot = rand(generative_dist)

# ╔═╡ 76cb64ad-e3e7-4af1-b5b3-4bd73f68883b
let
	scatter(D[1,:], D[2,:], marker=:x, markerstrokewidth=3, label=L"D")
	scatter!([x_dot[1]], [x_dot[2]], label=L"x_\bullet")
	plot!(range(0, 2), [1., 1., 1.], fillrange=2, alpha=0.4, color=:gray,label=L"S")
	plot!(xlim=(-3,3), ylim=(-3,3))
end

# ╔═╡ 085e9093-e3d1-41b7-8b3e-ae623809f13d
md"""
### Sampling
"""

# ╔═╡ 0a373d0f-601d-4fce-a1d5-e02732ece9fe
n_samples = 500001

# ╔═╡ b5e23f3d-b35a-40f1-8bd4-58d6128f2e29
count(in_rectangle, eachcol(rand(generative_dist, n_samples))) / n_samples

# ╔═╡ 51fb9204-b480-47f8-bb1e-515b5a0af730
md"""
### Using the `pdf`
"""

# ╔═╡ 042bab9b-e162-436e-894d-70e7ea98fb08
function ∫(f, S)
	prob = IntegralProblem(
		(x, p) -> f(x), 
		S
	)
	solve(prob, HCubatureJL(); abstol=1e-3).u
end

# ╔═╡ 3ca80762-1e7b-46c8-a101-9206de278077
∫(S) do x
	Distributions.pdf(generative_dist, x)
end

# ╔═╡ 0dd835ca-b7c6-11ef-0e33-1329e4ba13d8
md"""
### The Gaussian Distribution

Consider a random (vector) variable ``x \in \mathbb{R}^M`` that is "normally" (i.e., Gaussian) distributed. The *moment* parameterization of the Gaussian distribution is completely specified by its *mean* ``\mu`` and *variance* ``\Sigma`` and given by

```math
p(x | \mu, \Sigma) = \mathcal{N}(x|\mu,\Sigma) \triangleq \frac{1}{\sqrt{(2\pi)^M |\Sigma|}} \,\exp\left\{-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) \right\}\,.
```

where ``|\Sigma| \triangleq \mathrm{det}(\Sigma)`` is the determinant of ``\Sigma``.  

For the scalar real variable ``x \in \mathbb{R}``, this works out to 

```math
p(x | \mu, \sigma^2) =  \frac{1}{\sqrt{2\pi\sigma^2 }} \,\exp\left\{-\frac{(x-\mu)^2}{2 \sigma^2} \right\}\,.
```

"""

# ╔═╡ 0dd84542-b7c6-11ef-3115-0f8b26aeaa5d
md"""
Alternatively, the <a id="natural-parameterization">*canonical* (a.k.a. *natural*  or *information* ) parameterization</a> of the Gaussian distribution is given by

```math
\begin{equation*}
p(x | \eta, \Lambda) = \mathcal{N}_c(x|\eta,\Lambda)  = \exp\left\{ a + \eta^T x - \frac{1}{2}x^T \Lambda x \right\}\,.
\end{equation*}
```

```math
a = -\frac{1}{2} \left( M \log(2 \pi) - \log |\Lambda| + \eta^T \Lambda \eta\right)
```

is the normalizing constant that ensures that ``\int p(x)\mathrm{d}x = 1``.

```math
\Lambda = \Sigma^{-1}
```

is called the *precision matrix*.

```math
\eta = \Sigma^{-1} \mu
```

is the *natural* mean or for clarity often called the *precision-weighted* mean.

"""

# ╔═╡ 0dd8528a-b7c6-11ef-3bc9-eb09c0c530d8
md"""
### Why the Gaussian?

Why is the Gaussian distribution so ubiquitously used in science and engineering? (see also [Jaynes, section 7.14](http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf#page=250), and the whole chapter 7 in his book).

"""

# ╔═╡ 0dd85c94-b7c6-11ef-06dc-7b8797c13fda
md"""
(1) Operations on probability distributions tend to lead to Gaussian distributions:

  * Any smooth function with single rounded maximum, if raised to higher and higher powers, goes into a Gaussian function. (useful in sequential Bayesian inference).
  * The [Gaussian distribution has higher entropy](https://en.wikipedia.org/wiki/Differential_entropy#Maximization_in_the_normal_distribution) than any other with the same variance. 

      * Therefore any operation on a probability distribution that discards information but preserves variance gets us closer to a Gaussian.
      * As an example, see [Jaynes, section 7.1.4](http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf#page=250) for how this leads to the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem), which results from performing convolution operations on distributions.

"""

# ╔═╡ 0dd8677a-b7c6-11ef-357f-2328b10f5274
md"""
(2) Once the Gaussian has been attained, this form tends to be preserved. e.g.,   

  * The convolution of two Gaussian functions is another Gaussian function (useful in sum of 2 variables and linear transformations)
  * The product of two Gaussian functions is another Gaussian function (useful in Bayes rule).
  * The Fourier transform of a Gaussian function is another Gaussian function.

"""

# ╔═╡ 0dd86f40-b7c6-11ef-2ae8-a3954469bcee
md"""
### Transformations and Sums of Gaussian Variables

A **linear transformation** ``z=Ax+b`` of a Gaussian variable ``x \sim \mathcal{N}(\mu_x,\Sigma_x)`` is Gaussian distributed as

```math
p(z) = \mathcal{N} \left(z \,|\, A\mu_x+b, A\Sigma_x A^T \right) \tag{SRG-4a}
```

In fact, after a linear transformation ``z=Ax+b``, no matter how ``x`` is distributed, the mean and variance of ``z`` are always given by ``\mu_z = A\mu_x + b``  and ``\Sigma_z = A\Sigma_x A^T``, respectively (see   [probability theory review lesson](https://bmlip.github.io/course/lectures/Probability%20Theory%20Review.html#linear-transformation)). In case ``x`` is not Gaussian, higher order moments may be needed to specify the distribution for ``z``. 

"""

# ╔═╡ 0dd87a3a-b7c6-11ef-2bc2-bf2b4969537c
md"""
The **sum of two independent Gaussian variables** is also Gaussian distributed. Specifically, if ``x \sim \mathcal{N} \left(\mu_x, \Sigma_x \right)`` and ``y \sim \mathcal{N} \left(\mu_y, \Sigma_y \right)``, then the PDF for ``z=x+y`` is given by

```math
\begin{align*}
p(z) &= \mathcal{N}(x\,|\,\mu_x,\Sigma_x) \ast \mathcal{N}(y\,|\,\mu_y,\Sigma_y) \\
  &= \mathcal{N} \left(z\,|\,\mu_x+\mu_y, \Sigma_x +\Sigma_y \right) \tag{SRG-8}
\end{align*}
```

The sum of two Gaussian *distributions* is NOT a Gaussian distribution. Why not?

"""

# ╔═╡ 5978ce29-3fd1-44c1-a3cf-37fd336e2a35
a = rand(Normal(4.3, 2), 51000)

# ╔═╡ 2c31df74-726b-48ff-a402-5a9f67392060
b = rand(Normal(3.0, 2.0), 51000)

# ╔═╡ e90979ad-cfd8-4fe8-9bd3-a434e394a8f2
md"""
The product of two Guassian distributed variables is **not** Gaussian distributed!
"""

# ╔═╡ 272ae2be-7784-4100-a03f-4b3aa400dd18
hist(data; kwargs...) = histogram(data; bins=-30.5:30, xlim=(-30,30), size=(650,90), legend=nothing, kwargs...)

# ╔═╡ 0837d757-9beb-4956-836c-a2c487bea9e3
hist(a; color="blue")

# ╔═╡ 09eb45b0-4117-46ae-adaf-fe1e82578478
hist(b; color="red")

# ╔═╡ 8554d2d0-0acf-47a1-92e1-9b025031a5d6
let
	sum = a .+ b

	hist(sum; color="purple", normalize=:pdf)
	plot!(x -> Distributions.pdf(Normal(mean(sum),std(sum)),x))
end

# ╔═╡ 3c454c77-2320-49b8-b95c-e869f9e9dcd9
let
	prod = a .* b

	hist(prod; color="purple", normalize=:pdf)
	plot!(x -> Distributions.pdf(Normal(mean(prod),std(prod)),x))
end

# ╔═╡ 2ed19557-cf6b-4a2d-819a-0638b17094a6


# ╔═╡ 4acf4347-91c9-4d36-a6d3-2764f0c2296c


# ╔═╡ 61c0c5e5-5d52-47d0-973a-a006ad538870


# ╔═╡ 7b3c8631-51c9-4e4d-8d69-42999dc50858


# ╔═╡ 810a5527-0b40-49d0-bd3f-9007e09e2ba3


# ╔═╡ d651f10f-8a02-489c-9f3c-93ba4c59250b


# ╔═╡ aa4eb90e-2166-4407-97aa-666f309d5e34
md"""
``a + \mu + μ`` and ``\boldsymbol{a} + \boldsymbol{\mu} + \boldsymbol{μ}``
"""

# ╔═╡ 470a0c46-3eae-490b-b13b-5a74eae16a3d
md"""
``\def\S{\boldsymbol{\Sigma}}\def\m{\boldsymbol{\mu}} \S + \m_a`` and ``\foo + \bar``
"""

# ╔═╡ 6108deb0-2c3f-41c2-819d-106f48d210e7
md"""
# This lecture:


## Gaussian distribution is useful

### Occurance in nature, occurance in math, Entropy

### Preserved in common math (addition, ~mult~, fourier transform)

See the parameters of the new distribution





## Gaussian distribution in code

pdf

cdf

integral

RxInfer





## Gaussian distribution in math



pdf * pdf = pdf



## Bayesian inference

Model:

```math
x \sim \mathcal{N}(\mu,\sigma^2)
```

with ``\sigma^2 \in \mathbb{R}_{\geq 0}`` and ``\mu \in \mathbb{R}``.

Given samples from ``x``, we want to infer the value of ``\mu``, when the value of ``\sigma^2`` is known.

As prior, we set:

```math
\mu \sim \mathcal{N}(0, \sigma_{\mu}^2)
```

with a chosen ``\sigma_{\mu}^2 \in \mathbb{R}_{\geq 0}``.

---

ACTUALLY i want to immediately do this for multiple observations, that makes more sense with data

---

So the model is

```math
x \sim \mathcal{N}(\mu,\sigma^2)
```

with 

```math
\mu \sim \mathcal{N}(0, \sigma_{\mu}^2)
```

and given ``\sigma^2, \sigma_{\mu}^2 \in \mathbb{R}_{\geq 0}``


---


The result is:





Derivation:

Easy because you can sum precisions in the posterior.




OKE i will check Bishop for how to properly write this

Would be nice to have a general scheme:

- model
- priors
- data
- knowns
- (bayes)
- posterior


### Inferring sigma (or both)

Optional section

Inferring both might be hard closed form?, also did not manage in RxInfer.



## 
"""

# ╔═╡ a6e21f35-5563-43b4-a6cf-8526be3b5f25


# ╔═╡ 1b26aadc-62a1-4ac1-8cf4-ecdbac7ef63f


# ╔═╡ 2033d8df-7e4c-462f-9c82-f2452460e80a


# ╔═╡ 82d49f7f-cd7f-4406-a883-641069025699


# ╔═╡ 806c47e9-6415-424a-9ee3-67699dbca385


# ╔═╡ 0dd544c8-b7c6-11ef-106b-99e6f84894c3
md"""
# Continuous Data and the Gaussian Distribution

"""

# ╔═╡ a9ab8f29-d72b-472a-b038-c5a884808e2f


# ╔═╡ b738c85f-09f1-4c99-801e-daa90f476fce
convert(NormalWeightedMeanPrecision, Normal(3,0.2))

# ╔═╡ a0427b32-7ce2-46a9-b962-7fab500c9b2f


# ╔═╡ a72dadff-e813-4b18-8fa9-86c9b1515fe2
md"""
The **Gaussian Distribution** is very common in science! What makes this distribution so special?
- It occurs naturally in many processes (see the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem)).
- Gaussian distrbutions are easy to work with mathematically.
"""

# ╔═╡ 0dd88110-b7c6-11ef-0b82-2ffe13a68cad
md"""
### Example: Gaussian Signals in a Linear System

<p style="text-align:center;"><img src="./figures/fig-linear-system.png" width="400px"></p>

Given independent variables

```math
x \sim \mathcal{N}(\mu_x,\sigma_x^2)
```

and ``y \sim \mathcal{N}(\mu_y,\sigma_y^2)``, what is the PDF for ``z = A\cdot(x -y) + b`` ? (for answer, see [Exercises](https://github.com/bmlip/course/tree/main/exercises/Exercises-The-Gaussian-Distribution.ipynb))

"""

# ╔═╡ 0dd88a84-b7c6-11ef-133c-3d85f0703c19
md"""
Think about the role of the Gaussian distribution for stochastic linear systems in relation to what sinusoidals mean for deterministic linear system analysis.

"""

# ╔═╡ c50f6bf0-a1e5-4bd6-a6a2-7201a4302c2f
TODO(md"Think about? ")

# ╔═╡ d93fd4bb-7bf2-453b-97be-602ee494a786


# ╔═╡ 9612754a-7ed3-4d4d-b4a9-9db3c2059271


# ╔═╡ 5dd56dd1-3db0-4b5e-bf74-1ba47f04b58a
TODO(md"""
What is defined here? Is ``\theta`` a stochastic var?
""")

# ╔═╡ 0dd890ee-b7c6-11ef-04b7-e7671227d8cb
md"""
### Bayesian Inference for the Gaussian

Let's estimate a constant ``\theta`` from one 'noisy' measurement ``x`` about that constant. 

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

"""

# ╔═╡ 0dd89b6e-b7c6-11ef-2525-73ee0242eb91
md"""
##### Model specification

Note that you can rewrite these specifications in probabilistic notation as follows:

```math
\begin{align*}
    p(x|\theta) &=  \mathcal{N}(x|\theta,\sigma^2) \\
    p(\theta) &=\mathcal{N}(\theta|\mu_0,\sigma_0^2)
\end{align*}
```

"""

# ╔═╡ 0dd8b5d6-b7c6-11ef-1eb9-4f4289261e79
md"""
(**Notational convention**). Note that we write ``\epsilon \sim \mathcal{N}(0,\sigma^2)`` but not ``\epsilon \sim \mathcal{N}(\epsilon | 0,\sigma^2)``, and we write  ``p(\theta) =\mathcal{N}(\theta|\mu_0,\sigma_0^2)`` but not ``p(\theta) =\mathcal{N}(\mu_0,\sigma_0^2)``. 

"""

# ╔═╡ 0dd8c024-b7c6-11ef-3ca4-f9e8286cbb64
md"""
##### Inference

For simplicity, we assume that the variance ``\sigma^2`` is given and will proceed to derive a Bayesian posterior for the mean ``\theta``. The case for Bayesian inference of ``\sigma^2`` with a given mean is [discussed in the optional slides](#inference-for-precision).

"""

# ╔═╡ 0dd8d976-b7c6-11ef-051f-4f6cb3db3d1b
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

# ╔═╡ 0dd8df66-b7c6-11ef-011a-8d90bba8e2cd
md"""
(Just as an aside,) this computational 'trick' for multiplying two Gaussians is called **completing the square**. The procedure makes use of the equality 

```math
ax^2+bx+c_1 = a\left(x+\frac{b}{2a}\right)^2+c_2
```

"""

# ╔═╡ 0dd8ea56-b7c6-11ef-0116-691b99023eb5
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

"""

# ╔═╡ 36a731be-8795-4bb3-9d0c-6b9f5abe2a53
TODO(
	md"""
	Ik vind dit voorbeeld eigenlijk niet zo nuttig, misschien gelijk naar het voorbeeld met meerdere waarnemingen? Dat is nuttig voor data, deze niet
	"""
)

# ╔═╡ 0dd8f1fe-b7c6-11ef-3386-e37f33577577
md"""
### (Multivariate) Gaussian Multiplication

So, multiplication of two Gaussian distributions yields another (unnormalized) Gaussian with

  * posterior precision equals **sum of prior precisions**
  * posterior precision-weighted mean equals **sum of prior precision-weighted means**

"""

# ╔═╡ 0dd8fbe2-b7c6-11ef-1f78-63dfd48146fd
md"""
As we just saw, a Gaussian prior, combined with a Gaussian likelihood, make Bayesian inference analytically solvable (!):

```math
\begin{equation*}
\underbrace{\text{Gaussian}}_{\text{posterior}}
 \propto \underbrace{\text{Gaussian}}_{\text{likelihood}} \times \underbrace{\text{Gaussian}}_{\text{prior}}
\end{equation*}
```

"""

# ╔═╡ 0dd90644-b7c6-11ef-2fcf-2948d45f43bb
md"""
<a id="Gaussian-multiplication"></a>In general, the multiplication of two multi-variate Gaussians over ``x`` yields an (unnormalized) Gaussian over ``x``:

```math
\begin{equation*}
\boxed{\mathcal{N}(x|\mu_a,\Sigma_a) \cdot \mathcal{N}(x|\mu_b,\Sigma_b) = \underbrace{\mathcal{N}(\mu_a|\, \mu_b, \Sigma_a + \Sigma_b)}_{\text{normalization constant}} \cdot \mathcal{N}(x|\mu_c,\Sigma_c)} \tag{SRG-6}
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

# ╔═╡ 0dd91b7a-b7c6-11ef-1326-7bbfe5ac16bf
md"""
Check out that normalization constant ``\mathcal{N}(\mu_a|\, \mu_b, \Sigma_a + \Sigma_b)``. Amazingly, this constant can also be expressed by a Gaussian!

"""

# ╔═╡ 0dd9264e-b7c6-11ef-0fa9-d3e4e5053654
md"""
```math
\Rightarrow
```

Note that Bayesian inference is trivial in the [*canonical* parameterization of the Gaussian](#natural-parameterization), where we would get

```math
\begin{align*}
 \Lambda_c &= \Lambda_a + \Lambda_b  \quad &&\text{(precisions add)}\\
 \eta_c &= \eta_a + \eta_b \quad &&\text{(precision-weighted means add)}
\end{align*}
```

This property is an important reason why the canonical parameterization of the Gaussian distribution is useful in Bayesian data processing. 

"""

# ╔═╡ 0dd93204-b7c6-11ef-143e-2b7b182f8be1
md"""
### Code Example: Product of Two Gaussian PDFs

Let's plot the exact product of two Gaussian PDFs as well as the normalized product according to the above derivation.

"""

# ╔═╡ 0dd93236-b7c6-11ef-2656-b914f13c4ecd
let
	d1 = Normal(0, 1) # μ=0, σ^2=1
	d2 = Normal(3, 2) # μ=3, σ^2=4
	
	# Calculate the parameters of the product d1*d2
	s2_prod = (d1.σ^-2 + d2.σ^-2)^-1
	m_prod = s2_prod * ((d1.σ^-2)*d1.μ + (d2.σ^-2)*d2.μ)
	d_prod = Normal(m_prod, sqrt(s2_prod)) # Note that we neglect the normalization constant.
	
	# Plot stuff
	x = range(-4, stop=8, length=100)
	plot(x, pdf.(d1,x), label=L"\mathcal{N}(0,1)", fill=(0, 0.1))                                   # Plot the first Gaussian
	plot!(x, pdf.(d2,x), label=L"\mathcal{N}(3,4)", fill=(0, 0.1))                                  # Plot the second Gaussian
	plot!(x, pdf.(d1,x) .* pdf.(d2,x), label=L"\mathcal{N}(0,1) \mathcal{N}(3,4)", fill=(0, 0.1))   # Plot the exact product
	plot!(x, pdf.(d_prod,x), label=L"Z^{-1} \mathcal{N}(0,1) \mathcal{N}(3,4)", fill=(0, 0.1))      # Plot the normalized Gaussian product
end

# ╔═╡ 0dd93f08-b7c6-11ef-3ad5-97d01baafa7c
md"""
### Bayesian Inference with multiple Observations

Now consider that we measure a data set ``D = \{x_1, x_2, \ldots, x_N\}``, with measurements

```math
\begin{aligned}
x_n &= \theta + \epsilon_n \\
\epsilon_n &\sim \mathcal{N}(0,\sigma^2)
\end{aligned}
```

and the same prior for ``\theta``:

```math
\theta \sim \mathcal{N}(\mu_0,\sigma_0^2) \\
```

Let's derive the distribution ``p(x_{N+1}|D)`` for the next sample . 

"""

# ╔═╡ 0dd94cb4-b7c6-11ef-0d42-5f5f3b071afa
md"""
##### inference

First, we derive the posterior for ``\theta``:

```math
\begin{align*}
p(\theta|D) \propto  \underbrace{\mathcal{N}(\theta|\mu_0,\sigma_0^2)}_{\text{prior}} \cdot \underbrace{\prod_{n=1}^N \mathcal{N}(x_n|\theta,\sigma^2)}_{\text{likelihood}}
\end{align*}
```

which is a multiplication of ``N+1`` Gaussians and is therefore also Gaussian-distributed.

"""

# ╔═╡ 0dd96092-b7c6-11ef-08b6-99348eca8529
md"""
Using the property that precisions and precision-weighted means add when Gaussians are multiplied, we can immediately write the posterior 

```math
p(\theta|D) = \mathcal{N} (\theta |\, \mu_N, \sigma_N^2)
```

as 

```math
\begin{align*}
  \frac{1}{\sigma_N^2}  &= \frac{1}{\sigma_0^2} + \sum_n  \frac{1}{\sigma^2} \qquad &\text{(B-2.142)} \\
  \mu_N   &= \sigma_N^2 \, \left( \frac{1}{\sigma_0^2} \mu_0 + \sum_n \frac{1}{\sigma^2} x_n  \right) \qquad &\text{(B-2.141)}
\end{align*}
```

"""

# ╔═╡ 0dd992ee-b7c6-11ef-3add-cdf7452bc514
md"""
##### application: prediction of future sample

We now have a posterior for the model parameters. Let's write down what we know about the next sample ``x_{N+1}``.

```math
\begin{align*}
p(x_{N+1}|D) &= \int p(x_{N+1}|\theta) p(\theta|D)\mathrm{d}\theta \\
  &= \int \mathcal{N}(x_{N+1}|\theta,\sigma^2) \mathcal{N}(\theta|\mu_N,\sigma^2_N) \mathrm{d}\theta \\
  &= \int \mathcal{N}(\theta|x_{N+1},\sigma^2) \mathcal{N}(\theta|\mu_N,\sigma^2_N) \mathrm{d}\theta \\
  &= \int  \mathcal{N}(x_{N+1}|\mu_N, \sigma^2_N +\sigma^2 ) \mathcal{N}(\theta|\cdot,\cdot)\mathrm{d}\theta \tag{use SRG-6} \\
  &= \mathcal{N}(x_{N+1}|\mu_N, \sigma^2_N +\sigma^2 ) \underbrace{\int \mathcal{N}(\theta|\cdot,\cdot)\mathrm{d}\theta}_{=1} \\
  &=\mathcal{N}(x_{N+1}|\mu_N, \sigma^2_N +\sigma^2 )
\end{align*}
```

"""

# ╔═╡ 0dd9a40a-b7c6-11ef-2864-8318d8f3d827
md"""
Uncertainty about ``x_{N+1}`` involved both uncertainty about the parameter (``\sigma_N^2``) and observation noise ``\sigma^2``.

"""

# ╔═╡ 14169498-8a04-446a-bf4d-41f5026a2d4c
TODO(md"``\mu_N`` is not the same as ``\mu_N``")

# ╔═╡ 0dd9b71a-b7c6-11ef-2c4a-a3f9e7f2bc87
md"""
### Maximum Likelihood Estimation for the Gaussian

In order to determine the *maximum likelihood* estimate of ``\theta``, we let ``\sigma_0^2 \rightarrow \infty`` (leads to uniform prior for ``\theta``), yielding $ \frac{1}{\sigma_N^2} = \frac{N}{\sigma^2}$ and consequently

```math
\begin{align*}
  \mu_{\text{ML}}  = \left.\mu_N\right\vert_{\sigma_0^2 \rightarrow \infty} = \sigma_N^2 \, \left(   \frac{1}{\sigma^2}\sum_n  x_n  \right) = \frac{1}{N} \sum_{n=1}^N x_n 
  \end{align*}
```

"""

# ╔═╡ 0dd9ccfa-b7c6-11ef-2379-2967a0b4ad07
md"""
As expected, having an expression for the maximum likelihood estimate, it is now possible to rewrite the (Bayesian) posterior mean for ``\theta`` as 

```math
\begin{align*}
  \underbrace{\mu_N}_{\text{posterior}}   &= \sigma_N^2 \, \left( \frac{1}{\sigma_0^2} \mu_0 + \sum_n \frac{1}{\sigma^2} x_n  \right) \\
  &= \frac{\sigma_0^2 \sigma^2}{N\sigma_0^2 + \sigma^2} \, \left( \frac{1}{\sigma_0^2} \mu_0 + \sum_n \frac{1}{\sigma^2} x_n  \right) \\
  &= \frac{ \sigma^2}{N\sigma_0^2 + \sigma^2}   \mu_0 + \frac{N \sigma_0^2}{N\sigma_0^2 + \sigma^2} \mu_{\text{ML}}   \\
  &= \underbrace{\mu_0}_{\text{prior}} + \underbrace{\underbrace{\frac{N \sigma_0^2}{N \sigma_0^2 + \sigma^2}}_{\text{gain}}\cdot \underbrace{\left(\mu_{\text{ML}} - \mu_0 \right)}_{\text{prediction error}}}_{\text{correction}}\tag{B-2.141}
\end{align*}
```

"""

# ╔═╡ 0dd9db78-b7c6-11ef-1005-73e5d7a4fc4b
md"""
Hence, the posterior mean always lies somewhere between the prior mean ``\mu_0`` and the maximum likelihood estimate (the "data" mean) ``\mu_{\text{ML}}``.

"""

# ╔═╡ 0dd9ed22-b7c6-11ef-19e5-038711d75259
md"""
### Conditioning and Marginalization of a Gaussian

Let ``z = \begin{bmatrix} x \\ y \end{bmatrix}`` be jointly normal distributed as

```math
\begin{align*}
p(z) &= \mathcal{N}(z | \mu, \Sigma) 
  =\mathcal{N} \left( \begin{bmatrix} x \\ y \end{bmatrix} \left| \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix}, 
  \begin{bmatrix} \Sigma_x & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_y \end{bmatrix} \right. \right)
\end{align*}
```

"""

# ╔═╡ 0dd9fb08-b7c6-11ef-0350-c529776149da
md"""
Since covariance matrices are by definition symmetric, it follows that ``\Sigma_x`` and ``\Sigma_y`` are symmetric and ``\Sigma_{xy} = \Sigma_{yx}^T``.

"""

# ╔═╡ 0dda09f4-b7c6-11ef-2429-377131c95b8e
md"""
Let's factorize ``p(z) = p(x,y)`` as ``p(x,y) = p(y|x) p(x)`` through conditioning and marginalization.

"""

# ╔═╡ 0dda16ce-b7c6-11ef-3b84-056673f08e89
md"""
```math
\begin{equation*}
\text{conditioning: }\boxed{ p(y|x) = \mathcal{N}\left(y\,|\,\mu_y + \Sigma_{yx}\Sigma_x^{-1}(x-\mu_x),\, \Sigma_y - \Sigma_{yx}\Sigma_x^{-1}\Sigma_{xy} \right)}
\end{equation*}
```

"""

# ╔═╡ 0dda22f4-b7c6-11ef-05ec-ef5e23c533a1
md"""
```math
\begin{equation*}
\text{marginalization: } \boxed{ p(x) = \mathcal{N}\left( x|\mu_x, \Sigma_x \right)}
\end{equation*}
```

"""

# ╔═╡ 0dda301e-b7c6-11ef-0188-0d6a9782abfa
md"""
**proof**: in Bishop pp.87-89

"""

# ╔═╡ 0dda3d8e-b7c6-11ef-0e2e-9942afc06c32
md"""
Hence, conditioning and marginalization in Gaussians leads to Gaussians again. This is very useful for applications to Bayesian inference in jointly Gaussian systems.

"""

# ╔═╡ 0dda4b3a-b7c6-11ef-17c2-5f5ccd912eee
md"""
With a natural parameterization of the Gaussian ``p(z) = \mathcal{N}_c(z|\eta,\Lambda)`` with precision matrix ``\Lambda = \Sigma^{-1} = \begin{bmatrix} \Lambda_x & \Lambda_{xy} \\ \Lambda_{yx} & \Lambda_y \end{bmatrix}``,  the conditioning operation results in a simpler result, see Bishop pg.90, eqs. 2.96 and 2.97. 

"""

# ╔═╡ 0dda6b2e-b7c6-11ef-14ee-25d9a3acaf11
md"""
As an exercise, interpret the formula for the conditional mean (``\mathbb{E}[y|x]=\mu_y + \Sigma_{yx}\Sigma_x^{-1}(x-\mu_x)``) as a prediction-correction operation.

"""

# ╔═╡ 0dda770e-b7c6-11ef-2988-397f0085c3a3
md"""
### Code Example: Joint, Marginal, and Conditional Gaussian Distributions

Let's plot of the joint, marginal, and conditional distributions.

"""

# ╔═╡ 0dda774a-b7c6-11ef-2750-4960eef0932b
using Plots, LaTeXStrings, Distributions

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
plot(plot_1, plot_2, plot_3, layout=(1,3), size=(1200,300))


# ╔═╡ 0dda842e-b7c6-11ef-24b6-19e2fad91333
md"""
As is clear from the plots, the conditional distribution is a renormalized slice from the joint distribution.

"""

# ╔═╡ 0dda9086-b7c6-11ef-2455-732cd6d69407
md"""
### Example: Conditioning of Gaussian

Consider (again) the system 

```math
\begin{align*}
p(x\,|\,\theta) &= \mathcal{N}(x\,|\,\theta,\sigma^2) \\
p(\theta) &= \mathcal{N}(\theta\,|\,\mu_0,\sigma_0^2)
\end{align*}
```

"""

# ╔═╡ 0dda9d36-b7c6-11ef-1ab4-7b341b8cfcdf
md"""
Let ``z = \begin{bmatrix} x \\ \theta \end{bmatrix}``. The distribution for ``z`` is then given by (Exercise)

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

# ╔═╡ 0ddaa9f4-b7c6-11ef-01a0-a78e551e6414
md"""
Direct substitution of the rule for Gaussian conditioning leads to the <a id="precision-weighted-update">posterior</a> (derivation as an Exercise):

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

# ╔═╡ 0ddab62e-b7c6-11ef-1b65-df9e3d1087d6
md"""
```math
\Rightarrow
```

Moral: For jointly Gaussian systems, we can do inference simply in one step by using the formulas for conditioning and marginalization.

"""

# ╔═╡ 0ddae00e-b7c6-11ef-33f2-b565ce8fc3ba
md"""
### Recursive Bayesian Estimation for Adaptive Signal Processing

Consider the signal ``x_t=\theta+\epsilon_t``, where ``D_t= \left\{x_1,\ldots,x_t\right\}`` is observed *sequentially* (over time).

**Problem**: Derive a recursive algorithm for ``p(\theta|D_t)``, i.e., an update rule for (posterior) ``p(\theta|D_t)`` based on (prior) ``p(\theta|D_{t-1})`` and (new observation) ``x_t``.

"""

# ╔═╡ 0ddafb7a-b7c6-11ef-3c3f-c9fa7af39c92
md"""
##### Model specification

Let's define the estimate after ``t`` observations (i.e., our *solution* ) as ``p(\theta|D_t) = \mathcal{N}(\theta\,|\,\mu_t,\sigma_t^2)``.

We define the joint distribution for ``\theta`` and ``x_t``, given background ``D_{t-1}``, by

```math
\begin{align*} p(x_t,\theta \,|\, D_{t-1}) &= p(x_t|\theta) \, p(\theta|D_{t-1}) \\
  &= \underbrace{\mathcal{N}(x_t\,|\, \theta,\sigma^2)}_{\text{likelihood}} \, \underbrace{\mathcal{N}(\theta\,|\,\mu_{t-1},\sigma_{t-1}^2)}_{\text{prior}}
\end{align*}
```

"""

# ╔═╡ 0ddb085c-b7c6-11ef-34fd-6b1b18a95ff1
md"""
##### Inference

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

# ╔═╡ 0ddb163a-b7c6-11ef-2b06-a1d6677b7191
md"""
This linear *sequential* estimator of mean and variance in Gaussian observations is called a **Kalman Filter**.

<!–- - The new observation ``x_t`` 'updates' the old estimate ``\mu_{t-1}`` by a quantity that is proportional to the *innovation* (or *residual*)  ``\left( x_t - \mu_{t-1} \right)``. –-> 

"""

# ╔═╡ 0ddb2302-b7c6-11ef-1f50-27711dbe4d33
md"""
The so-called Kalman gain ``K_t`` serves as a "learning rate" (step size) in the parameter update equation ``\mu_t = \mu_{t-1} + K_t \cdot (x_t-\mu_{t-1})``. Note that *you* don't need to choose the learning rate. Bayesian inference computes its own (optimal) learning rates.  

"""

# ╔═╡ 0ddb2fa0-b7c6-11ef-3ac5-8979f2a0a00c
md"""
Note that the uncertainty about ``\theta`` decreases over time (since ``0<(1-K_t)<1``). If we assume that the statistics of the system do not change (stationarity), each new sample provides new information about the process, so the uncertainty decreases. 

"""

# ╔═╡ 0ddb3c34-b7c6-11ef-2a77-895cbc5796f3
md"""
Recursive Bayesian estimation as discussed here is the basis for **adaptive signal processing** algorithms such as Least Mean Squares (LMS) and Recursive Least Squares (RLS). Both RLS and LMS are special cases of Recursive Bayesian estimation.

"""

# ╔═╡ 0ddb4b54-b7c6-11ef-121d-5d00e547debd
md"""
### Code Example: Kalman Filter

Let's implement the Kalman filter described above. We'll use it to recursively estimate the value of ``\theta`` based on noisy observations.

"""

# ╔═╡ 0ddb4bb4-b7c6-11ef-373a-ab345190363a
using Plots, Distributions

n = 100         # specify number of observations
θ = 2.0         # true value of the parameter we would like to estimate
noise_σ2 = 0.3  # variance of observation noise

observations = noise_σ2 * randn(n) .+ θ

function perform_kalman_step(prior :: Normal, x :: Float64, noise_σ2 :: Float64)
    K = prior.σ / (noise_σ2 + prior.σ)          # compute the Kalman gain
    posterior_μ = prior.μ + K*(x - prior.μ)     # update the posterior mean
    posterior_σ = prior.σ * (1.0 - K)           # update the posterior standard deviation
    return Normal(posterior_μ, posterior_σ)     # return the posterior distribution
end

post_μ = fill!(Vector{Float64}(undef,n + 1), NaN)     # means of p(θ|D) over time
post_σ2 = fill!(Vector{Float64}(undef,n + 1), NaN)    # variances of p(θ|D) over time

prior = Normal(0, 1)    # specify the prior distribution (you can play with the parameterization of this to get a feeling of how the Kalman filter converges)

post_μ[1] = prior.μ     # save prior mean and variance to show these in plot
post_σ2[1] = prior.σ

for (i, x) in enumerate(observations)                           # note that this loop demonstrates Bayesian learning on streaming data; we update the prior distribution using observation(s), after which this posterior becomes the new prior for future observations
    posterior = perform_kalman_step(prior, x, noise_σ2)         # compute the posterior distribution given the observation
    post_μ[i + 1] = posterior.μ                                 # save the mean of the posterior distribution
    post_σ2[i + 1] = posterior.σ                                # save the variance of the posterior distribution
    prior = posterior                                           # the posterior becomes the prior for future observations
end

obs_scale = collect(2:n+1)
scatter(obs_scale, observations, label=L"D", )  
post_scale = collect(1:n+1)                                                         # scatter the observations
plot!(post_scale, post_μ, ribbon=sqrt.(post_σ2), linewidth=3, label=L"p(θ | D_t)")  # lineplot our estimated means of intermediate posterior distributions
plot!(post_scale, θ*ones(n + 1), linewidth=2, label=L"θ")                           # plot the true value of θ



# ╔═╡ 0ddb7294-b7c6-11ef-0585-3f1a218aeb42
md"""
The shaded area represents 2 standard deviations of posterior ``p(\theta|D)``. The variance of the posterior is guaranteed to decrease monotonically for the standard Kalman filter.

"""

# ╔═╡ 0ddb9904-b7c6-11ef-3808-35b8ee37dd04
md"""
### <a id="product-of-gaussians">Product of Normally Distributed Variables</a>

(We've seen that) the sum of two Gausssian distributed variables is also Gaussian distributed.

"""

# ╔═╡ 0ddba9ee-b7c6-11ef-3148-9db5fbb13d77
md"""
Has the *product* of two Gaussian distributed variables also a Gaussian distribution?

"""

# ╔═╡ 0ddbba2e-b7c6-11ef-04cf-1119024af1d1
md"""
**No**! In general this is a difficult computation. As an example, let's compute ``p(z)`` for ``Z=XY`` for the special case that ``X\sim \mathcal{N}(0,1)`` and ``Y\sim \mathcal{N}(0,1)``.

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

# ╔═╡ 0ddbc78a-b7c6-11ef-2ce4-f76fa4153e4b
md"""
### Code Example: Product of Gaussian Distributions

We plot ``p(Z=XY)`` and ``p(X)p(Y)`` for ``X\sim\mathcal{N}(0,1)`` and ``Y \sim \mathcal{N}(0,1)`` to give an idea of how these distributions differ.

"""

# ╔═╡ 0ddbc7c8-b7c6-11ef-004f-8bfaa5f29eba
using Plots, Distributions, SpecialFunctions, LaTeXStrings
X = Normal(0,1)
Y = Normal(0,1)
pdf_product_std_normals(z::Vector) = (besselk.(0, abs.(z))./π)
range1 = collect(range(-4,stop=4,length=100))
plot(range1, pdf.(X, range1), label=L"p(X)=p(Y)=\mathcal{N}(0,1)", fill=(0, 0.1))
plot!(range1, pdf.(X,range1).*pdf.(Y,range1), label=L"p(X)*p(Y)", fill=(0, 0.1))
plot!(range1, pdf_product_std_normals(range1), label=L"p(Z=X*Y)", fill=(0, 0.1))

# ╔═╡ 0ddbd3ce-b7c6-11ef-20e1-070d736f7b95
md"""
In short, Gaussian-distributed variables remain Gaussian in linear systems, but this is not the case in non-linear systems. 

"""

# ╔═╡ 0ddbf246-b7c6-11ef-16a5-bbf396f80915
md"""
### Solution to Example Problem

We apply maximum likelihood estimation to fit a 2-dimensional Gaussian model (``m``) to data set ``D``. Next, we evaluate ``p(x_\bullet \in S | m)`` by (numerical) integration of the Gaussian pdf over ``S``: ``p(x_\bullet \in S | m) = \int_S p(x|m) \mathrm{d}x``.

"""

# ╔═╡ 0ddbf278-b7c6-11ef-20f5-7ffd3163b14f
using HCubature, LinearAlgebra, Plots, Distributions# Numerical integration package
# Maximum likelihood estimation of 2D Gaussian
N = length(sum(D,dims=1))
μ = 1/N * sum(D,dims=2)[:,1]
D_min_μ = D - repeat(μ, 1, N)
Σ = Hermitian(1/N * D_min_μ*D_min_μ')
m = MvNormal(μ, convert(Matrix, Σ));

contour(range(-3, 4, length=100), range(-3, 4, length=100), (x, y) -> pdf(m, [x, y]))

# Numerical integration of p(x|m) over S:
(val,err) = hcubature((x)->pdf(m,x), [0., 1.], [2., 2.])
println("p(x⋅∈S|m) ≈ $(val)")

scatter!(D[1,:], D[2,:], marker=:x, markerstrokewidth=3, label=L"D")
scatter!([x_dot[1]], [x_dot[2]], label=L"x_\bullet")
plot!(range(0, 2), [1., 1., 1.], fillrange=2, alpha=0.4, color=:gray, label=L"S")

# ╔═╡ 0ddc02d6-b7c6-11ef-284e-018c7895536e
md"""
### Summary

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

# ╔═╡ 0ddc1028-b7c6-11ef-1eec-6d72e52f4431
md"""
## <center> OPTIONAL SLIDES</center>

"""

# ╔═╡ 0ddc1c2e-b7c6-11ef-00b6-e98913a96420
md"""
### <a id="inference-for-precision">Inference for the Precision Parameter of the Gaussian</a>

Again, we consider an observed data set ``D = \{x_1, x_2, \ldots, x_N\}`` and try to explain these data by a Gaussian distribution.

"""

# ╔═╡ 0ddc287e-b7c6-11ef-1f72-910e6e7b06bb
md"""
We discussed earlier Bayesian inference for the mean with a given variance. Now we will derive a posterior for the variance if the mean is given. (Technically, we will do the derivation for a precision parameter ``\lambda = \sigma^{-2}``, since the discussion is a bit more straightforward for the precision parameter).

"""

# ╔═╡ 0ddc367a-b7c6-11ef-38f9-09fb462987dc
md"""
##### model specification

The likelihood for the precision parameter is 

```math
\begin{align*}
p(D|\lambda) &= \prod_{n=1}^N \mathcal{N}\left(x_n \,|\, \mu, \lambda^{-1} \right) \\
  &\propto \lambda^{N/2} \exp\left\{ -\frac{\lambda}{2}\sum_{n=1}^N \left(x_n - \mu \right)^2\right\} \tag{B-2.145}
\end{align*}
```

"""

# ╔═╡ 0ddc4796-b7c6-11ef-2156-8b3d6899a8c0
md"""
The conjugate distribution for this function of ``\lambda`` is the [*Gamma* distribution](https://en.wikipedia.org/wiki/Gamma_distribution), given by

```math
p(\lambda\,|\,a,b) = \mathrm{Gam}\left( \lambda\,|\,a,b \right) \triangleq \frac{1}{\Gamma(a)} b^{a} \lambda^{a-1} \exp\left\{ -b \lambda\right\}\,, \tag{B-2.146}
```

where ``a>0`` and ``b>0`` are known as the *shape* and *rate* parameters, respectively. 

<img src="./figures/B-fig-2.13.png" width="600px">

(Bishop fig.2.13). Plots of the Gamma distribution ``\mathrm{Gam}\left( \lambda\,|\,a,b \right) $ for different values of $a`` and ``b``.

"""

# ╔═╡ 0ddc55f6-b7c6-11ef-3975-9d92d7e3feca
md"""
The mean and variance of the Gamma distribution evaluate to ``\mathrm{E}\left( \lambda\right) = \frac{a}{b}`` and ``\mathrm{var}\left[\lambda\right] = \frac{a}{b^2}``. 

"""

# ╔═╡ 0ddc7284-b7c6-11ef-0c0c-bd949a5ef015
md"""
##### inference

We will consider a prior ``p(\lambda) = \mathrm{Gam}\left( \lambda\,|\,a_0, b_0\right)``, which leads by Bayes rule to the posterior

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

# ╔═╡ 0ddc7f40-b7c6-11ef-0253-6342085f708a
md"""
Hence the **posterior is again a Gamma distribution**. By inspection of B-2.150 and B-2.151, we deduce that we can interpret ``2a_0`` as the number of a priori (pseudo-)observations. 

"""

# ╔═╡ 0ddc8b70-b7c6-11ef-13cb-3daa72032cf9
md"""
Since the most uninformative prior is given by ``a_0=b_0 \rightarrow 0``, we can derive the **maximum likelihood estimate** for the precision as

```math
\lambda_{\text{ML}} = \left.\mathrm{E}\left[ \lambda\right]\right\vert_{a_0=b_0\rightarrow 0} = \left. \frac{a_N}{b_N}\right\vert_{a_0=b_0\rightarrow 0} = \frac{N}{\sum_{n=1}^N \left(x_n-\mu \right)^2}
```

"""

# ╔═╡ 0ddc9aac-b7c6-11ef-3d8e-f5a5d0e715f8
md"""
In short, if we do density estimation with a Gaussian distribution ``\mathcal{N}\left(x_n\,|\,\mu,\sigma^2 \right)`` for an observed data set ``D = \{x_1, x_2, \ldots, x_N\}``, the <a id="ML-for-Gaussian">maximum likelihood estimates</a> for ``\mu`` and ``\sigma^2`` are given by

```math
\begin{align*}
\mu_{\text{ML}} &= \frac{1}{N} \sum_{n=1}^N x_n \qquad &&\text{(B-2.121)} \\
\sigma^2_{\text{ML}} &= \frac{1}{N} \sum_{n=1}^N \left(x_n - \mu_{\text{ML}} \right)^2 \qquad &&\text{(B-2.122)}
\end{align*}
```

These estimates are also known as the *sample mean* and *sample variance* respectively. 

"""

# ╔═╡ 0ddc9ae8-b7c6-11ef-33a5-771f934e6ae8
md"""
open("../../styles/aipstyle.html") do f
    display("text/html", read(f, String))
end
""" |> TODO

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Integrals = "de52edbc-65ea-441a-8357-d3a637375a31"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
RxInfer = "86711068-29c9-4ff7-b620-ae75d7495b3d"

[compat]
Distributions = "~0.25.113"
Integrals = "~4.5.0"
LaTeXStrings = "~1.4.0"
Plots = "~1.40.9"
PlutoTeachingTools = "~0.3.1"
PlutoUI = "~0.7.60"
RxInfer = "~3.7.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.9"
manifest_format = "2.0"
project_hash = "82b35afbc231f5f9f5a93f4006cb6e8a412d8881"

[[deps.ADTypes]]
git-tree-sha1 = "e2478490447631aedba0823d4d7a80b2cc8cdb32"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.14.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "9606d7832795cbef89e06a550475be300364a8aa"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.19.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "4e25216b8fea1908a0ce0f5d87368587899f75be"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BayesBase]]
deps = ["Distributions", "DomainSets", "LinearAlgebra", "LoopVectorization", "Random", "SpecialFunctions", "StaticArrays", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "TinyHugeNumbers"]
git-tree-sha1 = "06664ca85dc72f940617c9d10bd3dd099084f36c"
uuid = "b4ee3484-f114-42fe-b91c-797d54a0c67e"
version = "1.5.4"
weakdeps = ["FastCholesky"]

    [deps.BayesBase.extensions]
    FastCholeskyExt = "FastCholesky"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BitSetTuples]]
deps = ["TupleTools"]
git-tree-sha1 = "aa19428fb6ad21db22f8568f068de4f443d3bacc"
uuid = "0f2f92aa-23a3-4d05-b791-88071d064721"
version = "1.1.5"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "a8c0f363186263d75e97a41878d10dd842797561"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.6.3"

    [deps.BlockArrays.extensions]
    BlockArraysAdaptExt = "Adapt"
    BlockArraysBandedMatricesExt = "BandedMatrices"

    [deps.BlockArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "5a97e67919535d6841172016c9530fd69494e5ec"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.6"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "062c5e1a5bf6ada13db96a4ae4749a4c2234f521"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.9"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

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

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

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

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "a86af9c4c4f33e16a2b2ff43c2113b2f390081fa"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.5"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "210933c93f39f832d92f9efbbe69a49c453db36d"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.1"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

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

[[deps.DomainIntegrals]]
deps = ["CompositeTypes", "DomainSets", "FastGaussQuadrature", "GaussQuadrature", "HCubature", "IntervalSets", "LinearAlgebra", "QuadGK", "StaticArrays"]
git-tree-sha1 = "95c6b8fd44ee7e41d166c1adf7b1c94309be6195"
uuid = "cc6bae93-f070-4015-88fd-838f9505a86c"
version = "0.4.6"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "a7e9f13f33652c533d49868a534bfb2050d1365f"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.7.15"

    [deps.DomainSets.extensions]
    DomainSetsMakieExt = "Makie"

    [deps.DomainSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EnumX]]
git-tree-sha1 = "bddad79635af6aec424f53ed8aad5d7555dc6f00"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.5"

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

[[deps.ExponentialFamily]]
deps = ["BayesBase", "BlockArrays", "Distributions", "DomainSets", "FastCholesky", "FillArrays", "ForwardDiff", "HCubature", "HypergeometricFunctions", "IntervalSets", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "LoopVectorization", "PositiveFactorizations", "Random", "SparseArrays", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "TinyHugeNumbers"]
git-tree-sha1 = "91857b13ac2767b830afb867c7dc8734f6b89962"
uuid = "62312e5e-252a-4322-ace9-a5f4bf9b357b"
version = "1.7.1"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExproniconLite]]
git-tree-sha1 = "c13f0b150373771b0fdc1713c97860f8df12e6c2"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.14"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FastCholesky]]
deps = ["LinearAlgebra", "PositiveFactorizations"]
git-tree-sha1 = "1c0a81e006e40e9fcbd5f6f6cb42ac2700f86889"
uuid = "2d5283b6-8564-42b6-bb00-83ed8e915756"
version = "1.4.3"
weakdeps = ["StaticArraysCore"]

    [deps.FastCholesky.extensions]
    StaticArraysCoreExt = "StaticArraysCore"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "0f478d8bad6f52573fb7658a263af61f3d96e43a"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "0.5.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "b66970a70db13f45b7e57fbda1736e1cf72174ea"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

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

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "f089ab1f834470c525562030c8cfde4025d5e915"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.27.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedArguments]]
deps = ["TupleTools"]
git-tree-sha1 = "befa1ad59c77643dec6fc20d71fd6f5c3afcdadd"
uuid = "4130a065-6d82-41fe-881e-7a5c65156f7d"
version = "0.1.1"

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

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

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

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "4424dca1462cc3f19a0e6f07b809ad948ac1d62b"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.16"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d7ecfaca1ad1886de4f9053b5b8aef34f36ede7f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.16+0"

[[deps.GaussQuadrature]]
deps = ["SpecialFunctions"]
git-tree-sha1 = "eb6f1f48aa994f3018cbd029a17863c6535a266d"
uuid = "d54b0c1a-921d-58e0-8e36-89d8069c0969"
version = "0.5.8"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "fee60557e4f19d0fe5cd169211fdda80e494f4e8"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.84.0+0"

[[deps.GraphPPL]]
deps = ["BitSetTuples", "DataStructures", "Dictionaries", "MacroTools", "MetaGraphsNext", "NamedTupleTools", "Static", "StaticArrays", "TupleTools", "Unrolled"]
git-tree-sha1 = "896e1022b4c9372f2e31d25ac1eca0fb38ac6eb1"
uuid = "b3f8163a-e979-4e85-b43e-1f63d8c8b42c"
version = "4.3.4"

    [deps.GraphPPL.extensions]
    GraphPPLDistributionsExt = "Distributions"
    GraphPPLPlottingExt = ["Cairo", "GraphPlot"]

    [deps.GraphPPL.weakdeps]
    Cairo = "159f3aea-2a34-519c-b102-8c37f9878175"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    GraphPlot = "a2cc645c-3eea-5389-862e-a155d0052231"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "c5abfa0ae0aaee162a3fbb053c13ecda39be545b"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.0"

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

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "8e070b599339d622e9a081d17230d74a5c473293"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.17"

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

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.Integrals]]
deps = ["CommonSolve", "HCubature", "LinearAlgebra", "MonteCarloIntegration", "QuadGK", "Random", "Reexport", "SciMLBase"]
git-tree-sha1 = "cfdc4fb8d21c8f596572a59912ae863774123622"
uuid = "de52edbc-65ea-441a-8357-d3a637375a31"
version = "4.5.0"

    [deps.Integrals.extensions]
    IntegralsArblibExt = "Arblib"
    IntegralsCubaExt = "Cuba"
    IntegralsCubatureExt = "Cubature"
    IntegralsFastGaussQuadratureExt = "FastGaussQuadrature"
    IntegralsForwardDiffExt = "ForwardDiff"
    IntegralsMCIntegrationExt = "MCIntegration"
    IntegralsZygoteExt = ["Zygote", "ChainRulesCore"]

    [deps.Integrals.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Cuba = "8a292aeb-7a57-582c-b821-06e4c11590b1"
    Cubature = "667455a9-e2ce-5579-9412-b964f529a492"
    FastGaussQuadrature = "442a2c76-b920-505d-bb47-c5924d526838"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    MCIntegration = "ea1e2de9-7db7-4b42-91ee-0cd1bf6df167"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
git-tree-sha1 = "5fbb102dcb8b1a858111ae81d56682376130517d"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.11"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "8e071648610caa2d3a5351aba03a936a0c37ec61"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.13"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.Jieko]]
deps = ["ExproniconLite"]
git-tree-sha1 = "2f05ed29618da60c06a87e9c033982d4f71d0b6c"
uuid = "ae98c720-c025-4a4a-838c-29b094483192"
version = "0.2.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "6ac9e4acc417a5b534ace12690bc6973c25b862f"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.3"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

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
git-tree-sha1 = "4f34eaabe49ecb3fb0d58d6015e32fd31a733199"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.8"

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

[[deps.LatticeRules]]
deps = ["Random"]
git-tree-sha1 = "7f5b02258a3ca0221a6a9710b0a0a2e8fb4957fe"
uuid = "73f95e8e-ec14-4e6a-8b18-0d2e271c4e55"
version = "0.0.1"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "SparseArrays"]
git-tree-sha1 = "866ce84b15e54d758c11946aacd4e5df0e60b7a3"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "2.6.1"

    [deps.LazyArrays.extensions]
    LazyArraysBandedMatricesExt = "BandedMatrices"
    LazyArraysBlockArraysExt = "BlockArrays"
    LazyArraysBlockBandedMatricesExt = "BlockBandedMatrices"
    LazyArraysStaticArraysExt = "StaticArrays"

    [deps.LazyArrays.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

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

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "4adee99b7262ad2a1a4bbbc59d993d24e55ea96f"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.4.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "e5afce7eaf5b5ca0d444bcb4dc4fd78c54cbbac0"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.172"

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.LoopVectorization.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "4ef1c538614e3ec30cb6383b9eb0326a5c3a9763"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.3.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixCorrectionTools]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "73f93b21eae5714c282396bfae9d9f13d6ad04b6"
uuid = "41f81499-25de-46de-b591-c3cfc21e9eaf"
version = "1.2.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MetaGraphsNext]]
deps = ["Graphs", "JLD2", "SimpleTraits"]
git-tree-sha1 = "1e3b196ecbbf221d4d3696ea9de4288bea4c39f9"
uuid = "fa8bd995-216d-47f1-8a91-f3b68fbeb377"
version = "0.7.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MonteCarloIntegration]]
deps = ["Distributions", "QuasiMonteCarlo", "Random"]
git-tree-sha1 = "722ad522068d31954b4a976b66a26aeccbf509ed"
uuid = "4886b29c-78c9-11e9-0a6e-41e1f4161f7b"
version = "0.2.0"

[[deps.Moshi]]
deps = ["ExproniconLite", "Jieko"]
git-tree-sha1 = "453de0fc2be3d11b9b93ca4d0fddd91196dcf1ed"
uuid = "2e0e35c7-a2e4-4343-998d-7ef72827ed2d"
version = "0.3.5"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "25a6638571a902ecfb1ae2a18fc1575f86b1d4df"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.10.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NamedTupleTools]]
git-tree-sha1 = "90914795fc59df44120fe3fff6742bb0d7adb1d0"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.14.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

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
git-tree-sha1 = "9216a80ff3682833ac4b733caa8c00390620ba5d"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.0+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["Compat", "EnumX", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "61942645c38dd2b5b78e2082c9b51ab315315d10"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.13.2"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

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

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

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
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

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
git-tree-sha1 = "28ea788b78009c695eb0d637587c81d26bdf0e36"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.14"

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

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoLinks", "PlutoUI"]
git-tree-sha1 = "8252b5de1f81dc103eb0293523ddf917695adea1"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.3.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3151a0c8061cc3f887019beebf359e6c4b3daa08"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.65"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "13c5103482a8ed1536a54c08d0e742ae3dca2d42"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.4"

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
git-tree-sha1 = "2766344a35a1a5ec1147305c4b343055d7c22c90"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.QuasiMonteCarlo]]
deps = ["Accessors", "ConcreteStructs", "LatticeRules", "LinearAlgebra", "Primes", "Random", "Requires", "Sobol", "StatsBase"]
git-tree-sha1 = "cc086f8485bce77b6187141e1413c3b55f9a4341"
uuid = "8a4e6c94-4038-4cdc-81c3-7e6ffdb2a71b"
version = "0.3.3"
weakdeps = ["Distributions"]

    [deps.QuasiMonteCarlo.extensions]
    QuasiMonteCarloDistributionsExt = "Distributions"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.ReactiveMP]]
deps = ["BayesBase", "DataStructures", "DiffResults", "Distributions", "DomainIntegrals", "DomainSets", "ExponentialFamily", "FastCholesky", "FastGaussQuadrature", "FixedArguments", "ForwardDiff", "HCubature", "LazyArrays", "LinearAlgebra", "LoopVectorization", "MacroTools", "MatrixCorrectionTools", "Optim", "PositiveFactorizations", "Random", "Rocket", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "TinyHugeNumbers", "TupleTools", "Unrolled"]
git-tree-sha1 = "bcd97efd053980260aacdbd7f1833b07b13688fe"
uuid = "a194aa59-28ba-4574-a09c-4a745416d6e3"
version = "4.4.6"

    [deps.ReactiveMP.extensions]
    ReactiveMPOptimisersExt = "Optimisers"
    ReactiveMPProjectionExt = "ExponentialFamilyProjection"
    ReactiveMPRequiresExt = "Requires"

    [deps.ReactiveMP.weakdeps]
    ExponentialFamilyProjection = "17f509fa-9a96-44ba-99b2-1c5f01f0931b"
    Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"

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

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "2e154f7d7e38db1af0a14ec751aba33360c3bef9"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.33.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

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

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "f6f7d30fb0d61c64d0cfe56cf085a7c9e7d5bc80"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.8.0"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

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

[[deps.Rocket]]
deps = ["DataStructures", "Sockets", "Unrolled"]
git-tree-sha1 = "af6e944256dc654a534082f08729afc1189933e4"
uuid = "df971d30-c9d6-4b37-b8ff-e965b2cb3a40"
version = "1.8.2"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "86a8a8b783481e1ea6b9c91dd949cb32191f8ab4"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.15"

[[deps.RxInfer]]
deps = ["BayesBase", "DataStructures", "Distributions", "DomainSets", "ExponentialFamily", "FastCholesky", "GraphPPL", "LinearAlgebra", "MacroTools", "Optim", "ProgressMeter", "Random", "ReactiveMP", "Reexport", "Rocket", "Static", "TupleTools"]
git-tree-sha1 = "23b3a5b73c1208ac9ea65a78a9fa4e8b1dfbf178"
uuid = "86711068-29c9-4ff7-b620-ae75d7495b3d"
version = "3.7.2"

    [deps.RxInfer.extensions]
    ProjectionExt = "ExponentialFamilyProjection"

    [deps.RxInfer.weakdeps]
    ExponentialFamilyProjection = "17f509fa-9a96-44ba-99b2-1c5f01f0931b"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "456f610ca2fbd1c14f5fcf31c6bfadc55e7d66e0"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.43"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Moshi", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "a3fa02bf0ac97856e2f9129f5fb33a4b9ba0bae1"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.102.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseMLStyleExt = "MLStyle"
    SciMLBaseMakieExt = "Makie"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = ["Zygote", "ChainRulesCore"]

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    MLStyle = "d8e11817-5142-5d16-987a-aa16d5891078"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["Accessors", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "3249fe77f322fe539e935ecb388c8290cd38a3fc"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "1.3.1"
weakdeps = ["SparseArrays", "StaticArraysCore"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsSparseArraysExt = "SparseArrays"
    SciMLOperatorsStaticArraysCoreExt = "StaticArraysCore"

[[deps.SciMLStructures]]
deps = ["ArrayInterface"]
git-tree-sha1 = "566c4ed301ccb2a44cbd5a27da5f885e0ed1d5df"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sobol]]
deps = ["DelimitedFiles", "Random"]
git-tree-sha1 = "5a74ac22a9daef23705f010f72c81d6925b19df8"
uuid = "ed01d8cd-4d21-5b2a-85b4-cc3bdc58bad4"
version = "1.5.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

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

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "f737d444cb0ad07e61b3c1bef8eb91203c321eff"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.2.0"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"

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
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "b81c5035922cc89c2d9523afc6c54be512411466"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.5"

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

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "PrettyTables", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "b6a641e38efa01355aa721246dd246e10c7dcd4d"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.40"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

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

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "d969183d3d244b6c33796b5ed01ab97328f2db85"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.5"

[[deps.TinyHugeNumbers]]
git-tree-sha1 = "c8760444248aef64bc728b340ebc50df13148c93"
uuid = "783c9a47-75a3-44ac-a16b-f1ab7b3acf04"
version = "1.0.2"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.TupleTools]]
git-tree-sha1 = "41e43b9dc950775eac654b9f845c839cd2f1821e"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.6.0"

[[deps.URIs]]
git-tree-sha1 = "24c1c558881564e2217dcf7840a8b2e10caeb0f9"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d2282232f8a4d71f79e85dc4dd45e5b12a6297fb"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.23.1"
weakdeps = ["ConstructionBase", "ForwardDiff", "InverseFunctions", "Printf"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    PrintfExt = "Printf"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "af305cc62419f9bd61b6644d19170a4d258c7967"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.7.0"

[[deps.Unrolled]]
deps = ["MacroTools"]
git-tree-sha1 = "6cc9d682755680e0f0be87c56392b7651efc2c7b"
uuid = "9602ed7d-8fef-5bc8-8597-8f21381861e8"
version = "0.1.5"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "4ab62a49f1d8d9548a1c8d1a75e5f55cf196f64e"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.71"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "XML2_jll"]
git-tree-sha1 = "49be0be57db8f863a902d59c0083d73281ecae8e"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.23.1+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "54b8a029ac145ebe8299463447fd1590b2b1d92f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.44.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

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
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

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
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

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
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cd155272a3738da6db765745b89e466fa64d0830"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.49+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "c950ae0a3577aec97bfccf3381f66666bc416729"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.8.1+0"
"""

# ╔═╡ Cell order:
# ╟─3eeabdc9-1906-4578-a2d0-cdf28246a24d
# ╟─246dc845-2a07-4109-b6df-72722cc3e8a2
# ╟─ce5cf62a-bd84-4abe-8cef-07fde003624d
# ╠═41bc3e62-e454-4f14-9d3c-c725a5a93aa8
# ╟─24125436-1b30-4d05-8502-3c8ede5cc00f
# ╟─29f9c097-8bca-4339-a927-28d101cb3a22
# ╟─7fad63f9-c986-438f-99ec-bc8b8bf70ce5
# ╠═cc17e180-89f1-48b7-a9dd-fa1e7814ceb7
# ╠═b9aa17ac-ff44-4899-b15a-e2136eaadbed
# ╟─fa5e50f5-56b8-48c0-83a0-031fade560d6
# ╟─108ba4f6-9292-45a5-a4f3-bb560b43d31b
# ╟─3a99575c-2773-4b86-814c-465a74d297cf
# ╟─89c630f6-e7ef-4284-86cd-5eaa3383735e
# ╟─adac696f-1b5e-4af4-bd45-4a13e8606332
# ╟─b3f91134-c64c-49cc-b9a6-4da1fe2a37ad
# ╟─3271e5a2-d749-4a7c-801f-f1cb692c74ba
# ╟─21f842fe-e37a-4154-8561-ef814ccd3e54
# ╟─4ffd7bad-3d46-43f5-a0f3-d8a642b98f4c
# ╟─0501b18c-cdc9-4772-917f-195c75f05118
# ╟─57df5ab6-a9ae-4c8f-8147-78b7e16988d4
# ╟─9e0d5434-6953-4e7c-884b-9ef607441e1b
# ╟─7384f4ee-a889-44db-842b-df30ed1e6a49
# ╟─bc13ef80-d16f-48c2-b699-3c1043927c8e
# ╟─eed3b2da-8eb6-4462-a852-d39e95fdba48
# ╠═db1a7392-825a-454c-a777-954c05a6a310
# ╠═80612314-bf45-4bb8-9316-a910716ca459
# ╠═25032652-bf9b-4165-9c2f-15a6298145f4
# ╠═1c432cb9-bcae-44aa-a32e-1fd142a938b1
# ╠═75c3d336-f4dc-43d4-a9fe-d026d37dac66
# ╟─05938560-9159-4528-af37-c0451fd4aca0
# ╟─bd9e3aa4-e6a2-4b4e-bd45-ed9a80374cb4
# ╟─0dd817c0-b7c6-11ef-1f8b-ff0f59f7a8ce
# ╠═f4e55044-ff65-4c21-aeb3-2d8be2b97cf4
# ╠═06cad4f9-b40c-428f-b2b9-686274749dc9
# ╟─0dd82814-b7c6-11ef-3927-b3ec0b632c31
# ╠═452300fb-2cf2-4fab-8923-b847ed4b0c97
# ╠═a3638872-847f-47a2-941d-94b9c7d2218b
# ╠═ff222c45-8909-489d-b4a2-ec7d9914ede4
# ╠═5c9b660b-d44a-402f-a1f8-738911a183b3
# ╠═d34233a3-60fe-4ac6-9822-b6097749206b
# ╠═cd6cc0a8-eb41-4366-82e4-2b6ee92c7a37
# ╠═0aa00f04-a53e-4c2b-9b54-a9f93977c481
# ╠═76cb64ad-e3e7-4af1-b5b3-4bd73f68883b
# ╠═07b6e6be-eed0-4a89-bcdb-52f3ed79f100
# ╟─085e9093-e3d1-41b7-8b3e-ae623809f13d
# ╠═0a373d0f-601d-4fce-a1d5-e02732ece9fe
# ╠═b5e23f3d-b35a-40f1-8bd4-58d6128f2e29
# ╟─51fb9204-b480-47f8-bb1e-515b5a0af730
# ╠═3ca80762-1e7b-46c8-a101-9206de278077
# ╠═636389e3-f169-4255-9b4a-8d3426d042b0
# ╟─042bab9b-e162-436e-894d-70e7ea98fb08
# ╟─0dd835ca-b7c6-11ef-0e33-1329e4ba13d8
# ╟─0dd84542-b7c6-11ef-3115-0f8b26aeaa5d
# ╟─0dd8528a-b7c6-11ef-3bc9-eb09c0c530d8
# ╟─0dd85c94-b7c6-11ef-06dc-7b8797c13fda
# ╟─0dd8677a-b7c6-11ef-357f-2328b10f5274
# ╟─0dd86f40-b7c6-11ef-2ae8-a3954469bcee
# ╟─0dd87a3a-b7c6-11ef-2bc2-bf2b4969537c
# ╠═5978ce29-3fd1-44c1-a3cf-37fd336e2a35
# ╠═0837d757-9beb-4956-836c-a2c487bea9e3
# ╠═2c31df74-726b-48ff-a402-5a9f67392060
# ╠═09eb45b0-4117-46ae-adaf-fe1e82578478
# ╠═8554d2d0-0acf-47a1-92e1-9b025031a5d6
# ╟─e90979ad-cfd8-4fe8-9bd3-a434e394a8f2
# ╠═3c454c77-2320-49b8-b95c-e869f9e9dcd9
# ╟─272ae2be-7784-4100-a03f-4b3aa400dd18
# ╠═2ed19557-cf6b-4a2d-819a-0638b17094a6
# ╠═4acf4347-91c9-4d36-a6d3-2764f0c2296c
# ╠═61c0c5e5-5d52-47d0-973a-a006ad538870
# ╠═7b3c8631-51c9-4e4d-8d69-42999dc50858
# ╠═810a5527-0b40-49d0-bd3f-9007e09e2ba3
# ╠═d651f10f-8a02-489c-9f3c-93ba4c59250b
# ╠═aa4eb90e-2166-4407-97aa-666f309d5e34
# ╠═470a0c46-3eae-490b-b13b-5a74eae16a3d
# ╠═6108deb0-2c3f-41c2-819d-106f48d210e7
# ╠═a6e21f35-5563-43b4-a6cf-8526be3b5f25
# ╠═1b26aadc-62a1-4ac1-8cf4-ecdbac7ef63f
# ╠═2033d8df-7e4c-462f-9c82-f2452460e80a
# ╠═82d49f7f-cd7f-4406-a883-641069025699
# ╠═806c47e9-6415-424a-9ee3-67699dbca385
# ╟─0dd544c8-b7c6-11ef-106b-99e6f84894c3
# ╠═a9ab8f29-d72b-472a-b038-c5a884808e2f
# ╠═b738c85f-09f1-4c99-801e-daa90f476fce
# ╠═a0427b32-7ce2-46a9-b962-7fab500c9b2f
# ╟─a72dadff-e813-4b18-8fa9-86c9b1515fe2
# ╠═0dd88110-b7c6-11ef-0b82-2ffe13a68cad
# ╟─0dd88a84-b7c6-11ef-133c-3d85f0703c19
# ╟─c50f6bf0-a1e5-4bd6-a6a2-7201a4302c2f
# ╟─d93fd4bb-7bf2-453b-97be-602ee494a786
# ╟─9612754a-7ed3-4d4d-b4a9-9db3c2059271
# ╠═5dd56dd1-3db0-4b5e-bf74-1ba47f04b58a
# ╟─0dd890ee-b7c6-11ef-04b7-e7671227d8cb
# ╟─0dd89b6e-b7c6-11ef-2525-73ee0242eb91
# ╟─0dd8b5d6-b7c6-11ef-1eb9-4f4289261e79
# ╟─0dd8c024-b7c6-11ef-3ca4-f9e8286cbb64
# ╟─0dd8d976-b7c6-11ef-051f-4f6cb3db3d1b
# ╟─0dd8df66-b7c6-11ef-011a-8d90bba8e2cd
# ╟─0dd8ea56-b7c6-11ef-0116-691b99023eb5
# ╟─36a731be-8795-4bb3-9d0c-6b9f5abe2a53
# ╟─0dd8f1fe-b7c6-11ef-3386-e37f33577577
# ╟─0dd8fbe2-b7c6-11ef-1f78-63dfd48146fd
# ╟─0dd90644-b7c6-11ef-2fcf-2948d45f43bb
# ╟─0dd91b7a-b7c6-11ef-1326-7bbfe5ac16bf
# ╟─0dd9264e-b7c6-11ef-0fa9-d3e4e5053654
# ╟─0dd93204-b7c6-11ef-143e-2b7b182f8be1
# ╠═0dd93236-b7c6-11ef-2656-b914f13c4ecd
# ╟─0dd93f08-b7c6-11ef-3ad5-97d01baafa7c
# ╟─0dd94cb4-b7c6-11ef-0d42-5f5f3b071afa
# ╟─0dd96092-b7c6-11ef-08b6-99348eca8529
# ╟─0dd992ee-b7c6-11ef-3add-cdf7452bc514
# ╟─0dd9a40a-b7c6-11ef-2864-8318d8f3d827
# ╠═14169498-8a04-446a-bf4d-41f5026a2d4c
# ╟─0dd9b71a-b7c6-11ef-2c4a-a3f9e7f2bc87
# ╟─0dd9ccfa-b7c6-11ef-2379-2967a0b4ad07
# ╟─0dd9db78-b7c6-11ef-1005-73e5d7a4fc4b
# ╟─0dd9ed22-b7c6-11ef-19e5-038711d75259
# ╟─0dd9fb08-b7c6-11ef-0350-c529776149da
# ╟─0dda09f4-b7c6-11ef-2429-377131c95b8e
# ╟─0dda16ce-b7c6-11ef-3b84-056673f08e89
# ╟─0dda22f4-b7c6-11ef-05ec-ef5e23c533a1
# ╟─0dda301e-b7c6-11ef-0188-0d6a9782abfa
# ╟─0dda3d8e-b7c6-11ef-0e2e-9942afc06c32
# ╟─0dda4b3a-b7c6-11ef-17c2-5f5ccd912eee
# ╟─0dda6b2e-b7c6-11ef-14ee-25d9a3acaf11
# ╟─0dda770e-b7c6-11ef-2988-397f0085c3a3
# ╠═0dda774a-b7c6-11ef-2750-4960eef0932b
# ╟─0dda842e-b7c6-11ef-24b6-19e2fad91333
# ╟─0dda9086-b7c6-11ef-2455-732cd6d69407
# ╟─0dda9d36-b7c6-11ef-1ab4-7b341b8cfcdf
# ╟─0ddaa9f4-b7c6-11ef-01a0-a78e551e6414
# ╟─0ddab62e-b7c6-11ef-1b65-df9e3d1087d6
# ╟─0ddae00e-b7c6-11ef-33f2-b565ce8fc3ba
# ╟─0ddafb7a-b7c6-11ef-3c3f-c9fa7af39c92
# ╟─0ddb085c-b7c6-11ef-34fd-6b1b18a95ff1
# ╟─0ddb163a-b7c6-11ef-2b06-a1d6677b7191
# ╟─0ddb2302-b7c6-11ef-1f50-27711dbe4d33
# ╟─0ddb2fa0-b7c6-11ef-3ac5-8979f2a0a00c
# ╟─0ddb3c34-b7c6-11ef-2a77-895cbc5796f3
# ╟─0ddb4b54-b7c6-11ef-121d-5d00e547debd
# ╠═0ddb4bb4-b7c6-11ef-373a-ab345190363a
# ╟─0ddb7294-b7c6-11ef-0585-3f1a218aeb42
# ╟─0ddb9904-b7c6-11ef-3808-35b8ee37dd04
# ╟─0ddba9ee-b7c6-11ef-3148-9db5fbb13d77
# ╟─0ddbba2e-b7c6-11ef-04cf-1119024af1d1
# ╟─0ddbc78a-b7c6-11ef-2ce4-f76fa4153e4b
# ╠═0ddbc7c8-b7c6-11ef-004f-8bfaa5f29eba
# ╟─0ddbd3ce-b7c6-11ef-20e1-070d736f7b95
# ╟─0ddbf246-b7c6-11ef-16a5-bbf396f80915
# ╠═0ddbf278-b7c6-11ef-20f5-7ffd3163b14f
# ╟─0ddc02d6-b7c6-11ef-284e-018c7895536e
# ╟─0ddc1028-b7c6-11ef-1eec-6d72e52f4431
# ╟─0ddc1c2e-b7c6-11ef-00b6-e98913a96420
# ╟─0ddc287e-b7c6-11ef-1f72-910e6e7b06bb
# ╟─0ddc367a-b7c6-11ef-38f9-09fb462987dc
# ╟─0ddc4796-b7c6-11ef-2156-8b3d6899a8c0
# ╟─0ddc55f6-b7c6-11ef-3975-9d92d7e3feca
# ╟─0ddc7284-b7c6-11ef-0c0c-bd949a5ef015
# ╟─0ddc7f40-b7c6-11ef-0253-6342085f708a
# ╟─0ddc8b70-b7c6-11ef-13cb-3daa72032cf9
# ╟─0ddc9aac-b7c6-11ef-3d8e-f5a5d0e715f8
# ╠═0ddc9ae8-b7c6-11ef-33a5-771f934e6ae8
# ╠═5062b49b-34d0-4364-a9b3-7d29119ed599
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
