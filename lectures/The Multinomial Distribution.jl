### A Pluto.jl notebook ###
# v0.20.13

#> [frontmatter]
#> description = "Bayesian and maximum likelihood density estimation for discretely valued data sets."
#> 
#>     [[frontmatter.author]]
#>     name = "BMLIP"
#>     url = "https://github.com/bmlip"

using Markdown
using InteractiveUtils

# ╔═╡ d3a4a1dc-3fdf-479d-a51c-a1e23073c556
using PlutoUI, PlutoTeachingTools

# ╔═╡ d8422bf2-d294-11ef-0144-098f414c6454
md"""
# Discrete Data and the Multinomial Distribution

"""

# ╔═╡ 1c6d16be-e8e8-45f1-aa32-c3fb08af19ce
PlutoUI.TableOfContents()

# ╔═╡ d8424e52-d294-11ef-0083-fbb77df4d853
md"""
## Preliminaries

##### Goal 

  * Simple Bayesian and maximum likelihood-based density estimation for discretely valued data sets

##### Materials        

  * Mandatory

      * These lecture notes
  * Optional

      * [Bishop PRML book](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) (2006), pp. 67-70, 74-76, 93-94

"""

# ╔═╡ d842ad86-d294-11ef-3266-253f80ecf4b7
md"""
## Discrete Data: the 1-of-K Coding Scheme

Consider a coin-tossing experiment with outcomes ``x \in\{0,1\}`` (tail and head, respectively) and let ``0\leq \mu \leq 1`` represent the probability of heads. The data generating distribution for this model can written as a [**Bernoulli distribution**](https://en.wikipedia.org/wiki/Bernoulli_distribution):

```math
 
p(x|\mu) = \mu^{x}(1-\mu)^{1-x}
```

Note that the variable ``x`` acts as a (binary) **selector** for the tail or head probabilities. Think of this as an 'if'-statement in programming.

"""

# ╔═╡ d842d368-d294-11ef-024d-45e58ca994e0
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

# ╔═╡ d842fe4c-d294-11ef-15a9-a9a6e359f47d
md"""
## The Categorical Distribution

Consider a toss with a ``K``-sided die. We use a one-hot coding scheme, i.e., the outcome is encoded as 
```math
x_{k} = \begin{cases} 1 & \text{if the throw landed on $k$-th face}\\
0 & \text{otherwise} \end{cases} \,.
```

Assume the probabilities


```math 
p(x_{k}=1) = \mu_k \quad \text{with } \sum_k \mu_k  = 1 \,.
```
The data generating distribution for one-hot encoded outcome ``x = (x_{1},x_{2},\ldots,x_{K})`` (and ``\mu = (\mu_1,\mu_2,\dots,\mu_k)^T``) is then given by 

```math
p(x|\mu) = \mu_1^{x_1} \mu_2^{x_2} \cdots \mu_K^{x_K}=\prod_{k=1}^K \mu_k^{x_k} \tag{B-2.26}
```

This generalized Bernoulli distribution is called the [**categorical distribution**](https://en.wikipedia.org/wiki/Categorical_distribution).

"""

# ╔═╡ d843540a-d294-11ef-3846-2bf27b7e9b30
md"""
# Bayesian Density Estimation for a Loaded Die

Now let's proceed with learning the parameters for a model for ``N`` independent-and-identically-distributed (IID) rolls of a ``K``-sided die, based on observed data set ``D=\{x_1,\ldots,x_N\}``. 


"""

# ╔═╡ d84369a4-d294-11ef-38f7-7f393869b705
md"""
## Model specification

#### data-generating distribution

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

# ╔═╡ d8439866-d294-11ef-230b-dfde21aedfbf
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

# ╔═╡ d843a338-d294-11ef-2748-b95f2af1396b
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

# ╔═╡ d843b33c-d294-11ef-195d-2708fbfba49d
md"""
We recognize the ``(\alpha_k)``'s as prior pseudo-counts and the Dirichlet distribution shows to be a [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior) to the categorical/multinomial:

```math
\begin{align*}
\underbrace{\text{Dirichlet}}_{\text{posterior}} &\propto \underbrace{\text{categorical}}_{\text{likelihood}} \cdot \underbrace{\text{Dirichlet}}_{\text{prior}}
\end{align*}
```

"""

# ╔═╡ d843c228-d294-11ef-0d34-3520dc97859c
md"""
This is actually a generalization of the conjugate relation that we found for the binary coin toss: 

```math
\begin{align*}
\underbrace{\text{beta}}_{\text{posterior}} &\propto \underbrace{\text{binomial}}_{\text{likelihood}} \cdot \underbrace{\text{beta}}_{\text{prior}}
\end{align*}
```

"""

# ╔═╡ d843d0c4-d294-11ef-10b6-cb982615d58a
md"""
## $(HTML("<span id='prediction-loaded-die'>Prediction of next toss for the loaded die</span>"))

Let's apply what we have learned about the loaded die to compute the probability that we throw the ``k``-th face at the next toss. 

```math
\begin{align*}
p(x_{\bullet,k}=1|D)  &= \int p(x_{\bullet,k}=1|\mu)\,p(\mu|D) \,\mathrm{d}\mu \\
  &= \int_0^1 \mu_k \times  \mathcal{Dir}(\mu|\,\alpha+m) \,\mathrm{d}\mu  \\
  &= \mathrm{E}\left[ \mu_k \right] \\
  &= \frac{m_k + \alpha_k }{ N+ \sum_k \alpha_k}
\end{align*}
```

(You can find the mean of the Dirichlet distribution ``\mathrm{E}\left[ \mu_k \right]`` at its [Wikipedia site](https://en.wikipedia.org/wiki/Dirichlet_distribution)). 

This result is simply a generalization of [**Laplace's rule of succession**](https://en.wikipedia.org/wiki/Rule_of_succession).

"""

# ╔═╡ d843defc-d294-11ef-358b-f56f514dcf93
md"""
## Categorical, Multinomial and Related Distributions

In the above derivation, we noticed that the data generating distribution for ``N`` die tosses with data outcomes ``D=\{x_1,\ldots,x_N\}`` only depends on the **counts** ``m_k``:

```math
p(D|\mu) = \prod_n \underbrace{\prod_k \mu_k^{x_{nk}}}_{\text{categorical dist.}} = \prod_k \mu_k^{\sum_n x_{nk}} = \prod_k \mu_k^{m_k} \tag{B-2.29}
```

"""

# ╔═╡ d843efdc-d294-11ef-0f3a-630ecdd0acee
md"""
A related distribution is the distribution over count observations ``D_m=\{m_1,\ldots,m_K\}``, which is called the **multinomial distribution**,

```math
p(D_m|\mu) =\frac{N!}{m_1! m_2!\ldots m_K!} \,\prod_k \mu_k^{m_k}\,.
```

"""

# ╔═╡ d84422a6-d294-11ef-148b-c762a90cd620
md"""
We insert this slide only to alert you to the difference between using one-hot encoded outcomes ``D=\{x_1,x_2,\ldots,x_N\}`` as the data, versus using counts ``D_m = \{m_1mm_3,\ldots,m_K\}`` as the data. When used as a likelihood function for ``\mu``, it makes no difference whether you use ``p(D|\mu)`` or ``p(D_m|\mu)``. Why? 

"""

# ╔═╡ d8449f1a-d294-11ef-3cfa-4fc33a5daa00
md"""
## Maximum Likelihood Estimation for the Multinomial

#### Maximum likelihood as a special case of Bayesian estimation

We can obtain the maximum likelihood estimate for ``\mu_k`` based on ``N`` throws of a ``K``-sided die within the Bayesian framework by letting the prior for ``\mu`` approach a uniform distribution. For a Dirichlet prior ``\mathrm{Dir}(\mu | \alpha)``, this corresponds to setting
``\alpha \rightarrow (1, 1, \dots, 1)``.


Proof for yourself that 

```math
\begin{align*}
\hat{\mu}_k &= \arg\max_{\mu_k} p(D|\mu) = \frac{m_k}{N}\,.
\end{align*}
```

"""

# ╔═╡ 4482e857-af6b-4459-a0a2-cd7ad57ed94f
details("Click for proof",
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

# ╔═╡ d844bcfa-d294-11ef-0874-b154f3ed810b
md"""
#### $(HTML("<span id='ML-for-multinomial'>Maximum likelihood estimation by optimizing a constrained log-likelihood</span>"))

Of course, we shouldn't have to go through the full Bayesian framework to get the maximum likelihood estimate. Alternatively, we can find the maximum likelihood (ML) solution directly by optimizing the (constrained) log-likelihood.

The log-likelihood for the multinomial distribution is given by

```math
\begin{align*}
\mathrm{L}(\mu) &\triangleq \log p(D_m|\mu) \propto \log \prod_k \mu_k^{m_k} =  \sum_k m_k \log \mu_k 
\end{align*}
```

"""

# ╔═╡ d844d564-d294-11ef-0454-416352d43524
md"""
When doing ML estimation, we must obey the constraint ``\sum_k \mu_k  = 1``, which can be accomplished by a [Lagrange multiplier](https://en.wikipedia.org/wiki/Lagrange_multiplier). The **constrained log-likelihood** with Lagrange multiplier is then

```math
\tilde{\mathrm{L}}(\mu) = \sum_k m_k \log \mu_k  + \lambda \cdot \big(1 - \sum_k \mu_k \big)
```

The method of Lagrange multipliers is a mathematical method for transforming a constrained optimization problem to an unconstrained optimization problem (see [Bishop App.E](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf#page=727)). Unconstrained optimization problems can be solved by setting the derivative to zero. 

"""

# ╔═╡ d844fa76-d294-11ef-172a-85e68842c252
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

# ╔═╡ d8455278-d294-11ef-2455-376c205e7edf
md"""

# Summary

Assume a data set of ``N`` IID observations ``D=\{x_1,\dotsc,x_N\}``.

For discrete outcomes modeled by a 1-of-K **categorical distribution**, ``p(x_n) = \mathrm{Cat}(x_n|\mu)``, we find that the Maximum Likelihood (ML) estimate of ``\mu = (\mu_1,\mu_2,\ldots,\mu_K)^T`` is given by the sample proportions:

```math
\begin{align}
\hat\mu_k  = \frac{1}{N} \sum_n x_{nk} \quad \left(= \frac{m_k}{N} \right) \tag{sample proportion}
\end{align}
```

Let us contrast this with the earlier results for a multivariate **Gaussian model**, where the ML estimates for the mean and variance parameters coincide with the sample mean and sample variance, respectively.

```math
\begin{align}
\hat{\mu} &= \frac{1}{N}\sum_{n=1}^N x_n \tag{sample mean} \\  
\hat{\Sigma} &= \frac{1}{N} \sum_{n=1}^N (x_n-\mu)(x_n-\mu)^T \tag{sample variance}
\end{align}
```


Note the similarity for the ML estimates of the mean. 


"""

# ╔═╡ 204bec3f-6fde-48c1-b2b6-9f88d484c130
md"""
# Exercises
"""

# ╔═╡ 62b42d1d-be91-4740-bac6-b4527494959d
md"""

####  Maximum Likelihood estimation (**)

We consider IID data ``D = \{x_1,x_2,\ldots,x_N\}`` obtained from tossing a ``K``-sided die. We use a *binary selection variable*

```math
x_{nk} \equiv \begin{cases} 1 & \text{if $x_n$ lands on $k$-th face}\\
    0 & \text{otherwise}
\end{cases}
```

with probabilities ``p(x_{nk} = 1)=\mu_k``.         

- (a) Derive the log-likelihood ``\log p(D|\mu)``.        
- (b) Derive the maximum likelihood estimate for ``\mu``.

"""

# ╔═╡ 01c4c590-fece-49a5-8979-6e0d54f7850a
details("Click for solution",
md"""
Derivations are in the lecture notes.        
		
- (a)


```math
p(x_n|\mu) = \prod_k \mu_k^{x_{nk}} \quad \text{subject to} \quad \sum_k \mu_k = 1 \,.
```

```math
p(D|\mu)  = \sum_k m_k \log \mu_k
```

where ``m_k = \sum_n x_{nk}``.       

- (b)


```math
\hat \mu = \frac{m_k}{N}\,,
```

which is the *sample proportion*.
""")

# ╔═╡ d8443e38-d294-11ef-25db-b16df87850f4
md"""
#### Discrete Distributions (*)

Show that

- (a) the categorial distribution is a special case of the multinomial for ``N=1``.  

- (b) the Bernoulli is a special case of the categorial distribution for ``K=2``.    

- (c) the binomial is a special case of the multinomial for ``K=2``.

"""

# ╔═╡ 448d0679-b47a-4db9-ad7d-a45786350fef
details("Click for solution",
md"""

- (a) The probability mass function of a **multinomial distribution** is 
```math 
	p(D_m|\mu) =\frac{N!}{m_1! m_2!\ldots m_K!} \,\prod_k \mu_k^{m_k}
```
over the data frequencies ``D_m=\{m_1,\ldots,m_K\}`` with constraints that ``\sum_k \mu_k = 1`` and ``\sum_k m_k=N``. 

Setting ``N=1``, we see that ``p(D_m|\mu) \propto \prod_k \mu_k^{m_k}`` with ``\sum_k m_k=1``, making the sample-space one-hot coded. This is the **categorical distribution**.       
		
- (b) When ``K=2``, the constraint for the categorical distribution takes the form ``m_1=1-m_2`` leading to 

```math
	p(D_m|\mu) \propto \mu_1^{m_1}(1-\mu_1)^{1-m_1}
```
which is associated with the **Bernoulli distribution**.       

- (c) Plugging ``K=2`` into the multinomial distribution leads to ``p(D_m|\mu) =\frac{N!}{m_1! m_2!}\mu_1^{m_1}\left(\mu_2^{m_2}\right)`` with the constraints ``m_1+m_2=N`` and ``\mu_1+\mu_2=1``. Then plugging the constraints back in we obtain 
```math
	p(D_m|\mu) = \frac{N!}{m_1! (N-m1)!}\mu_1^{m_1}\left(1-\mu_1\right)^{N-m_1}
```
which is the **binomial distribution**.


""")

# ╔═╡ 72f24b54-ab22-4a54-9ece-7433048f4769
md"""

#### Laplace's Generalized Rule of Succession (**) 

Show that Laplace's generalized rule of succession can be worked out to a prediction that is composed of a prior prediction and data-based correction term.


"""

# ╔═╡ 3c2ee96d-18a6-45d0-a2cf-f2ebbf5e22f0
details("Click for solution",
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

# ╔═╡ 59fb1e66-cf05-4f2b-8027-7ff3b1a57c15
md"""
# Appendix
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoTeachingTools = "~0.3.1"
PlutoUI = "~0.7.62"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "65874a65b39702642b8f819a8904f6c2925e64f2"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "062c5e1a5bf6ada13db96a4ae4749a4c2234f521"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.9"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

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

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "6ac9e4acc417a5b534ace12690bc6973c25b862f"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.3"

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

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoweredCodeUtils]]
deps = ["Compiler", "JuliaInterpreter"]
git-tree-sha1 = "b882a7dd7ef37643066ae8f9380beea8fdd89cae"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.4.2"

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

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

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
git-tree-sha1 = "ec9e63bd098c50e4ad28e7cb95ca7a4860603298"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.68"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

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

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

    [deps.Revise.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

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

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─d8422bf2-d294-11ef-0144-098f414c6454
# ╟─1c6d16be-e8e8-45f1-aa32-c3fb08af19ce
# ╟─d8424e52-d294-11ef-0083-fbb77df4d853
# ╟─d842ad86-d294-11ef-3266-253f80ecf4b7
# ╟─d842d368-d294-11ef-024d-45e58ca994e0
# ╟─d842fe4c-d294-11ef-15a9-a9a6e359f47d
# ╟─d843540a-d294-11ef-3846-2bf27b7e9b30
# ╟─d84369a4-d294-11ef-38f7-7f393869b705
# ╟─d8439866-d294-11ef-230b-dfde21aedfbf
# ╟─d843a338-d294-11ef-2748-b95f2af1396b
# ╟─d843b33c-d294-11ef-195d-2708fbfba49d
# ╟─d843c228-d294-11ef-0d34-3520dc97859c
# ╟─d843d0c4-d294-11ef-10b6-cb982615d58a
# ╟─d843defc-d294-11ef-358b-f56f514dcf93
# ╟─d843efdc-d294-11ef-0f3a-630ecdd0acee
# ╟─d84422a6-d294-11ef-148b-c762a90cd620
# ╟─d8449f1a-d294-11ef-3cfa-4fc33a5daa00
# ╟─4482e857-af6b-4459-a0a2-cd7ad57ed94f
# ╟─d844bcfa-d294-11ef-0874-b154f3ed810b
# ╟─d844d564-d294-11ef-0454-416352d43524
# ╟─d844fa76-d294-11ef-172a-85e68842c252
# ╟─d8455278-d294-11ef-2455-376c205e7edf
# ╟─204bec3f-6fde-48c1-b2b6-9f88d484c130
# ╟─62b42d1d-be91-4740-bac6-b4527494959d
# ╟─01c4c590-fece-49a5-8979-6e0d54f7850a
# ╟─d8443e38-d294-11ef-25db-b16df87850f4
# ╟─448d0679-b47a-4db9-ad7d-a45786350fef
# ╟─72f24b54-ab22-4a54-9ece-7433048f4769
# ╟─3c2ee96d-18a6-45d0-a2cf-f2ebbf5e22f0
# ╟─59fb1e66-cf05-4f2b-8027-7ff3b1a57c15
# ╠═d3a4a1dc-3fdf-479d-a51c-a1e23073c556
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
