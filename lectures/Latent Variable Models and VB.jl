### A Pluto.jl notebook ###
# v0.20.13

#> [frontmatter]
#> image = "https://github.com/bmlip/course/blob/v2/assets/figures/fig-Bishop-A5-Old-Faithfull.png?raw=true"
#> description = "Introduction to latent variable models and variational inference via free energy minimization."
#> 
#>     [[frontmatter.author]]
#>     name = "BMLIP"
#>     url = "https://github.com/bmlip"

using Markdown
using InteractiveUtils

# ╔═╡ c90176ea-918b-4643-a10f-cef277c5ea75
using LinearAlgebra, PDMats, SpecialFunctions, Random

# ╔═╡ df171940-eb54-48e2-a2b8-1a8162cabf3e
using PlutoUI, PlutoTeachingTools

# ╔═╡ 58bd0d43-743c-4745-b353-4a89b35e85ba
using Distributions, Plots, StatsPlots

# ╔═╡ 9d2068d7-db54-460e-930c-b7c3273162ee
using HypertextLiteral

# ╔═╡ 26c56fd8-d294-11ef-236d-81deef63f37c
md"""
# Latent Variable Models and Variational Bayes

"""

# ╔═╡ ce7d086b-ff20-4da1-a4e8-52b5b7dc9e2b
PlutoUI.TableOfContents()

# ╔═╡ 26c58298-d294-11ef-2a53-2b42b48e0725
md"""
## Preliminaries

##### Goal 

  * Introduction to latent variable models and variational inference by Free energy minimization

##### Materials

  * Mandatory

      * These lecture notes
  * Optional 

      * Bishop (2016), [PRML book](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf), pp. 461-486 (sections 10.1, 10.2 and 10.3)
      * Ariel Caticha (2010), [Entropic Inference](https://arxiv.org/abs/1011.0723)

          * tutorial on entropic inference, which is a generalization to Bayes rule and provides a foundation for variational inference.
  * references $(HTML("<span id='references'></span>"))

      * Blei et al. (2017), [Variational Inference: A Review for Statisticians](https://doi.org/10.1080/01621459.2017.1285773)
      * Lanczos (1961), [The variational principles of mechanics](https://www.amazon.com/Variational-Principles-Mechanics-Dover-Physics/dp/0486650677)
      * Senoz et al. (2021), [Variational Message Passing and Local Constraint Manipulation in Factor Graphs](https://research.tue.nl/nl/publications/variational-message-passing-and-local-constraint-manipulation-in-)
      * Dauwels (2007), [On variational message passing on factor graphs](https://github.com/bmlip/course/blob/main/assets/files/Dauwels-2007-on-variational-message-passing-on-factor-graphs.pdf)
      * Shore and Johnson (1980), [Axiomatic Derivation of the Principle of Maximum Entropy and the Principle of Minimum Cross-Entropy](https://github.com/bmlip/course/blob/main/assets/files/ShoreJohnson-1980-Axiomatic-Derivation-of-the-Principle-of-Maximum-Entropy.pdf)

"""

# ╔═╡ e0d0f3a1-5e00-44f0-9c2b-4308cbd673ce
TODO("the figure above is very large; can we scale the size down a bit?")

# ╔═╡ f8c8013a-3e87-4d01-a3ae-86b39cf1f002
md"""
# The Gaussian Mixture Model
"""

# ╔═╡ 26c59b52-d294-11ef-1eba-d3f235f85eee
md"""
## Unobserved Classes

Consider again a set of observed data ``D=\{x_1,\dotsc,x_N\}``.

This time, we suspect that there are *unobserved* class labels that would help explain (or predict) the data, e.g.,

  * the observed data are the color of living things; the unobserved classes are animals and plants.
  * observed are wheel sizes; unobserved categories are trucks and personal cars.
  * observed is an audio signal; unobserved classes include speech, music, traffic noise, etc.

"""

# ╔═╡ 26c5a1f6-d294-11ef-3565-39d027843fbb
md"""
Classification problems with unobserved classes are called **Clustering** problems. In clustering problems, the learning algorithm needs to *discover the underlying classes from the observed data*.

"""

# ╔═╡ 26c5a93a-d294-11ef-23a1-cbcf0c370fc9
md"""
## The Gaussian Mixture Model

The spread of the data in the Old Faithful data set looks like it could be modeled by two Gaussians. Let's develop a model for this data set. 

"""

# ╔═╡ 26c5b896-d294-11ef-1d8e-0feb99d2d45b
md"""

We associate a one-hot coded hidden class label ``z_n`` with each observation ``x_n``:

```math
\begin{equation*}
z_{nk} = \begin{cases} 1 & \text{if } x_n \in \mathcal{C}_k \text{ (the $k$-th class)}\\
                       0 & \text{otherwise} \end{cases}
\end{equation*}
```

"""

# ╔═╡ 26c5c1ae-d294-11ef-15c6-13cae5bc0dc8
md"""
We consider the same model as we did in the [generative classification lesson](https://bmlip.github.io/course/lectures/Generative%20Classification.html#GDA): the data for each class is distributed as a Gaussian:

```math
\begin{align*}
p(x_n | z_{nk}=1) &= \mathcal{N}\left( x_n | \mu_k, \Sigma_k\right)\\
p(z_{nk}=1) &= \pi_k
\end{align*}
```

which can be summarized with the selection variables ``z_{nk}`` as

```math
\begin{align*}
p(x_n,z_n) &=  \prod_{k=1}^K (\underbrace{\pi_k \cdot \mathcal{N}\left( x_n | \mu_k, \Sigma_k\right) }_{p(x_n,z_{nk}=1)})^{z_{nk}} 
\end{align*}
```

*Again*, this is the same model  as we defined for the generative classification model: A Gaussian-Categorical model but now with unobserved classes. 

This model (with **unobserved class labels**) is known as a **Gaussian Mixture Model** (GMM).

"""

# ╔═╡ 26c5cfb4-d294-11ef-05bb-59d5e27cf37c
md"""
## The Marginal Distribution for the GMM

In the literature, the GMM is often introduced by the marginal distribution for an *observed* data point ``x_n``, given by

```math
\begin{align*}{}
p(x_n) &= \sum_{z_n} p(x_n,z_n)  \\
  &= \sum_{k=1}^K \pi_k \cdot \mathcal{N}\left( x_n | \mu_k, \Sigma_k \right) \tag{B-9.12}
\end{align*}
```


"""

# ╔═╡ c7351bf1-447e-475b-8965-d259c01bfd57
details("Click for proof",
md"""

```math
\begin{align}
p(x_n) &= \sum_{z_n} p(x_n,z_n) \\
  &= \sum_{z_n} \prod_{k=1}^K \left(\pi_k \cdot \mathcal{N}\left( x_n | \mu_k, \Sigma_k\right) \right)^{z_{nk}} \\ 
&= \sum_{j=1}^K \prod_{k=1}^K \left( \pi_k \cdot \mathcal{N}\left( x_n | \mu_k, \Sigma_k\right) \right)^{I_{kj}}  \;\; \text{(use }z_n \text{ is one-hot coded)}\\  
&= \sum_{j=1}^K  \pi_j \cdot \mathcal{N}\left( x_n | \mu_j, \Sigma_j\right) 
  \end{align}
```

where ``I_{kj} = 1`` if ``k=j`` and ``0`` otherwise.
	
""")

# ╔═╡ 3deadfd0-9fbb-476a-a7de-5dd694e55a65
md"""


Eq. B-9.12 reveals the link to the name Gaussian *mixture model*. The priors ``\pi_k`` for the ``k``-th class are also called **mixture coefficients**. 

Be aware that Eq. B-9.12 is not the generative model for the GMM! The generative model is the joint distribution ``p(x,z)`` over all variables, including the latent variables. 
"""

# ╔═╡ 26c5d734-d294-11ef-20a3-afd2c3324323
md"""
## GMM is a Flexible Model

GMMs are very popular models. They have decent computational properties and are **universal approximators of densities** (as long as there are enough Gaussians, of course).

![](https://github.com/bmlip/course/blob/v2/assets/figures/fig-ZoubinG-GMM-universal-approximation.png?raw=true)

(In the above figure, the Gaussian components are shown in $(html"<span style='color: red'>red</span>") and the pdf of the mixture models in $(html"<span style='color: blue'>blue</span>")).

"""

# ╔═╡ 26c5f8d6-d294-11ef-3bcd-4d5e0391698d
md"""
## Latent Variable Models

A GMM contains both **observed** variables ``\{x_n\}``, and **unobserved** variables, namely unobserved (synonym: latent, hidden) parameters ``\theta= \{\pi_k,\mu_k, \Sigma_k\}`` and unobserved  class labels ``\{z_{nk}\}``.

From a Bayesian viewpoint, both the class labels ``\{z_{nk}\}`` and the parameters ``\theta`` are just unobserved variables for which we can set a prior and compute a posterior by Bayes rule. 

Note that ``z_{nk}`` carries a subscript ``n``, indicating that its value depends not only on the class index ``k``, but also on the specific observation ``n``. This contrasts with global model parameters ``\theta``, which are shared across all data points. 

Observation-specific latent variables can be a powerful modeling tool for capturing additional structure in the data, particularly information about the hidden causes of individual observations. In the case of the Gaussian Mixture Model (GMM), the latent variables ``z_{nk}`` represent unobserved class memberships, specifying which component generated each data point.

Models that incorporate unobserved variables, often specific to each observation, are broadly known as **Latent Variable Models** (LVMs). These latent variables help explain the hidden structure or generative process underlying the observed data.

By adding model structure through (equations among) observation-dependent latent variables, we can often build more accurate models for very complex processes. Unfortunately, adding structure through observation-dependent latent variables in models is also often accompanied by a more complex inference task.

"""

# ╔═╡ 26c623f6-d294-11ef-13c0-19edd43592c0
md"""
## Inference for GMM is Difficult

Indeed, the fact that the observation-dependent class labels are *unobserved* for the GMM, leads to a problem for processing new data by Bayes rule in a GMM.

Consider a given data set ``D = \{x_1,x_2,\ldots,x_N\}``. We recall here the log-likelihood for the Gaussian-Categorial Model, see the [generative classification lesson](https://bmlip.github.io/course/lectures/Generative%20Classification.html):

```math
\log\, p(D|\theta) =  \sum_{n,k} y_{nk} \underbrace{ \log\mathcal{N}(x_n|\mu_k,\Sigma) }_{ \text{Gaussian} } + \underbrace{ \sum_{n,k} y_{nk} \log \pi_k }_{ \text{multinomial} } \,.
```

"""

# ╔═╡ 26c62ebe-d294-11ef-0cfb-ef186203e890
md"""
Since the class labels ``y_{nk} \in \{0,1\}`` were assumed to be given by the data set, maximization of this expression decomposed into a set of simple update rules for the Gaussian and multinomial distributions. 

"""

# ╔═╡ 26c6347c-d294-11ef-056f-7b78a9e22272
md"""
However, for the Gaussian mixture model (same log-likelihood function with ``z_{nk}`` replacing ``y_{nk}``), the class labels ``\{z_{nk}\}`` are *unobserved* and they need to be estimated alongside with the parameters.

"""

# ╔═╡ 26c64174-d294-11ef-2bbc-ab1a84532311
md"""
There is no known conjugate prior for the latent variables in the GMM likelihood. Therefore, Bayes rule does not yield a closed-form expression for the posterior over the latent variables:

```math
 \underbrace{p(\{z_{nk}\},\{\mu_k,\Sigma_k,\pi_k\} | D)}_{\text{posterior (no analytical solution)}} \propto \underbrace{p(D\,|\,\{z_{nk}\},\{\mu_k,\Sigma_k,\pi_k\})}_{\text{likelihood}} \cdot \underbrace{p( \{z_{nk}\},\{\mu_k,\Sigma_k,\pi_k\} )}_{\text{prior (no known conjugate)}} 
```

"""

# ╔═╡ 26c65092-d294-11ef-39cc-1953a725f285
md"""
Can we still compute an approximate posterior? In this lesson, we introduce an approximate Bayesian inference method known as **Variational Bayes** (VB) (also known as **Variational Inference**) that can be used for Bayesian inference in models with latent variables. Later in this lesson, we will use VB to do inference in the GMM.   

"""

# ╔═╡ f1f7407d-86a1-4f24-b78a-61a411d1f371
md"""
# Variational Inference
"""

# ╔═╡ 26c67f04-d294-11ef-03a4-838ae255689d
md"""
## The Variational Free Energy Functional

We'll start from scratch. Consider a model ``p(x,z) = p(x|z) p(z)``, where ``x`` and ``z`` are observed and latent variables, respectively. ``z`` may include parameters but also observation-dependent latent variables. 

The goal of Bayesian inference is to transform the (known) *likelihood-times-prior* factorization of the full model to a *posterior-times-evidence* decomposition: 

```math
 \underbrace{p(x|z) p(z)}_{\text{what we know}} \rightarrow \underbrace{p(z|x) p(x)}_{\text{what we want}} 
```

Remember from the [Bayesian machine learning lesson](https://bmlip.github.io/course/lectures/Bayesian%20Machine%20Learning.html#Bayesian-model-evidence) that negative log-evidence can be decomposed as "complexity" minus "accuracy" terms (the CA decomposition):

```math
 -\log p(x) =  \underbrace{ \int p(z|x) \log \frac{p(z|x)}{p(z)} \mathrm{d}z }_{\text{complexity}} - \underbrace{\int p(z|x) \log p(x|z) \mathrm{d}z}_{\text{accuracy}}
 
```

The CA decomposition cannot be evaluated because it depends on the posterior ``p(z|x)``, which cannot be evaluated since it is the objective of the inference process. 

Let's now introduce a distribution ``q(z)`` (called the "variational" or "approximate" posterior distribution) that we will use to *approximate* the posterior ``p(z|x)``. Since we propose the distribution ``q(z)`` ourselves, we will assume that ``q(z)`` can be evaluated. 

If will substitute ``q(z)`` for ``p(z|x)`` in the CA decomposition, then we obtain 

```math
 F[q] \triangleq  \underbrace{ \int q(z) \log \frac{q(z)}{p(z)} \mathrm{d}z }_{\text{complexity}} - \underbrace{\int q(z) \log p(x|z) \mathrm{d}z}_{\text{accuracy}}
 
```

This expression is called the **Variational Free Energy** (VFE), represented by the symbol ``F``. We treat ``F`` as a function of the posterior ``q(z)``. Technically, a function of a function is called a functional, and we write square brackets (e.g., ``F[q]``) to differentiate functionals from functions (e.g., ``q(z)``). 

Note that all factors in the CA decomposition of VFE (i.e., ``q(z)``, ``p(z)``, and ``p(x|z)``) can be evaluated as a function of ``z`` (and ``x`` is observed), and therefore the VFE can be evaluated. This is important: log-evidence ``\log p(x)`` cannot be evaluated, but ``F[q]`` *can* be evaluated! 

"""

# ╔═╡ 26c6e002-d294-11ef-15a4-33e30d0d76ec
md"""
## The Global VFE Minimum Recovers Bayes Rule

It turns out that we can perform (approximate) Bayesian inference by minimizing the Variational Free Energy (VFE) with respect to the variational distribution ``q``.

To explain inference by VFE minimization (abbreviated as VFEM), we first rewrite the VFE in terms of "inference bound" minus "log-evidence" terms (the bound-evidence (BE) decomposition):

```math
\begin{align*}
 F[q] &= \underbrace{ \int q(z) \log \frac{q(z)}{p(z)} \mathrm{d}z }_{\text{complexity}} - \underbrace{\int q(z) \log p(x|z) \mathrm{d}z}_{\text{accuracy}} \\
 &= \underbrace{\int q(z) \log \frac{q(z)}{p(z|x)}\mathrm{d}z}_{\text{inference bound}\geq 0} - \underbrace{\log p(x)}_{\text{log-evidence}} 
 \end{align*}
```



"""

# ╔═╡ ae7ed1fc-fc36-4327-be55-a142477ca0ad
details("Click for proof", 
md"""
```math
\begin{align*}
 F[q] &= \underbrace{ \int q(z) \log \frac{q(z)}{p(z)} \mathrm{d}z }_{\text{complexity}} - \underbrace{\int q(z) \log p(x|z) \mathrm{d}z}_{\text{accuracy}} \\
 &= \int q(z) \log \frac{q(z)}{ p(x|z) p(z) }\mathrm{d}z \\
 &= \int q(z) \log \frac{q(z)}{ p(z|x) p(x)}\mathrm{d}z \quad \text{( since }  p(x|z) p(z) =  p(z|x) p(x)\text{ )} \\
 &= \underbrace{\int q(z) \log \frac{q(z)}{p(z|x)}\mathrm{d}z}_{\text{inference bound}\geq 0} - \underbrace{\log p(x)}_{\text{log-evidence}} 
 \end{align*}
```
""")

# ╔═╡ de16b831-7afa-408f-83fa-99c6e24840f5
md"""
Note that the inference bound is a [Kullback-Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between an (approximate) posterior ``q(z)`` and the (perfect) Bayesian posterior ``p(z|x)``. See this [slide in the BML Class](https://bmlip.github.io/course/lectures/Bayesian%20Machine%20Learning.html#KLD) for more info on the KL divergence. 

Since the second term (log-evidence) does not involve ``q(z)``, VFEM over ``q`` will bring ``q(z)`` closer to the Bayesian posterior ``p(z|x)``.

Since ``D_{\text{KL}}[q(z),p(z|x)]\geq 0`` for any ``q(z)``, and ``D_{\text{KL}}[q(z),p(z|x)]= 0``  only if ``q(z) = p(z|x)``, the VFE is always an upper-bound on (minus) log-evidence, i.e.,

```math
F[q] \geq -\log p(x) \,.
```

As a result, global minimization of VFE leads to

```math
q^*(z) = \arg\min_q F[q]
```

where

```math
\begin{align}
   q^*(z) &= p(z|x) \tag{posterior}\\
   F[q^*] &= -\log p(x) \tag{evidence}
\end{align}
```
"""

# ╔═╡ e6aeee80-9e63-4937-9edf-428d5e3e38d3
keyconcept("", md"Global VFE minimization recovers Bayes rule!")

# ╔═╡ baec0494-9557-49d1-b4d8-a8030d3281b7
md"""
## Approximate Bayesian Inference by VFE Minimization

In practice, even if we cannot attain the global minimum of VFE, we can still use a local minimum, 

```math
\hat{q}(z) \approx \arg\min_q F[q]
```

to accomplish **approximate Bayesian inference** by: 

```math
\begin{align*}
\hat{q}(z) &\approx p(z|x) \\
F[\hat{q}] &\approx -\log p(x)
    \end{align*}
```

Executing inference by minimizing the VFE functional is called **Variational Bayes** (VB) or **Variational Inference** (VI). 


(As an aside), note that Bishop introduces in Eq. B-10.3 an *Evidence Lower BOund* (in modern machine learning literature abbreviated as **ELBO**) ``\mathcal{L}[q]`` that equals the *negative* VFE (``\mathcal{L}[q]=-F[q]``). In this class, we prefer to discuss inference in terms of minimizing VFE rather than maximizing ELBO, but note that these two concepts are equivalent. (The reason why we prefer the Free Energy formulation relates to the terminology in the Free Energy Principle, which we introduce in the [Intelligent Agents and active Inference lesson (B12)](https://bmlip.github.io/course/lectures/Intelligent%20Agents%20and%20Active%20Inference.html)). 
"""

# ╔═╡ 40ce0abb-a086-4977-9131-10f60ab44152
keyconcept("", md"VFE minimization transforms a Bayesian inference problem (that involves integration) into an optimization problem! Generally, optimization problems are easier to solve than integration problems.")  

# ╔═╡ 26c6f63c-d294-11ef-1090-e9238dd6ad3f
md"""
## Constrained VFE Minimization

It is common to add simplifying constraints to an optimization problem to make a difficult optimization task tractible. This is also standard practice when approximating Bayesian inference by FE minimization.

There are three important cases of adding constraints to the VFE functional that often alleviate the VFE minimization task:
  - form constraints on ``q(z)``
  - factorization constraints on ``q(z)``
  - other (ad hoc) constraints

We will shortly discuss these simplifications below.

"""

# ╔═╡ aea77d69-9ecd-4be0-b6fd-c944d27d68df
md"""
##### 1. Form constraints

For almost every practical setting, we constrain the posterior ``q(z)`` to belong to a **specific parameterized family** of probability distributions, e.g.,

```math
q(z) = \mathcal{N}\left( z | \mu, \Sigma \right)\,.
```

In this case, the *functional* minimization problem for ``F[q]`` simplifies to the minimization of an ordinary *function*

```math
F(\mu,\Sigma) = \int \mathcal{N}\left( z | \mu, \Sigma \right) \log \frac{\mathcal{N}\left( z | \mu, \Sigma \right)}{p(x,z)}\mathrm{d}z
```

with respect to the parameters ``\mu`` and ``\Sigma``. 


We can often use standard gradient-based optimization methods to minimize the function ``F(\mu,\Sigma)\,.``


"""

# ╔═╡ 3654551d-5d08-4bb0-8a0d-c7d42225bc69
md"""
##### 2. Factorization constraints

In addition to form constraints, it is also common to constrain the posterior ``q(z)`` by a specific factorization. For instance, in the *mean-field factorization* constraints, we constrain the variational posterior to factorize fully into a set of independent factors, i.e.,

```math
q(z) = \prod_{j=1}^m q_j(z_j)\,, \tag{B-10.5}
```

Variational inference with mean-field factorization has been worked out in detail as the **Coordinate Ascent Variational Inference** (CAVI) algorithm. See the [Optional Slide on CAVI](#CAVI) for details. 

Mean-field factorization is just an example of various _factorization constraints_ that have been successfully applied to VFEM.



"""

# ╔═╡ edb179df-5cff-4e7b-8645-6da4818dceee
md"""

##### 3. Other constraints, e.g., the Expectation-Minimization (EM) algorithm

Aside from form and factorization constraints on ``q(z)``, several ad hoc algorithms have been developed that ease the process of VFE minimization for particular models. 

In particular, the [Expectation-Maximization (EM) algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) is a famous special case of constrained VFE minimization. The EM algorithm places some constraints on both the posterior ``q(z)`` and the prior ``p(z)`` (see the [OPTIONAL SLIDE](#EM-Algorithm) for more info) that essentially reduce VFE minimization to maximum likelihood estimation.
"""

# ╔═╡ 757465a4-6a7f-4c8e-98de-6df5ca995b03
TODO("internal links do not seem to work")

# ╔═╡ 26c704f6-d294-11ef-1b3d-d52f0fb1c81d
md"""
## Visualization of Constrained VFEM

The following image by [David Blei](https://www.cs.columbia.edu/~blei/) illustrates the Variational Bayes approach:

![](https://github.com/bmlip/course/blob/v2/assets/figures/blei-variational-inference.png?raw=true)


"""

# ╔═╡ 26c728f0-d294-11ef-0c01-6143abe8c3f0
md"""
The Bayesian posterior ``p(z|x)`` (upper-right) is the posterior that would be obtained through executing Bayes rule, but unfortunately, Bayes rule is not tractable here. Instead, we propose a variational posterior ``q(z;\nu)`` that is parameterized by ``\nu``. The inside area of the ellipsis represents the area that is reachable by choosing values for the parameter ``\nu``. Note that ``p(z|x)`` is not reachable. We start the FE minimization process by choosing an initial value ``\nu^{\text{init}}``, which corresponds to posterior ``q(z;\nu^{\text{init}})``, as indicated in the figure. VFE minimization leads to a final value ``\nu^{*}`` that minimizes the KL-divergence between ``q(z;\nu)`` and ``p(z|x)``. 

"""

# ╔═╡ 06512595-bdb7-4adf-88ae-62af20210891
md"""
# Challenge Revisited: Modeling of the Old Faithful Data Set
"""

# ╔═╡ 26c73cf0-d294-11ef-297b-354eb9c71f57
md"""

## Derivation of VFEM Update Equations

Let's get back to the illustrative challenge at the beginning of this lesson: we want to do [density modeling for the Old Faithful data set](#illustrative-example).

"""

# ╔═╡ 3e897a59-e7b5-492c-8a8a-724248513a72
md"""
##### model specification

We consider a Gaussian Mixture Model, specified by 

```math
\begin{align*}
p(x,z|\theta) &= p(x|z,\mu,\Lambda)p(z|\pi) \\
&=  \prod_{n=1}^N \prod_{k=1}^K \mathcal{N}\left( x_n | \mu_k, \Lambda_k^{-1}\right)^{z_{nk}} \cdot \prod_{n=1}^N \prod_{k=1}^K \pi_k^{z_{nk}}  \\
  &= \prod_{n=1}^N \prod_{k=1}^K \left(\pi_k \cdot \mathcal{N}\left( x_n | \mu_k, \Lambda_k^{-1}\right)\right)^{z_{nk}} \tag{B-10.37,38}
\end{align*}
```

Let us introduce some priors for the parameters ``\pi``, ``\mu``, and ``\Lambda``. We factorize the prior and choose conjugate distributions by

```math
p(\pi,\mu,\Lambda) = p(\pi) p(\mu|\Lambda) p(\Lambda)
```

with 

```math
\begin{align}
p(\pi) &= \mathrm{Dir}(\pi|\alpha_0) = C(\alpha_0) \prod_k \pi_k^{\alpha_0-1} \tag{B-10.39} \\
p(\mu|\Lambda) &= \prod_k \mathcal{N}\left(\mu_k | m_0, \left( \beta_0 \Lambda_k\right)^{-1} \right) \tag{B-10.40} \\
p(\Lambda) &= \prod_k \mathcal{W}\left( \Lambda_k | W_0, \nu_0 \right) \tag{B-10.40}
\end{align}
```

where ``\mathcal{W}\left( \cdot \right)`` is a [Wishart distribution](https://en.wikipedia.org/wiki/Wishart_distribution) (i.e., a multi-dimensional Gamma distribution).

The full generative model is now specified by

```math
p(x,z,\pi,\mu,\Lambda) = \underbrace{p(x|z,\mu,\Lambda) p(z|\pi)}_{\text{B-10.37-38}} \underbrace{p(\pi) p(\mu|\Lambda) p(\Lambda)}_{\text{B-10.39-40}} \tag{B-10.41}
```

with hyperparameters ``\{ \alpha_0, m_0, \beta_0, W_0, \nu_0\}``.

"""

# ╔═╡ 93e7c7d5-a940-4764-8784-07af2f056e49
md"""
##### inference by constrained VFEM

Assume that we have observed ``D = \left\{x_1, x_2, \ldots, x_N\right\}`` and are interested to infer a posterior distribution for the parameters ``\pi``, ``\mu`` and ``\Lambda``.  

We will approximate Bayesian inference by VFE minimization. For the specified model, this leads to VFE minimization with respect to the hyperparameters, i.e., we need to minimize the function 

```math
F(\alpha_0, m_0, \beta_0, W_0, \nu_0) \,.
```

In general, this function can be optimized in various ways, e.g., by a gradient-descent procedure. 

It turns out that adding the following **factorization constraints** on the variational posterior makes the VFEM task analytically tractible:

```math
\begin{equation}
q(z,\pi,\mu,\Lambda) = q(z) \cdot q(\pi,\mu,\Lambda) \,. \tag{B-10.42}
\end{equation}
```

"""

# ╔═╡ 26c74c9a-d294-11ef-2d31-67bd57d56d7c
md"""

##### update equations

For this specific case (GMM model with factorization constraints), Bishop shows that the equations for the [optimal solutions (Eq. B-10.9)](#optimal-solutions) are analytically solvable, leading to the following variational update equations (for ``k=1,\ldots, K`` ): 

```math
\begin{align*}
\alpha_k &= \alpha_0 + N_k  \tag{B-10.58} \\
\beta_k &= \beta_0 + N_k  \tag{B-10.60} \\
m_k &= \frac{1}{\beta_k} \left( \beta_0 m_0 + N_k \bar{x}_k \right) \tag{B-10.61} \\
W_k^{-1} &= W_0^{-1} + N_k S_k + \frac{\beta_0 N_k}{\beta_0 + N_k}\left( \bar{x}_k - m_0\right) \left( \bar{x}_k - m_0\right)^T \tag{B-10.62} \\
\nu_k &= \nu_0 + N_k \tag{B-10.63}
\end{align*}
```

where we used

```math
\begin{align*}
\log \rho_{nk} &= \mathbb{E}\left[ \log \pi_k\right] + \frac{1}{2}\mathbb{E}\left[ \log | \Lambda_k | \right] - \frac{D}{2} \log(2\pi) \\ 
 & \qquad - \frac{1}{2}\mathbb{E}\left[(x_k - \mu_k)^T \Lambda_k(x_k - \mu_k)  \right]  \tag{B-10.46} \\
r_{nk} &= \frac{\rho_{nk}}{\sum_{j=1}^K \rho_{nj}} \tag{B-10.49} \\
N_k &= \sum_{n=1}^N r_{nk} x_n \tag{B-10.51} \\
\bar{x}_k &= \frac{1}{N_k} \sum_{n=1}^N r_{nk} x_n \tag{B-10.52} \\
S_k &= \frac{1}{N_k} \sum_{n=1}^N r_{nk} \left( x_n - \bar{x}_k\right) \left( x_n - \bar{x}_k\right)^T \tag{B-10.53}
\end{align*}
```

"""

# ╔═╡ 26c75b5e-d294-11ef-173e-b3f46a1df536
md"""
Exam guide: Working out VFE minimization for the GMM to these update equations (eqs B-10.58 through B-10.63) is not something that you need to reproduce without assistance at the exam. Rather, the essence is that *it is possible* to arrive at closed-form variational update equations for the GMM. You should understand though how FEM works conceptually and in principle be able to derive variational update equations for very simple models that do not involve clever mathematical tricks.

"""

# ╔═╡ cc547bfa-a130-4382-af47-73de56e4741b
old_faithful = 
	# CSV.read(download("https://github.com/bmlip/course/blob/v2/assets/datasets/old_faithful.csv?raw=true"), DataFrame);

	# inlining the dataset is the most reliable :)s
[
	3.600000 79.000000
	1.800000 54.000000
	3.333000 74.000000
	2.283000 62.000000
	4.533000 85.000000
	2.883000 55.000000
	4.700000 88.000000
	3.600000 85.000000
	1.950000 51.000000
	4.350000 85.000000
	1.833000 54.000000
	3.917000 84.000000
	4.200000 78.000000
	1.750000 47.000000
	4.700000 83.000000
	2.167000 52.000000
	1.750000 62.000000
	4.800000 84.000000
	1.600000 52.000000
	4.250000 79.000000
	1.800000 51.000000
	1.750000 47.000000
	3.450000 78.000000
	3.067000 69.000000
	4.533000 74.000000
	3.600000 83.000000
	1.967000 55.000000
	4.083000 76.000000
	3.850000 78.000000
	4.433000 79.000000
	4.300000 73.000000
	4.467000 77.000000
	3.367000 66.000000
	4.033000 80.000000
	3.833000 74.000000
	2.017000 52.000000
	1.867000 48.000000
	4.833000 80.000000
	1.833000 59.000000
	4.783000 90.000000
	4.350000 80.000000
	1.883000 58.000000
	4.567000 84.000000
	1.750000 58.000000
	4.533000 73.000000
	3.317000 83.000000
	3.833000 64.000000
	2.100000 53.000000
	4.633000 82.000000
	2.000000 59.000000
	4.800000 75.000000
	4.716000 90.000000
	1.833000 54.000000
	4.833000 80.000000
	1.733000 54.000000
	4.883000 83.000000
	3.717000 71.000000
	1.667000 64.000000
	4.567000 77.000000
	4.317000 81.000000
	2.233000 59.000000
	4.500000 84.000000
	1.750000 48.000000
	4.800000 82.000000
	1.817000 60.000000
	4.400000 92.000000
	4.167000 78.000000
	4.700000 78.000000
	2.067000 65.000000
	4.700000 73.000000
	4.033000 82.000000
	1.967000 56.000000
	4.500000 79.000000
	4.000000 71.000000
	1.983000 62.000000
	5.067000 76.000000
	2.017000 60.000000
	4.567000 78.000000
	3.883000 76.000000
	3.600000 83.000000
	4.133000 75.000000
	4.333000 82.000000
	4.100000 70.000000
	2.633000 65.000000
	4.067000 73.000000
	4.933000 88.000000
	3.950000 76.000000
	4.517000 80.000000
	2.167000 48.000000
	4.000000 86.000000
	2.200000 60.000000
	4.333000 90.000000
	1.867000 50.000000
	4.817000 78.000000
	1.833000 63.000000
	4.300000 72.000000
	4.667000 84.000000
	3.750000 75.000000
	1.867000 51.000000
	4.900000 82.000000
	2.483000 62.000000
	4.367000 88.000000
	2.100000 49.000000
	4.500000 83.000000
	4.050000 81.000000
	1.867000 47.000000
	4.700000 84.000000
	1.783000 52.000000
	4.850000 86.000000
	3.683000 81.000000
	4.733000 75.000000
	2.300000 59.000000
	4.900000 89.000000
	4.417000 79.000000
	1.700000 59.000000
	4.633000 81.000000
	2.317000 50.000000
	4.600000 85.000000
	1.817000 59.000000
	4.417000 87.000000
	2.617000 53.000000
	4.067000 69.000000
	4.250000 77.000000
	1.967000 56.000000
	4.600000 88.000000
	3.767000 81.000000
	1.917000 45.000000
	4.500000 82.000000
	2.267000 55.000000
	4.650000 90.000000
	1.867000 45.000000
	4.167000 83.000000
	2.800000 56.000000
	4.333000 89.000000
	1.833000 46.000000
	4.383000 82.000000
	1.883000 51.000000
	4.933000 86.000000
	2.033000 53.000000
	3.733000 79.000000
	4.233000 81.000000
	2.233000 60.000000
	4.533000 82.000000
	4.817000 77.000000
	4.333000 76.000000
	1.983000 59.000000
	4.633000 80.000000
	2.017000 49.000000
	5.100000 96.000000
	1.800000 53.000000
	5.033000 77.000000
	4.000000 77.000000
	2.400000 65.000000
	4.600000 81.000000
	3.567000 71.000000
	4.000000 70.000000
	4.500000 81.000000
	4.083000 93.000000
	1.800000 53.000000
	3.967000 89.000000
	2.200000 45.000000
	4.150000 86.000000
	2.000000 58.000000
	3.833000 78.000000
	3.500000 66.000000
	4.583000 76.000000
	2.367000 63.000000
	5.000000 88.000000
	1.933000 52.000000
	4.617000 93.000000
	1.917000 49.000000
	2.083000 57.000000
	4.583000 77.000000
	3.333000 68.000000
	4.167000 81.000000
	4.333000 81.000000
	4.500000 73.000000
	2.417000 50.000000
	4.000000 85.000000
	4.167000 74.000000
	1.883000 55.000000
	4.583000 77.000000
	4.250000 83.000000
	3.767000 83.000000
	2.033000 51.000000
	4.433000 78.000000
	4.083000 84.000000
	1.833000 46.000000
	4.417000 83.000000
	2.183000 55.000000
	4.800000 81.000000
	1.833000 57.000000
	4.800000 76.000000
	4.100000 84.000000
	3.966000 77.000000
	4.233000 81.000000
	3.500000 87.000000
	4.366000 77.000000
	2.250000 51.000000
	4.667000 78.000000
	2.100000 60.000000
	4.350000 82.000000
	4.133000 91.000000
	1.867000 53.000000
	4.600000 78.000000
	1.783000 46.000000
	4.367000 77.000000
	3.850000 84.000000
	1.933000 49.000000
	4.500000 83.000000
	2.383000 71.000000
	4.700000 80.000000
	1.867000 49.000000
	3.833000 75.000000
	3.417000 64.000000
	4.233000 76.000000
	2.400000 53.000000
	4.800000 94.000000
	2.000000 55.000000
	4.150000 76.000000
	1.867000 50.000000
	4.267000 82.000000
	1.750000 54.000000
	4.483000 75.000000
	4.000000 78.000000
	4.117000 79.000000
	4.083000 78.000000
	4.267000 78.000000
	3.917000 70.000000
	4.550000 79.000000
	4.083000 70.000000
	2.417000 54.000000
	4.183000 86.000000
	2.217000 50.000000
	4.450000 90.000000
	1.883000 54.000000
	1.850000 54.000000
	4.283000 77.000000
	3.950000 79.000000
	2.333000 64.000000
	4.150000 75.000000
	2.350000 47.000000
	4.933000 86.000000
	2.900000 63.000000
	4.583000 85.000000
	3.833000 82.000000
	2.083000 57.000000
	4.367000 82.000000
	2.133000 67.000000
	4.350000 74.000000
	2.200000 54.000000
	4.450000 83.000000
	3.567000 73.000000
	4.500000 73.000000
	4.150000 88.000000
	3.817000 80.000000
	3.917000 71.000000
	4.450000 83.000000
	2.000000 56.000000
	4.283000 79.000000
	4.767000 78.000000
	4.533000 84.000000
	1.850000 58.000000
	4.250000 83.000000
	1.983000 43.000000
	2.250000 60.000000
	4.750000 75.000000
	4.117000 81.000000
	2.150000 46.000000
	4.417000 90.000000
	1.817000 46.000000
	4.467000 74.000000
]

# ╔═╡ 0349720e-5de4-4b39-babd-c0881588f1de
X = Array(Matrix{Float64}(old_faithful)')

# ╔═╡ 8555aec9-4e80-49e7-8514-ef4a2236801b
N = size(X, 2)

# ╔═╡ 666680b2-315a-4d95-8f7f-3ae50018e112
K = 6

# ╔═╡ 86c33a7c-135a-461f-a17e-b50bca418e13
function sufficientStatistics(X,r,k::Int) #function to compute sufficient statistics
    N_k = sum(r[k,:])
    hat_x_k = sum([r[k,n]*X[:,n] for n in 1:N]) ./ N_k
    S_k = sum([r[k,n]*(X[:,n]-hat_x_k)*(X[:,n]-hat_x_k)' for n in 1:N]) ./ N_k
    return N_k, hat_x_k, S_k
end

# ╔═╡ 98a0ed70-a627-48d6-a1f8-3dec7aba2bb2
function updateMeanPrecisionPi(m_0,β_0,W_0,ν_0,α_0,r) #variational maximisation function
    m = Array{Float64}(undef,2,K) #mean of the clusters 
    β = Array{Float64}(undef,K) #precision scaling for Gausian distribution
    W = Array{Float64}(undef,2,2,K) #precision prior for Wishart distributions
    ν = Array{Float64}(undef,K) #degrees of freedom parameter for Wishart distribution
    α = Array{Float64}(undef,K) #Dirichlet distribution parameter 
    for k=1:K
        sst = sufficientStatistics(X,r,k)
        α[k] = α_0[k] + sst[1]; β[k] = β_0[k] + sst[1]; ν[k] = ν_0[k] .+ sst[1]
        m[:,k] = (1/β[k])*(β_0[k].*m_0[:,k] + sst[1].*sst[2])
        W[:,:,k] = inv(inv(W_0[:,:,k])+sst[3]*sst[1] + ((β_0[k]*sst[1])/(β_0[k]+sst[1])).*(sst[2]-m_0[:,k])*(sst[2]-m_0[:,k])')
    end
    return m,β,W,ν,α
end

# ╔═╡ 55a1c42b-20d8-47a3-aa00-7af905db537c
function updateR(Λ,m,α,ν,β) #variational expectation function
    r = Array{Float64}(undef,K,N) #responsibilities 
    hat_π = Array{Float64}(undef,K) 
    hat_Λ = Array{Float64}(undef,K)
    for k=1:K
        hat_Λ[k] = 1/2*(2*log(2)+logdet(Λ[:,:,k])+digamma(ν[k]/2)+digamma((ν[k]-1)/2))
        hat_π[k] = exp(digamma(α[k])-digamma(sum(α)))
        for n=1:N
           r[k,n] = hat_π[k]*exp(-hat_Λ[k]-1/β[k] - (ν[k]/2)*(X[:,n]-m[:,k])'*Λ[:,:,k]*(X[:,n]-m[:,k]))
        end
    end
    for n=1:N
        r[:,n] = r[:,n] ./ sum(r[:,n]) #normalize to ensure r represents probabilities 
    end
    return r
end

# ╔═╡ 26c796c8-d294-11ef-25be-17dcd4a9d315
md"""
The generated figure resembles Figure 10.6 in Bishop. The plots show VFEM results for a GMM of ``K = 6`` Gaussians applied to the Old Faithful data set. The ellipses denote the one standard-deviation density contours for each of the components, and the color coding of the data points reflects the "soft" class label assignments. Components whose expected mixing coefficients are numerically indistinguishable from zero are not plotted.

"""

# ╔═╡ 0090be18-2453-4ad3-8e2c-6953649b171e
TODO("Fons: can you make this into a cool code example where user gets to run through the iterations?")

# ╔═╡ f42a1a65-20ce-452f-9974-bc8146943574
md"""
# Theoretical Underpinning of VFE Minimization
"""

# ╔═╡ 26c7b428-d294-11ef-150a-bb37e37f4b5d
md"""
## Observations as Variational Constraints

We derived variational inference by substituting a variational posterior ``q(z)`` for the Bayesian posterior ``p(z|x)`` in the CA decomposition of (negative log) Bayesian evidence for a model. This led to a straightforward derivation of the VFE functional, but revealed nothing about the foundations of variational inference. Is variational inference any good?

To approach this question, let us first recognize that, in the context of a given model ``p(x,z)``, new observations ``x`` can generally be formulated as a constraint on a posterior distribution ``q``. For instance, observing a new data point ``x_1 = 5`` can be formalized as a constraint ``q(x_1) = \delta(x_1 - 5)``, where ``\delta(\cdot)`` is the Dirac delta function. 

Viewing observations as delta-function constraints enables us to interpret them as a specific instance of variational constraints, on par with form and factorization (and other) constraints, all of which shape the variational posterior in constrained VFE minimization.


"""

# ╔═╡ b3bb7349-1965-4734-83ed-ba6fef0ccc41
md"""

## Variational Inference and The Maximum Entropy Principle

In [Caticha (2010)](https://arxiv.org/abs/1011.0723) (based on earlier work by [Shore and Johnson (1980)](https://github.com/bmlip/course/blob/main/assets/files/ShoreJohnson-1980-Axiomatic-Derivation-of-the-Principle-of-Maximum-Entropy.pdf)), the [Principle of Maximum (Relative) Entropy](https://en.wikipedia.org/wiki/Principle_of_maximum_entropy) is developed as a method for rational updating of priors to posteriors when faced with new information in the form of constraints.


Caticha's argumentation is as follows:

  * Consider prior beliefs (i.e., a generative model) ``p(x,z)`` about observed and latent variables ``x`` and ``z``. Assume that new information in the form of constraints is obtained, and we are interested in the "best update" to posterior beliefs ``q(x,z)``.

  * In order to define what "best update" means, Caticha assumed a ranking function ``S[q]`` that generates a preference score for each candidate posterior ``q`` for a given prior ``p``. The best update from ``p`` to ``q`` is then identified as

```math
q^* = \arg\max_q S[q]\,, \quad \text{subject to all constraints.} 
```

Similarly to [Cox' method](https://en.wikipedia.org/wiki/Cox%27s_theorem) for deriving Probability Theory from a set of sensical axioms, Caticha then introduced the following axioms, based on a rational principle (the **principle of minimal updating**, see [Caticha 2010](https://arxiv.org/abs/1011.0723)), that the ranking function needs to adhere to: 

  1. *Locality*: local information has local effects.
  2. *Coordinate invariance*: the system of coordinates carries no information.
  3. *Independence*: When systems are known to be independent, it should not matter whether they are treated separately or jointly.

It turns out that these three criteria **uniquely identify the Relative Entropy** as the proper ranking function: 

```math
\begin{align*}
S[q] = - \sum_z q(x,z) \log \frac{q(x,z)}{p(x,z)}
\end{align*}
```

This procedure for finding the variational posterior ``q`` is called the **Principle of  Maximum (Relative) Entropy** (PME). Note that, since ``S[q]=-F[q]``, constrained Relative Entropy maximization is equivalent to constrained VFE minimization! 

Therefore, when information is supplied in the form of constraints on the posterior (such as form/factorization constraints and new observations as data constraints), we *should* select the posterior that minimizes the constrained Variational Free Energy. **Constrained FE minimization is the proper method for inference!**

Bayes rule is the global solution of constrained VFEM when all constraints are data constraints, ie, delta distributions on ``q(x)``. Hence, Bayes rule is a special case of constrained VFEM. Bayes rule only applies to updating beliefs on the basis of new observations. 
 
"""

# ╔═╡ 06170e31-e865-4178-8af0-41d82df95d71
keyconcept("","Constrained VFE minimization is consistent with the Maximum Entropy Principle, which prescribes how to rationally update beliefs when new information becomes available. In this framework, the updated posterior is the distribution that minimizes VFE (or equivalently, KL divergence to the prior) subject to the imposed constraints. ")

# ╔═╡ bbdca8c2-022f-42be-bcf7-80d86f7f269c
md"""

## Model Performance Evaluation, Revisited

Let us reconsider the Bound-Evidence decomposition of the VFE for a model ``p(x,z)`` with variational posterior ``q(z)``,

```math
\begin{align}
\mathrm{F}[q] = \underbrace{\sum_z q(z) \log \frac{q(z)}{p(z|x)}}_{\text{inference bound}\geq 0} \underbrace{- \log p(x)}_{\text{surprise}} \tag{BE} 
\end{align}
```

The VFE comprises two cost terms:

  - The **surprise** (or negative log-evidence), ``-\log p(x)``, reflects the cost of predicting the data ``x`` using a model ``p(x, z)``, assuming that (ideal) Bayesian inference can be performed. Specifically, the evidence ``p(x)`` is obtained from the joint model ``p(x, z)`` by marginalizing over the latent variables:
```math
p(x) = \sum_z p(x,z)  \,.
```

  - The **inference bound**, given by the Kullback–Leibler divergence
```math
\sum_z q(z) \log \frac{q(z)}{p(z |x)} \geq 0 \,,
``` 
quantifies the cost of imperfect inference, i.e., the discrepancy between the variational posterior ``q(z)`` and the true Bayesian posterior ``p(z | x)``.

In any practical setting, using a model *implies* performing inference within that model. Therefore, the effective cost of applying a model is not merely the surprise but also must include the cost of inference. 

Put more bluntly: a model with very high Bayesian evidence ``p(x)`` may still be practically unusable due to exorbitant inference costs.

In the literature, the VFE is typically interpreted as an approximation (more precisely, an upper-bound) to the surprise, ``-\log p(x)``, which is often regarded as the “true” measure of model performance. However, we argue that this perspective should be reversed: the VFE should be considered the true performance metric in practice, as it accounts for both model fit and the tractability of inference. The surprise can be viewed as a special case of the VFE, corresponding to a zero inference bound, that only applies when ideal Bayesian inference is computationally feasible. 

"""

# ╔═╡ 26c8068a-d294-11ef-3983-a1be55128b3f
md"""
## Variational Inference in Practice

For most realistic models of complex real-world problems, Bayes rule is not tractable in closed form. As a result, the use of approximate variational Bayesian inference has seen rapid growth in practical applications.

Toolboxes such as [RxInfer](http://rxinfer.com) enable users to define sophisticated probabilistic models and automate the inference process via constrained VFE minimization. Remarkably, specifying even complex models typically requires no more than a single page of code. 

In contrast to traditional algorithm design, where solving a problem might require implementing a custom solution in, say, ``40`` pages of code, automated inference in a probabilistic model offers a radically more efficient and modular approach. This shift has the potential to fundamentally change how we design and deploy information processing systems in the future.

"""

# ╔═╡ 56bea391-b812-4fc4-8f27-fcb4cb984cf4
md"""
# Exercises
"""

# ╔═╡ 5a94e2a4-7134-462e-9dc5-56083769049f
md"""
#### Entropy and The Free Energy Functional (*)

The Free energy functional ``\mathrm{F}[q] = -\sum_z q(z) \log p(x,z) - \sum_z q(z) \log \frac{1}{q(z)}`` decomposes into "Energy minus Entropy". So apparently the entropy of the posterior ``q(z)`` is maximized. This entropy maximization may seem puzzling at first because inference should intuitively lead to *more* informed posteriors, i.e., posterior distributions whose entropy is smaller than the entropy of the prior. Explain why entropy maximization is still a reasonable objective. 

 
"""

# ╔═╡ 747a7e1e-b921-4882-b00a-1b00bef8433d
details("Click for answer",
md"""

Note that Free Energy minimization is a balancing act: FE minimization implies entropy maximization *and at the same time* energy minimization. Minimizing the energy term leads to aligning ``q(z)`` with ``\log p(x,z)``, ie, it tries to move the bulk of the function ``q(z)`` to areas in ``z``-space where ``p(x,z)`` is large (``p(x,z)`` is here just a function of ``z``, since x is observed). 
	   
However, aside from aligning with ``p(x,z)``, we want ``q(z)`` to be as uninformative as possible. Everything that can be inferred should be represented in ``p(x,z)`` (which is prior times likelihood). We don't want to learn anything that is not in either the prior or the likelihood. The entropy term balances the energy term by favoring distributions that are as uninformative as possible.
 
""")

# ╔═╡ 2d4adbf6-6de8-4e3a-ad6f-fa8bbfa5999e
md"""

#### Mean Updating (*)

Explain the following update rule for the [mean of the Gaussian cluster-conditional data distribution](#update-equations):

```math
m_k = \frac{1}{\beta_k} \left( \beta_0 m_0 + N_k \bar{x}_k \right) \tag{B-10.61} 
```

"""

# ╔═╡ 208ba1bb-a4bf-4b8c-93d2-0d6c6c8d16d4
details("Click for answer",
md"""
We see here an example of "precision-weighted means add" when two sources of information are fused, just like precision-weighted means add when two Gaussians are multiplied, eg a prior and likelihood. In this case, the prior is ``m_0`` and the likelihood estimate is ``\bar{x}``. ``\beta_0`` can be interpreted as the number of pseudo-observations in the prior.

""")

# ╔═╡ 2f490e1f-e495-4f55-a3f8-60d6fd716d4e
md"""
#### The Expectation-Maximization (EM) algorithm (**)

Consider a model ``p(x,z|\theta)``, where ``D=\{x_1,x_2,\ldots,x_N\}`` is observed, ``z`` are unobserved variables, and ``\theta`` are parameters. The **Expectation-Maximization** (EM) algorithm estimates the parameters by iterating over the following two equations (``i`` is the iteration index):

```math
\begin{align*}
q^{(i)}(z) &= p(z|D,\theta^{(i-1)}) \\
\theta^{(i)} &= \arg\max_\theta \sum_z q^{(i)}(z) \cdot \log p(D,z|\theta)
\end{align*}
```

Proof that this algorithm minimizes the Free Energy functional 

```math
\begin{align*}
F[q](\theta) =  \sum_z q(z) \log \frac{q(z)}{p(D,z|\theta)} 
\end{align*}
```


"""

# ╔═╡ b91bc3b6-b815-4942-b297-c0e2b4b99654
details("Click for answer",
md"""
		
Let's start with a prior estimate ``\theta^{(i-1)}`` and we want to minimize the free energy functional wrt ``q``. This leads to


```math
\begin{align*}
q^{(i)}(z) &= \arg\min_q F[q](\theta^{(i-1)}) \\
  &= \arg\min_q \sum_z q(z) \log \frac{q(z)}{p(D,z|\theta^{(i-1)})} \\
  &= \arg\min_q \sum_z q(z) \log \frac{q(z)}{p(z|D,\theta^{(i-1)}) \cdot p(D|\theta^{(i-1)})} \\
  &= p(z|D,\theta^{(i-1)})
\end{align*}
```

Next, we use ``q^{(i)}(z)=p(z|D,\theta^{(i-1)})`` and minimize the free energy w.r.t. ``\theta``, leading to

```math
\begin{align*}
  \theta^{(i)} &= \arg\min_\theta F[q^{(i)}](\theta) \\
  &= \arg\min_\theta \sum_z p(z|D,\theta^{(i-1)}) \log \frac{p(z|D,\theta^{(i-1)})}{p(D,z|\theta)} \\
  &= \arg\max_\theta \sum_z \underbrace{p(z|D,\theta^{(i-1)})}_{q^{(i)}(z)} \log p(D,z|\theta)
\end{align*}
```
		""")

# ╔═╡ 26c8160c-d294-11ef-2a74-6f7009a7c51e
md"""
# $(HTML("<span id='optional-slides'>OPTIONAL SLIDES</span>"))

"""

# ╔═╡ 26c82f16-d294-11ef-0fe1-07326b56282f
md"""
## VFE Minimization with Mean-field Factorization Constraints: $(HTML("<span id='CAVI'>the CAVI Approach</span>"))

Let's work out VFE minimization with additional mean-field constraints (=full factorization) constraints:  

```math
q(z) = \prod_{j=1}^m q_j(z_j)\,.
```

In other words, the posteriors for ``z_j`` are all considered independent. This is a strong constraint but often leads to good solutions.

Given the mean-field constraints, it is possible to derive the following expression for the $(HTML("<span id='optimal-solutions'>optimal solutions</span>")) ``q_j^*(z_j)``, for ``j=1,\ldots,m``: 

```math
\begin{align} 
\log q_j^*(z_j) &\propto \mathrm{E}_{q_{-j}^*}\left[ \log p(x,z) \right]  \\
  &= \underbrace{\sum_{z_{-j}} q_{-j}^*(z_{-j}) \underbrace{\log p(x,z)}_{\text{"field"}}}_{\text{"mean field"}} 
\end{align} 
```

where we defined ``q_{-j}^*(z_{-j}) \triangleq q_1^*(z_1)q_2^*(z_2)\cdots q_{j-1}^*(z_{j-1})q_{j+1}^*(z_{j+1})\cdots q_m^*(z_m)``.

**Proof** (from [Blei, 2017](https://doi.org/10.1080/01621459.2017.1285773)): We first rewrite the FE as a function of ``q_j(z_j)`` only: 

```math
 F[q_j] = \mathbb{E}_{q_{j}}\left[ \mathbb{E}_{q_{-j}}\left[ \log p(x,z_j,z_{-j})\right]\right] - \mathbb{E}_{q_j}\left[ \log q_j(z_j)\right] + \mathtt{const.}\,,
```

where the constant holds all terms that do not depend on ``z_j``. This expression can be written as 

```math
 F[q_j] = \sum_{z_j} q_j(z_j) \log \frac{q_j(z_j)}{\exp\left( \mathbb{E}_{q_{-j}}\left[ \log p(x,z_j,z_{-j})\right]\right)}
```

which is a KL-divergence that is minimized by Eq. B-10.9.  (end proof)

This is not yet a full solution to the FE minimization task since the solution ``q_j^*(z_j)`` depends on expectations that involve other solutions ``q_{i\neq j}^*(z_{i \neq j})``, and each of these other solutions ``q_{i\neq j}^*(z_{i \neq j})`` depends on an expection that involves ``q_j^*(z_j)``. 

In practice, we solve this chicken-and-egg problem by an iterative approach: we first initialize all ``q_j(z_j)`` (for ``j=1,\ldots,m``) to an appropriate initial distribution and then cycle through the factors in turn by solving eq.B-10.9 and update ``q_{-j}^*(z_{-j})`` with the latest estimates. (See [Blei, 2017](https://doi.org/10.1080/01621459.2017.1285773), Algorithm 1, p864).  

This algorithm for approximating Bayesian inference is known **Coordinate Ascent Variational Inference** (CAVI).   

"""

# ╔═╡ 26c85a22-d294-11ef-3c8e-7b72a4313ced
md"""
## $(HTML("<span id='EM-Algorithm'>FE Minimization by the Expectation-Maximization (EM) Algorithm</span>"))

The EM algorithm is a special case of VFE minimization that focuses on Maximum-Likelihood estimation for models with latent variables. 

Consider a model 

```math
p(x,z,\theta)
```

with observations ``x = \{x_n\}``, latent variables ``z=\{z_n\}`` and parameters ``\theta``.

We can write the following VFE functional for this model:

```math
\begin{align*}
F[q] =  \sum_z \sum_\theta q(z,\theta) \log \frac{q(z,\theta)}{p(x,z,\theta)} 
\end{align*}
```

The EM algorithm makes the following simplifying assumptions:

1. The prior for the parameters is uninformative (uniform). This implies that

```math
p(x,z,\theta) = p(x,z|\theta) p(\theta) \propto p(x,z|\theta)
```

2. A factorization constraint 

```math
q(z,\theta) = q(z) q(\theta)
```

3. The posterior for the parameters is a delta function:

```math
q(\theta) = \delta(\theta - \hat{\theta})
```

Basically, these three assumptions turn VFE minimization into maximum likelihood estimation for the parameters ``\theta`` and the VFE simplifies to 

```math
\begin{align*}
F[q,\theta] =  \sum_z q(z) \log \frac{q(z)}{p(x,z|\theta)} 
\end{align*}
```

The EM algorithm minimizes this FE by iterating (iteration counter: ``i``) over 

```math
\begin{align} \mathcal{L}^{(i)}(\theta) &= \sum_z \overbrace{p(z|x,\theta^{(i-1)})}^{q^{(i)}(z)}  \log p(x,z|\theta) \tag{the E-step} \\
\theta^{(i)} &= \arg\max_\theta \mathcal{L}^{(i)}(\theta) \tag{the M-step} \end{align}

```

These choices are optimal for the given FE functional. In order to see this, consider the two decompositions

```math
\begin{align*}
F[q,\theta] &= \underbrace{-\sum_z q(z) \log p(x,z|\theta)}_{\text{energy}} - \underbrace{\sum_z q(z) \log \frac{1}{q(z)}}_{\text{entropy}} \qquad &&\text{(EE)}\\
  &= \underbrace{\sum_z q(z) \log \frac{q(z)}{p(z|x,\theta)}}_{\text{divergence}} - \underbrace{\log p(x|\theta)}_{\text{log-likelihood}}  \qquad &&\text{(DE)}
\end{align*}
```

The DE decomposition shows that the FE is minimized for the choice ``q(z) := p(z|x,\theta)``. Also, for this choice, the FE equals the (negative) log-evidence (, which is this case simplifies to the log-likelihood). 

The EE decomposition shows that the FE is minimized wrt ``\theta`` by minimizing the energy term. The energy term is computed in the E-step and optimized in the M-step.

  * Note that in the EM literature, the energy term is often called the *expected complete-data log-likelihood*.)

In order to execute the EM algorithm, it is assumed that we can analytically execute the E- and M-steps. For a large set of models (including models whose distributions belong to the exponential family of distributions), this is indeed the case and hence the large popularity of the EM algorithm. 

The EM algorihm imposes rather severe assumptions on the FE (basically approximating Bayesian inference by maximum likelihood estimation). Over the past few years, the rise of Probabilistic Programming languages has dramatically increased the range of models for which the parameters can by estimated autmatically by (approximate) Bayesian inference, so the popularity of EM is slowly waning. (More on this in the Probabilistic Programming lessons). 

Bishop (2006) works out EM for the GMM in section 9.2.

"""

# ╔═╡ 26c867d8-d294-11ef-2372-d75ed0bcc02d
md"""
## Code Example: EM-algorithm for the GMM on the Old-Faithful data set

We'll perform clustering on the data set from the [illustrative example](#illustrative-example) by fitting a GMM consisting of two Gaussians using the EM algorithm. 

"""

# ╔═╡ de049d59-9863-4bac-91c3-32851cad15d9
TODO("verify the `π_hat` situation in the code above")


# ╔═╡ 26c8a2a4-d294-11ef-1cd3-850e877d7a25
md"""
<!–- Note that you can step through the interactive demo yourself by running [this script](https://github.com/bertdv/AIP-5SSB0/blob/master/lessons/notebooks/scripts/interactive_em_demo.jl) in julia. You can run a script in julia by     `julia> include("path/to/script-name.jl")` –>

"""

# ╔═╡ 26c8b682-d294-11ef-1331-2bcf8baec73f
md"""
## Message Passing for Free Energy Minimization

The Sum-Product (SP) update rule implements perfect Bayesian inference. 

Sometimes, the SP update rule is not analytically solvable. 

Fortunately, for many well-known Bayesian approximation methods, a message passing update rule can be created, e.g. [Variational Message Passing](https://en.wikipedia.org/wiki/Variational_message_passing) (VMP) for variational inference. 

In general, all of these message passing algorithms can be interpreted as minimization of a constrained free energy (e.g., see [Senoz et al. (2021)](https://research.tue.nl/nl/publications/variational-message-passing-and-local-constraint-manipulation-in-), and hence these message passing schemes comply with [Caticha's Method of Maximum Relative Entropy](https://arxiv.org/abs/1011.0723), which, as discussed in the [variational Bayes lesson](https://bmlip.github.io/course/lectures/Latent%20Variable%20Models%20and%20VB.html) is the proper way for updating beliefs. 

Different message passing updates rules can be combined to get a hybrid inference method in one model. 

"""

# ╔═╡ 26c8c7fa-d294-11ef-0444-6555ecf5c721
md"""
## The Local Free Energy in a Factor Graph

Consider an edge ``x_j`` in a Forney-style factor graph for a generative model ``p(x) = p(x_1,x_2,\ldots,x_N)``.

Assume that the graph structure (factorization) is specified by

```math
p(x) = \prod_{a=1}^M p_a(x_a)
```

where ``a`` is a set of indices.

Also, we assume a mean-field approximation for the posterior:

```math
q(x) = \prod_{i=1}^N q_i(x_i)
```

and consequently a corresponding free energy functional  

```math
\begin{align*}
F[q] &= \sum_x q(x) \log \frac{q(x)}{p(x)} \\
  &= \sum_i \sum_{x_i} \left(\prod_{i=1}^N q_i(x_i)\right) \log \frac{\prod_{i=1}^N q_i(x_i)}{\prod_{a=1}^M p_a(x_a)}
\end{align*}
```

With these assumptions, it can be shown that the FE evaluates to (exercise)

```math
F[q] = \sum_{a=1}^M \underbrace{\sum_{x_a} \left( \prod_{j\in N(a)} q_j(x_j)\cdot \left(-\log p_a(x_a)\right) \right) }_{\text{node energy }U[p_a]} - \sum_{i=1}^N \underbrace{\sum_{x_i} q_i(x_i) \log \frac{1}{q_i(x_i)}}_{\text{edge entropy }H[q_i]}
```

In words, the FE decomposes into a sum of (expected) energies for the nodes minus the entropies on the edges. 

"""

# ╔═╡ 26c8e172-d294-11ef-2a9e-89e0f4cbf475
md"""
## Variational Message Passing

Let us now consider the local free energy that is associated with edge corresponding to ``x_j``. 

![](https://github.com/bmlip/course/blob/v2/assets/figures/VMP-two-nodes.png?raw=true)

Apparently (see previous slide), there are three contributions to the free energy for ``x_j``:

  * one entropy term for the edge ``x_j``
  * two energy terms: one for each node that attaches to ``x_j`` (in the figure: nodes ``p_a`` and ``p_b``)

The local free energy for ``x_j`` can be written as (exercise)

```math
  F[q_j] \propto \sum_{x_j} q(x_j) \log \frac{q_j(x_j)}{\nu_a(x_j)\cdot \nu_b(x_j)}
  
```

where

```math
\begin{align*} 
  \nu_a(x_j) &\propto \exp\left( \mathbb{E}_{q_{k}}\left[ \log p_a(x_a)\right]\right) \\
  \nu_b(x_j) &\propto \exp\left( \mathbb{E}_{q_{l}}\left[ \log p_b(x_b)\right]\right) 
  \end{align*}
```

and ``\mathbb{E}_{q_{k}}\left[\cdot\right]`` is an expectation w.r.t. all ``q(x_k)`` with ``k \in N(a)\setminus {j}``.

``\nu_a(x_j)`` and ``\nu_b(x_j)``  can be locally computed in nodes ``a`` and ``b`` respectively and can be interpreted as colliding messages over edge ``x_j``. 

Local free energy minimization is achieved by setting

```math
  q_j(x_j) \propto \nu_a(x_j) \cdot \nu_b(x_j)
  
```

Note that message ``\nu_a(x_j)`` depends on posterior beliefs over incoming edges (``k``) for node ``a``, and in turn, the message from node ``a`` towards edge ``x_k`` depends on the belief ``q_j(x_j)``. I.o.w., direct mutual dependencies exist between posterior beliefs over edges that attach to the same node. 

These considerations lead to the [Variational Message Passing](https://en.wikipedia.org/wiki/Variational_message_passing) procedure, which is an iterative free energy minimization procedure that can be executed completely through locally computable messages.  

Procedure VMP, see [Dauwels (2007), section 3](https://github.com/bmlip/course/blob/main/assets/files/Dauwels-2007-on-variational-message-passing-on-factor-graphs.pdf)

> 1. Initialize all messages ``q`` and ``ν``, e.g., ``q(\cdot) \propto 1`` and ``\nu(\cdot) \propto 1``. <br/>
> 2. Select an edge ``z_k`` in the factor graph of ``f(z_1,\ldots,z_m)``.<br/>
> 3. Compute the two messages ``\overrightarrow{\nu}(z_k)`` and ``\overleftarrow{\nu}(z_k)`` by applying the following generic rule:
> ```math
>   \overrightarrow{\nu}(y) \propto \exp\left( \mathbb{E}_{q}\left[ \log > g(x_1,\dots,x_n,y)\right] \right)   
> ```
> 4. Compute the marginal ``q(z_k)``
> ```math
>  q(z_k) \propto \overrightarrow{\nu}(z_k) \overleftarrow{\nu}(z_k)  
> ```
>  and send it to the two nodes connected to the edge ``x_k``.
>
> 5. Iterate 2–4 until convergence.


"""

# ╔═╡ 26c9121e-d294-11ef-18e6-ed8105503adc
md"""
## The Bethe Free Energy and Belief Propagation

We showed that, under mean field assumptions, the FE can be decomposed into a sum of local FE contributions for the nodes (``a``) and edges (``i``):

```math
\begin{align*}
F[q] = \sum_{a=1}^M \underbrace{\sum_{x_a} \left( \prod_{j\in N(a)} q_j(x_j)\cdot \left(-\log p_a(x_a)\right) \right) }_{\text{node energy }U[p_a]} - \sum_{i=1}^N \underbrace{\sum_{x_i} q_i(x_i) \log \frac{1}{q_i(x_i)}}_{\text{edge entropy }H[q_i]}
\end{align*}
```

The mean field assumption is very strong and may lead to large inference costs (``\mathrm{KL}(q(x),p(x|\text{data}))``). A more relaxed assumption is to allow joint posterior beliefs over the variables that attach to a node. This idea is expressed by the Bethe Free Energy:

```math
\begin{align*}
F_B[q] = \sum_{a=1}^M \left( \sum_{x_a} q_a(x_a) \log \frac{q_a(x_a)}{p_a(x_a)} \right)  - \sum_{i=1}^N (d_i - 1) \sum_{x_i} q_i(x_i) \log {q_i(x_i)}
\end{align*}
```

where ``q_a(x_a)`` is the posterior joint belief over the variables ``x_a`` (i.e., the set of variables that attach to node ``a``), ``q_i(x_i)`` is the posterior marginal belief over the variable ``x_i`` and ``d_i`` is the number of factor nodes that link to edge ``i``. Moreover, ``q_a(x_a)`` and ``q_i(x_i)`` are constrained to obey the following equalities:

```math
\begin{align*}
  \sum_{x_a \backslash x_i} q_a(x_a) &= q_i(x_i), ~~~ \forall i, \forall a \\
  \sum_{x_i} q_i(x_i) &= 1, ~~~ \forall i \\
  \sum_{x_a} q_a(x_a) &= 1, ~~~ \forall a \\
\end{align*}
```

We form the Lagrangian by augmenting the Bethe Free Energy functional with the constraints:

```math
\begin{align*}
L[q] = F_B[q] + \sum_i\sum_{a \in N(i)} \lambda_{ai}(x_i) \left(q_i(x_i) - \sum_{x_a\backslash x_i} q(x_a) \right) + \sum_{i} \gamma_i \left(  \sum_{x_i}q_i(x_i) - 1\right) + \sum_{a}\gamma_a \left(  \sum_{x_a}q_a(x_a) -1\right)
\end{align*}
```

The stationary solutions for this Lagrangian are given by

```math
\begin{align*}
q_a(x_a) &= f_a(x_a) \exp\left(\gamma_a -1 + \sum_{i \in N(a)} \lambda_{ai}(x_i)\right) \\ 
q_i(x_i) &= \exp\left(1- \gamma_i + \sum_{a \in N(i)} \lambda_{ai}(x_i)\right) ^{\frac{1}{d_i - 1}}
\end{align*}
```

where ``N(i)`` denotes the factor nodes that have ``x_i`` in their arguments and ``N(a)`` denotes the set of variables in the argument of ``f_a``.

Stationary solutions are functions of Lagrange multipliers. This means that Lagrange multipliers need to be determined. Lagrange multipliers can be determined by plugging the stationary solutions back into the constraint specification and solving for the multipliers which ensure that the constraint is satisfied. The first constraint we consider is normalization, which yields the following identification:

```math
\begin{align*}
\gamma_a &= 1 - \log \Bigg(\sum_{x_a}f_a(x_a)\exp\left(\sum_{i \in N(a)}\lambda_{ai}(x_i)\right)\Bigg)\\
\gamma_i &= 1 + (d_i-1) \log\Bigg(\sum_{x_i}\exp\left( \frac{1}{d_i-1}\sum_{a \in N(i)} \lambda_{ai}(x_i)\right)\Bigg).
\end{align*}
```

The functional form of the Lagrange multipliers that corresponds to the normalization constraint enforces us to obtain the Lagrange multipliers that correspond to the marginalization constraint. To do so we solve for 

```math
\begin{align*} \sum_{x_a \backslash x_i} f_a(x_a) \exp\left(\sum_{i \in N(a)} \lambda_{ai}(x_i)\right) &= \exp\left(\sum_{a \in N(i)} \lambda_{ai}(x_i)\right) ^{\frac{1}{d_i - 1}} \exp\left(\lambda_{ai}(x_i)\right)\sum_{x_a \backslash x_i} f_a(x_a) \exp\Bigg(\sum_{\substack{{j \in N(a)}  j \neq i}}\lambda_{aj}(x_j)\Bigg) \\
&= \exp\left(\sum_{a \in N(i)} \lambda_{ai}(x_i)\right) ^{\frac{1}{d_i - 1}} \exp\left(\lambda_{ai}(x_i) + \lambda_{ia}(x_i)\right) \\
&= \exp\left(\sum_{a \in N(i)} \lambda_{ai}(x_i)\right) ^{\frac{1}{d_i - 1}}\, , 
\end{align*}
```

where we defined an auxilary function

```math
\begin{align*}
\exp(\lambda_{ia}(x_i)) \triangleq \sum_{x_a \backslash x_i} f_a(x_a) \exp\Bigg(\sum_{\substack{{j \in N(a)} j \neq i}}\lambda_{aj}(x_j)\Bigg) \,.
\end{align*}
```

This definition is valid since it can be inverted by the relation

```math
\begin{align*}
\lambda_{ia}(x_i) = \frac{2-d_i}{d_i - 1}\lambda_{ai}(x_i) + \frac{1}{d_i -1}\sum_{\substack{c \in N(i)\\c \neq a}}\lambda_{ci}(x_i)
\end{align*}
```

In general it is not possible to solve for the Lagrange multipliers analytically and we resort to iteratively obtaining the solutions. This leads to the **Belief Propagation algorithm** where the exponentiated Lagrange multipliers (messages) are updated iteratively via 

```math
\begin{align*} 
\mu_{ia}^{(k+1)}(x_i) &= \sum_{x_a \backslash x_i} f_a(x_a) \prod_{\substack{{j \in N(a)}  j \neq i}}\mu^{(k)}_{aj}(x_j)  \mu_{ai}^{(k)}(x_i) \\
&= \prod_{\substack{c \in N(i) c \neq a}}\mu^{(k)}_{ic}(x_i)\,, 
\end{align*}
```

where ``k`` denotes iteration number and the messages are defined as

```math
\begin{align*}
\mu_{ia}(x_i) &\triangleq \exp(\lambda_{ia}(x_i))\\
\mu_{ai}(x_i) &\triangleq \exp(\lambda_{ai}(x_i))\,.
\end{align*}
```

For a more complete overview of message passing as Bethe Free Energy minimization, see [Senoz et al. (2021)](https://research.tue.nl/nl/publications/variational-message-passing-and-local-constraint-manipulation-in-).

"""

# ╔═╡ 55570464-89c8-4d9b-b667-dfa64ac62294
md"""
# Appendix
"""

# ╔═╡ 489cbd24-1a69-4a00-a2e9-53c2c57cef65
function plotGMM(X::Matrix, clusters::Vector, γ::Matrix, title)
	# Plot data set and (fitted) mixture model consisting of two Gaussian distributions
	# X contains a 2-d data set (every column holds a data point)
	# clusters holds the 2 Gaussian elements of the mixture model
	# γ contains p(cluster|X), and should contain NaN elements if not yet known

	# Plot contours of the element distributions
	K = length(clusters)
	result = plot(title=title)
	for k=1:K
		X1 = Matrix{Float64}(undef,50,50)
		X2 = Matrix{Float64}(undef,50,50)
		d = Matrix{Float64}(undef,50,50)
		# Create bounding box for thse contour plot
		lims = [-2*sqrt(cov(clusters[k])[1,1]) 2*sqrt(cov(clusters[k])[1,1]);
				-2*sqrt(cov(clusters[k])[2,2]) 2*sqrt(cov(clusters[k])[2,2])] + repeat(mean(clusters[k]), 1, 2)
		X1 = LinRange(lims[1,1], lims[1,2], 50)
		X2 = LinRange(lims[2,1], lims[2,2], 50)
		alpha = sum(γ[k,:])/sum(γ)
		covellipse!(clusters[k].μ, clusters[k].Σ; label="", alpha=max(0.1, alpha), color=:cyan)
	end


	# Plot data points
	scatter!(X[1,:], X[2,:]; label="observations", markersize=2, linewidth=0)
	return result
end

# ╔═╡ 4ee377c2-a126-4c40-8053-517d40c5ef9d
let
	max_iter = 120
	#store the inference results in these vectors
	ν = fill(3.0, K, max_iter)
	β = fill(1.0, K, max_iter)
	α = fill(0.01, K, max_iter)
	R = Array{Float64}(undef,K,N,max_iter)
	M = Array{Float64}(undef,2,K,max_iter)
	Λ = Array{Float64}(undef,2,2,K,max_iter)
	clusters_vb = Array{Distribution}(undef,K,max_iter) #clusters to be plotted
	#initialize prior distribution parameters
	M[:,:,1] = rand(MersenneTwister(42), 2, K) .* [4, 50] .+ [1, 50]
	for k in 1:K
	    Λ[:,:,k,1] = [1.0 0;0 0.01]
	    R[k,:,1] = 1/(K)*ones(N)
	    clusters_vb[k,1] = MvNormal(M[:,k,1],PDMats.PDMat(convert(Matrix,Hermitian(inv(ν[1,1].*Λ[:,:,k,1])))))
	end
	#variational inference
	for i in 1:max_iter-1
	    #variational expectation 
	    R[:,:,i+1] = updateR(Λ[:,:,:,i],M[:,:,i],α[:,i],ν[:,i],β[:,i]) 
	    #variational minimisation
	    M[:,:,i+1],β[:,i+1],Λ[:,:,:,i+1],ν[:,i+1],α[:,i+1] = updateMeanPrecisionPi(M[:,:,i],β[:,i],Λ[:,:,:,i],ν[:,i],α[:,i],R[:,:,i+1])
	    for k in 1:K
	        clusters_vb[k,i+1] = MvNormal(M[:,k,i+1],PDMats.PDMat(convert(Matrix,Hermitian(inv(ν[k,i+1].*Λ[:,:,k,i+1])))))
	    end
	end
	
	plots = [plotGMM(X, clusters_vb[:,1], R[:,:,1], "Initial situation")]
	for i in LinRange(2, 120, 5)
	    i = round(Int,i)
	    push!(plots, plotGMM(X, clusters_vb[:,i], R[:,:,i], "After $(i) iterations"))
	end
	plot(plots..., layout=(2,3), size=(1100, 600))
end

# ╔═╡ 7a3c0ff7-0b32-4954-ae28-b644f4d966ef
begin
	# Initialize the GMM. We assume 2 clusters.
	clusters = [MvNormal([4.;60.], [.5 0;0 10^2]); 
	            MvNormal([2.;80.], [.5 0;0 10^2])];
	π_hat = [0.5; 0.5]                    # Mixing weights
	γ = fill!(Matrix{Float64}(undef,2,N), NaN)  # Responsibilities (row per cluster)
	
	# Define functions for updating the parameters and responsibilities
	function updateResponsibilities!(X, clusters, π_hat, γ)
	    # Expectation step: update γ
	    norm = [pdf(clusters[1], X) pdf(clusters[2], X)] * π_hat
	    γ[1,:] = (π_hat[1] * pdf(clusters[1],X) ./ norm)'
	    γ[2,:] = 1 .- γ[1,:]
	end
	
	# TODO: Julia convention is that the arguments that get modified (`clusters` and `π_hat` (but see comment below)) appear before arguments that are not modified (`X`, `γ`)
	function updateParameters!(X, clusters, π_hat, γ)
	    # Maximization step: update π_hat and clusters using ML estimation
	    m = sum(γ, dims=2)
	
		# TODO: this does not update π_hat globally, it only creates a new local variable `π_hat` (which is not used)
	    π_hat = m / N
	    μ_hat = (X * γ') ./ m'
	    for k=1:2
	        Z = (X .- μ_hat[:,k])
	        Σ_k = Symmetric(((Z .* (γ[k,:])') * Z') / m[k])
	        clusters[k] = MvNormal(μ_hat[:,k], convert(Matrix, Σ_k))
	    end
	end
		
	
	# Execute the algorithm: iteratively update parameters and responsibilities
	plots = [plotGMM(X, clusters, γ, "Initial situation")]
	
	updateResponsibilities!(X, clusters, π_hat, γ)
	push!(plots, plotGMM(X, clusters, γ, "After first E-step"))
	updateParameters!(X, clusters, π_hat, γ)
	push!(plots, plotGMM(X, clusters, γ, "After first M-step"))
	
	local iter_counter = 1
	for i=1:3
	    for j=1:i+1
	        updateResponsibilities!(X, clusters, π_hat, γ)
	        updateParameters!(X, clusters, π_hat, γ)
	        iter_counter += 1
	    end
	    push!(plots, plotGMM(X, clusters, γ, "After $(iter_counter) iterations"))
	end
	
	plot(plots..., layout=(2,3), size=(1100, 600))
end

# ╔═╡ deba376e-59bd-4b07-814c-8f7937db52a5
challenge_header(
	title; 
	color="green",
	big::Bool=false,
	header_level::Int=2,
	challenge_text="Challenge:",
) = HypertextLiteral.@htl """
<$("h$header_level") class="ptt-section $(big ? "big" : "")" style="--ptt-accent: $(color);"><span>$(challenge_text)</span> $(title)</$("h$header_level")>
	
<style>
.ptt-section::before {
	content: "";
	display: block;
	position: absolute;
	left: -25px;
	right: -6px;
	top: -4px;
	height: 200px;
	border: 4px solid salmon;
	border-bottom: none;
	border-image-source: linear-gradient(to bottom, var(--ptt-accent), transparent);
	border-image-slice: 1;
	opacity: .7;
	pointer-events: none;
}

.big.ptt-section::before {
	height: 500px;
}
	

.ptt-section > span {
	color: color-mix(in hwb, var(--ptt-accent) 60%, black);
	@media (prefers-color-scheme: dark) {
		color: color-mix(in hwb, var(--ptt-accent) 30%, white);
	}
	font-style: italic;
}
</style>
"""

# ╔═╡ 26c591fc-d294-11ef-0423-b7a854d09bad
md"""

$(challenge_header("Density Modeling for the Old Faithful Data Set"; challenge_text="Challenge:"))

You're now asked to build a density model for a data set ([Old Faithful](https://en.wikipedia.org/wiki/Old_Faithful), Bishop pg. 681) that clearly is not distributed as a single Gaussian:

![](https://github.com/bmlip/course/blob/v2/assets/figures/fig-Bishop-A5-Old-Faithfull.png?raw=true)

"""

# ╔═╡ 26c7696e-d294-11ef-25f2-dbc0946c0858
md"""

$(challenge_header("VFEM for GMM on Old Faithfull data set"; challenge_text="Code Example:"))


Below we exemplify training of a Gaussian Mixture Model on the Old Faithful data set by VFE minimization, with the constraints as specified above. 

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.118"
HypertextLiteral = "~0.9.5"
PDMats = "~0.11.32"
Plots = "~1.40.10"
PlutoTeachingTools = "~0.3.1"
PlutoUI = "~0.7.23"
SpecialFunctions = "~2.5.0"
StatsPlots = "~0.15.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "59e8044d279caf7d25835d1497116e8f1412478f"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

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
version = "1.1.2"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

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

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "06ee8d1aa558d2833aa799f6f0b31b30cada405f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.2"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

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

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "3a3dfb30697e96a440e4149c8c51bf32f818c0f3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.17.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

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

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

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

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

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
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "797762812ed063b9b94f6cc7742bc8883bb5e69e"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.9.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

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

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "ba51324b894edaf1df3ab16e2cc6bc3280a2f1a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.10"

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

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

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

[[deps.LoweredCodeUtils]]
deps = ["Compiler", "JuliaInterpreter"]
git-tree-sha1 = "bc54ba0681bb71e56043a1b923028d652e78ee42"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.4.1"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

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

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "8a3271d8309285f4db73b4f662b1b290c715e85e"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.21"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

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
git-tree-sha1 = "87510f7292a2b21aeff97912b0898f9553cc5c2c"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

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
git-tree-sha1 = "55818b50883d7141bd98cdf5fc2f4ced96ee075f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.16"

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

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
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
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

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

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

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

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

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
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

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

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "e9aeb174f95385de31e70bd15fa066a505ea82b9"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.7"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

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

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
"""

# ╔═╡ Cell order:
# ╟─26c56fd8-d294-11ef-236d-81deef63f37c
# ╟─ce7d086b-ff20-4da1-a4e8-52b5b7dc9e2b
# ╟─26c58298-d294-11ef-2a53-2b42b48e0725
# ╟─26c591fc-d294-11ef-0423-b7a854d09bad
# ╟─e0d0f3a1-5e00-44f0-9c2b-4308cbd673ce
# ╟─f8c8013a-3e87-4d01-a3ae-86b39cf1f002
# ╟─26c59b52-d294-11ef-1eba-d3f235f85eee
# ╟─26c5a1f6-d294-11ef-3565-39d027843fbb
# ╟─26c5a93a-d294-11ef-23a1-cbcf0c370fc9
# ╟─26c5b896-d294-11ef-1d8e-0feb99d2d45b
# ╟─26c5c1ae-d294-11ef-15c6-13cae5bc0dc8
# ╟─26c5cfb4-d294-11ef-05bb-59d5e27cf37c
# ╟─c7351bf1-447e-475b-8965-d259c01bfd57
# ╟─3deadfd0-9fbb-476a-a7de-5dd694e55a65
# ╟─26c5d734-d294-11ef-20a3-afd2c3324323
# ╟─26c5f8d6-d294-11ef-3bcd-4d5e0391698d
# ╟─26c623f6-d294-11ef-13c0-19edd43592c0
# ╟─26c62ebe-d294-11ef-0cfb-ef186203e890
# ╟─26c6347c-d294-11ef-056f-7b78a9e22272
# ╟─26c64174-d294-11ef-2bbc-ab1a84532311
# ╟─26c65092-d294-11ef-39cc-1953a725f285
# ╟─f1f7407d-86a1-4f24-b78a-61a411d1f371
# ╟─26c67f04-d294-11ef-03a4-838ae255689d
# ╟─26c6e002-d294-11ef-15a4-33e30d0d76ec
# ╟─ae7ed1fc-fc36-4327-be55-a142477ca0ad
# ╟─de16b831-7afa-408f-83fa-99c6e24840f5
# ╟─e6aeee80-9e63-4937-9edf-428d5e3e38d3
# ╟─baec0494-9557-49d1-b4d8-a8030d3281b7
# ╟─40ce0abb-a086-4977-9131-10f60ab44152
# ╟─26c6f63c-d294-11ef-1090-e9238dd6ad3f
# ╟─aea77d69-9ecd-4be0-b6fd-c944d27d68df
# ╟─3654551d-5d08-4bb0-8a0d-c7d42225bc69
# ╟─edb179df-5cff-4e7b-8645-6da4818dceee
# ╟─757465a4-6a7f-4c8e-98de-6df5ca995b03
# ╟─26c704f6-d294-11ef-1b3d-d52f0fb1c81d
# ╟─26c728f0-d294-11ef-0c01-6143abe8c3f0
# ╟─06512595-bdb7-4adf-88ae-62af20210891
# ╟─26c73cf0-d294-11ef-297b-354eb9c71f57
# ╟─3e897a59-e7b5-492c-8a8a-724248513a72
# ╟─93e7c7d5-a940-4764-8784-07af2f056e49
# ╟─26c74c9a-d294-11ef-2d31-67bd57d56d7c
# ╟─26c75b5e-d294-11ef-173e-b3f46a1df536
# ╟─26c7696e-d294-11ef-25f2-dbc0946c0858
# ╠═c90176ea-918b-4643-a10f-cef277c5ea75
# ╟─cc547bfa-a130-4382-af47-73de56e4741b
# ╠═0349720e-5de4-4b39-babd-c0881588f1de
# ╠═8555aec9-4e80-49e7-8514-ef4a2236801b
# ╠═666680b2-315a-4d95-8f7f-3ae50018e112
# ╠═86c33a7c-135a-461f-a17e-b50bca418e13
# ╠═98a0ed70-a627-48d6-a1f8-3dec7aba2bb2
# ╠═55a1c42b-20d8-47a3-aa00-7af905db537c
# ╠═4ee377c2-a126-4c40-8053-517d40c5ef9d
# ╟─26c796c8-d294-11ef-25be-17dcd4a9d315
# ╠═0090be18-2453-4ad3-8e2c-6953649b171e
# ╟─f42a1a65-20ce-452f-9974-bc8146943574
# ╟─26c7b428-d294-11ef-150a-bb37e37f4b5d
# ╟─b3bb7349-1965-4734-83ed-ba6fef0ccc41
# ╟─06170e31-e865-4178-8af0-41d82df95d71
# ╟─bbdca8c2-022f-42be-bcf7-80d86f7f269c
# ╟─26c8068a-d294-11ef-3983-a1be55128b3f
# ╟─56bea391-b812-4fc4-8f27-fcb4cb984cf4
# ╟─5a94e2a4-7134-462e-9dc5-56083769049f
# ╟─747a7e1e-b921-4882-b00a-1b00bef8433d
# ╟─2d4adbf6-6de8-4e3a-ad6f-fa8bbfa5999e
# ╟─208ba1bb-a4bf-4b8c-93d2-0d6c6c8d16d4
# ╟─2f490e1f-e495-4f55-a3f8-60d6fd716d4e
# ╟─b91bc3b6-b815-4942-b297-c0e2b4b99654
# ╟─26c8160c-d294-11ef-2a74-6f7009a7c51e
# ╟─26c82f16-d294-11ef-0fe1-07326b56282f
# ╟─26c85a22-d294-11ef-3c8e-7b72a4313ced
# ╟─26c867d8-d294-11ef-2372-d75ed0bcc02d
# ╟─7a3c0ff7-0b32-4954-ae28-b644f4d966ef
# ╠═de049d59-9863-4bac-91c3-32851cad15d9
# ╟─26c8a2a4-d294-11ef-1cd3-850e877d7a25
# ╟─26c8b682-d294-11ef-1331-2bcf8baec73f
# ╟─26c8c7fa-d294-11ef-0444-6555ecf5c721
# ╟─26c8e172-d294-11ef-2a9e-89e0f4cbf475
# ╟─26c9121e-d294-11ef-18e6-ed8105503adc
# ╟─55570464-89c8-4d9b-b667-dfa64ac62294
# ╠═df171940-eb54-48e2-a2b8-1a8162cabf3e
# ╠═58bd0d43-743c-4745-b353-4a89b35e85ba
# ╠═489cbd24-1a69-4a00-a2e9-53c2c57cef65
# ╠═9d2068d7-db54-460e-930c-b7c3273162ee
# ╠═deba376e-59bd-4b07-814c-8f7937db52a5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
