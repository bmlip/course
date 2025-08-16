### A Pluto.jl notebook ###
# v0.20.13

#> [frontmatter]
#> image = "https://github.com/bmlip/course/blob/v2/assets/figures/Figure4.9.png?raw=true"
#> description = "Introduction to discriminative classification models and Bayesian logistic regression."
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

# ╔═╡ a759653c-0da4-40b7-9e9e-1e3d2e4df4ea
using Random, Plots, LaTeXStrings

# ╔═╡ 6a20aa94-e2fa-45ab-9889-62d44cbfc1ba
using Optim # Optimization library

# ╔═╡ ad196ae6-c65e-4aaa-b0cc-bd72daa41952
using MarkdownLiteral: @mdx

# ╔═╡ 616e84d7-063d-4d9d-99e4-56aecf3c7ee4
using Distributions

# ╔═╡ e379cc2a-43f8-432f-84fc-a88fd4f3ad0a
using PlutoUI, PlutoTeachingTools

# ╔═╡ 25eefb10-d294-11ef-0734-2daf18636e8e
md"""
# Discriminative Classification

"""

# ╔═╡ e7c45ff8-9fa2-4ea3-a06f-5769d877540e
PlutoUI.TableOfContents()

# ╔═╡ 25ef12bc-d294-11ef-1557-d98ba829a804
md"""
## Preliminaries

##### Goal 

  * Introduction to discriminative classification models

##### Materials        

  * Mandatory

      * These lecture notes
  * Optional

      * [Bishop PRML book](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) (2006), pp. 213 - 217 (Laplace approximation)
      * [Bishop PRML book](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) (2006), pp. 217 - 220 (Bayesian logistic regression)
      * [T. Minka (2005), Discriminative models, not discriminative training](https://github.com/bmlip/course/blob/main/assets/files/Minka-2005-Discriminative-models-not-discriminative-training.pdf)

"""

# ╔═╡ fe66a986-2f55-4417-a71d-b3b99f6369cc
section_outline("Challenge:", "difficult class-conditional data distributions" , color= "red" )

# ╔═╡ 25ef2806-d294-11ef-3cb6-0f3e76b9177e
md"""
Our task will be the same as in the preceding class on (generative) classification. But this time, the class-conditional data distributions look very non-Gaussian, yet the linear discriminative boundary looks easy enough:

"""

# ╔═╡ 4ceede48-a4d5-446b-bb34-26cec4af357a
begin
	N_bond = @bindname N Slider(9:200; default=120, show_value=true)
end

# ╔═╡ cf829697-6283-4d2f-b0dd-bbfbd689a145
md"""
### Implementation
"""

# ╔═╡ b5bc379c-5a0e-4b13-99fe-92632250a35e
md"""
Split X based on class label:
"""

# ╔═╡ ddb2dc0d-3ef9-4add-93e9-ee575aabd81a
X_test = [3.75; 1.0] # Features of 'new' data point

# ╔═╡ d1bbdc6a-e5ff-4cd6-9175-860b5ec04f3c
md"""
# Bayesian Logistic Regression
"""

# ╔═╡ 25ef6ece-d294-11ef-270a-999c8d457b24
md"""
## Framework

A data set is given by  ``D = \{(x_1,y_1),\dotsc,(x_N,y_N)\}`` with ``x_n \in \mathbb{R}^M`` and ``y_n \in \mathcal{C}_k``, with ``k=1,\ldots,K``.

"""

# ╔═╡ 25ef7f54-d294-11ef-3f05-0d85fe6e7a17
md"""
Sometimes, the precise assumptions of the (Gaussian-Categorical) generative model 

```math
p(x_n,y_n\in\mathcal{C}_k|\theta) =  \pi_k \cdot \mathcal{N}(x_n|\mu_k,\Sigma)
```

clearly do not match the data distribution.

"""

# ╔═╡ 25efa2fe-d294-11ef-172f-9bb09277f59e
md"""
Here's an **IDEA**! Let's model the posterior 

```math
p(y_n\in\mathcal{C}_k|x_n)
```

*directly*, without any assumptions on the class densities.

"""

# ╔═╡ 25efbe42-d294-11ef-3e4e-cfea366757da
md"""
Similarly to regression, we will assume that the inputs ``x`` are given, so we wil not add a model ``p(x)`` for input uncertainties.

"""

# ╔═╡ 25efd6b6-d294-11ef-3b21-6363ef531eb5
md"""
## Model Specification 
We will work this idea out for a 2-class problem. Assume a data set is given by  ``D = \{(x_1,y_1),\dotsc,(x_N,y_N)\}`` with ``x_n \in \mathbb{R}^M`` and ``y_n \in \{0,1\}``.

"""

# ╔═╡ 25f02ac6-d294-11ef-26c4-f142b8ac4b5f
md"""
What model should we use for the posterior distribution ``p(y_n \in \mathcal{C}_k|x_n)``?

"""

# ╔═╡ 25f0adde-d294-11ef-353e-4b4773df9ff5
md"""
#### Data-generating distribution

We will take inspiration from the [generative classification](https://bmlip.github.io/course/lectures/Generative%20Classification.html#softmax) approach, where we derived the class posterior 

```math
p(y_{nk} = 1\,|\,x_n,\beta_k,\gamma_k) = \sigma(\beta_k^T x_n + \gamma_k)
```

as a **softmax** function of a linear map of the input.  

Here, in logistic regression, we *choose* the 2-class softmax function (which is called the [**logistic** function](https://en.wikipedia.org/wiki/Logistic_function)) with linear discrimination bounderies for the posterior class probability:

```math
p(y_n =1 \,|\, x_n, w) = \sigma(w^T x_n) \,.
```

where 

```math
\sigma(a) = \frac{1}{1+e^{-a}}
```

is the *logistic* function.

Clearly, it follows from this assumption that ``p(y_n =0 \,|\, x_n, w) = 1- \sigma(w^T x_n)``.

"""

# ╔═╡ 7d8f376c-b43f-46f4-87bc-c4d7f94d60e1
TODO("can we make a mini on the softmax function and its special case, the logistic function? The graph below, including its approximation should be generated in Julia, not taken as an png from Bishop.")

# ╔═╡ 25f0f618-d294-11ef-0d94-bf80c8e2957b
md"""
![](https://github.com/bmlip/course/blob/v2/assets/figures/Figure4.9.png?raw=true)

(Bishop fig.4.9). The logistic function ``\sigma(a) = 1/(1+e^{-a})`` (red), together with the $(HTML("<span id='scaled-probit'>scaled probit function</span>")) ``\Phi(\lambda a)``, for ``\lambda^2=\pi/8`` (in blue). We will use this approximation later in the [Laplace approximation](https://bmlip.github.io/course/minis/Laplace%20Approximation.html#gaussian-cdf).

"""

# ╔═╡ 25f12528-d294-11ef-0c65-97c61935e9c2
md"""
Adding the other class (``y_n=0``) leads to the following posterior class distribution:

```math
\begin{align*}
p(y_n \,|\, x_n, w) &= \mathrm{Bernoulli}\left(y_n \,|\, \sigma(w^T x_n) \right) \\
&= \sigma(w^T x_n)^{y_n} \left(1 - \sigma(w^T x_n)\right)^{(1-y_n)} \tag{B-4.89} \\
  &= \sigma\left( (2y_n-1) w^T x_n\right)
\end{align*}
```

Note that for the 3rd equality, we have made use of the fact that ``\sigma(-a) = 1-\sigma(a)``.

Each of these three models in B-4.89 are **equivalent**. We mention all three notational options since they all appear in the literature.  

"""

# ╔═╡ 25f14226-d294-11ef-369f-e545d5fe2700
md"""
This choice for the class posterior is called **logistic regression**, in analogy to [linear regression](https://bmlip.github.io/course/lectures/Regression.html#likelihood-function):

```math
\begin{align}
p(y_n|x_n,w) &= \mathcal{N}(y_n|w^T x_n,\beta^{-1}) \tag{for linear regression} \\
p(y_n|x_n,w) &= \sigma\left( (2y_n-1) w^T x_n\right) \tag{for logistic regression}
\end{align}
```

"""

# ╔═╡ 25f14f82-d294-11ef-02fb-2dc632b8f118
md"""
In the discriminative approach, the parameters ``w`` are **not** structured into ``\{\mu,\Sigma,\pi \}``. In principle they are "free" parameters for which we can choose any value that seems appropriate. This provides discriminative approach with more flexibility than the generative approach. 

"""

# ╔═╡ 25f15e0a-d294-11ef-3737-79a68c9b3c61
md"""
#### Prior

In *Bayesian* logistic regression, we often add a **Gaussian prior on the weights**: 

```math
\begin{align*}
p(w) = \mathcal{N}(w \,|\, m_0, S_0) \tag{B-4.140}
\end{align*}
```
"""

# ╔═╡ 25f19ed8-d294-11ef-3298-efa16dda1dde
md"""
## Parameter Inference

Note that for generative classification, for the sake of simplicity, we used maximum likelihood estimation for the model parameters. We could have used Bayesian parameter estimation for the generative classification model but the math is not suited for an introductory lesson. 

In this lesson on discriminative classification, we specify both a prior and likelihood function for the parameters ``w``, which allows us to compute a Bayesian posterior for the weights. 

As before, once the model is specified, everything else follows directly from the rules of probability theory.


"""

# ╔═╡ 25f1390c-d294-11ef-364d-17e4c93b9a57
md"""
For the data set ``D = \{(x_1,y_1),\dotsc,(x_N,y_N)\}``, the **likelihood function** for the parameters ``w`` is given by

```math
p(D|w) = \prod_{n=1}^N p(y_n|x_n,w) = \prod_{n=1}^N \sigma\left( (2y_n-1) w^T x_n\right)
```

"""

# ╔═╡ bda07a2e-3769-4ffe-9bc5-2b8a515247f6
md"""


The posterior for the weights follows by Bayes rule,

```math
\begin{align*}
\underbrace{p(w \,|\, D)}_{\text{posterior}} &=  \frac{p(w) p(D|w)}{\int p(w) p(D|w) \mathrm{d}w} \\ &= \frac{\overbrace{\mathcal{N}(w \,|\, m_0, S_0)}^{\text{prior}} \cdot \overbrace{\prod_{n=1}^N \sigma\left( (2y_n-1) w^T x_n\right)}^{\text{likelihood}}}{\underbrace{\int \mathcal{N}(w \,|\, m_0, S_0) \prod_{n=1}^N \sigma\left( (2y_n-1) w^T x_n\right) \mathrm{d}w}_{\text{evidence}}} \tag{B-4.142}
\end{align*}
```

In principle, Bayesian learning of the parameters is done now! 

Unfortunately, the posterior ``p(w | D)`` is not Gaussian, and the evidence ``p(D)`` is also not analytically computable. (We will deal with this later).
"""

# ╔═╡ 25f1ab08-d294-11ef-32ed-493792e121b7
md"""
## Application: the predictive distribution

For a new data point ``x_\bullet``, the predictive distribution for ``y_\bullet=1`` is given by 

```math
\begin{align*}
p(y_\bullet = 1 | x_\bullet, D) &= \int p(y_\bullet = 1 \,|\, x_\bullet, w) \, p(w| D) \,\mathrm{d}w \\
  &= \int \sigma(w^T x_\bullet) \, p(w| D) \,\mathrm{d}w \tag{B-4.145}
\end{align*}
```

"""

# ╔═╡ 25f1b404-d294-11ef-1c3a-a5a8142bb202
md"""
While Eq. B-4.145 gives the expression for the Bayesian predictive class distribution, the integral becomes analytically intractable when we substitute the posterior distribution over weights, ``p(w | D)`` (from Eq. B-4.142), into it :(

"""

# ╔═╡ 25f1c2a0-d294-11ef-009c-69b64e87e5fb
md"""
Many methods have been developed to approximate these types of Bayesian integrals. Here, we present the **Laplace approximation**, which is one of the simplest methods with broad applicability to Bayesian calculations.

"""

# ╔═╡ 3422dd29-6da9-4e0f-a4ab-646f223c2244
md"""
## Working out Numerics with Laplace Approximation
"""

# ╔═╡ 8b0bb225-bdc1-45ec-bd34-68d674d6f08d
md"""
The **Laplace Approximation** approximates a function by a Gaussian-shaped function. In this case, we will approximate the weight posterior ``p(w|D)`` by a Gaussian distribution

```math
q(w) = \mathcal{N}\left(w\,|\, w_{N}, S_N\right) \tag{B-4.144}
```

with

```math
\begin{align}
w_N &= \arg\max_w \log p(w|D) \\
S_N^{-1} &= S_0^{-1} + \sum_n \sigma_n (1-\sigma_n) x_n x_n^T \tag{B-4.143}
\end{align}
```
where we used short-hand ``\sigma_n = \sigma\left((2y_n-1) w^T x_n\right)``.

If we substitute the Gaussian approximation from Eq. B-4.143 into the expression for the predictive class distribution (Eq. B-4.145), we obtain (after some additional approximations):

```math
\begin{align*}
p(y_\bullet = 1 \mid x_\bullet, D) &= \int p(y_\bullet = 1 \,|\, x_\bullet, w) \cdot p(w\,|\, D) \,\mathrm{d}w \\
  &\approx  \int \sigma(w^T x_\bullet) \cdot \mathcal{N}\left(w \,|\, w_N, S_N\right) \,\mathrm{d}w \tag{B-4.145} \\
&\approx \Phi\left( \frac{\mu_a}{\sqrt(\lambda^{-2} +\sigma_a^2)}\right) \tag{B-4.152}
\end{align*}
```

where 

```math
\begin{align}
\lambda^2 &= \pi / 8 \\
\mu_a  &= w^T_{N} x_\bullet \tag{B-4.149} \\
\sigma_a^2 &= x^T_\bullet S_N x_\bullet \tag{B-4.150}
\end{align}
```
and ``\Phi(x)= \frac{1}{\sqrt(2\pi)}\int_{-\infty}^{x}e^{-t^2/2}\mathrm{d}t`` is the Gaussian cumulative distribution function (CDF) . The Gaussian CDF closely approximates the logistic sigmoid function, with 
``\Phi(\sqrt{\pi/8} a) \approx \sigma(a)``. 


"""

# ╔═╡ ae2b23f0-853e-4237-aab2-81c961f52cf6
md"""
Although the intermediate equations may look intimidating, the final result for the predictive distribution Eq. B-4.152 has a simple closed-form expression.

"""

# ╔═╡ e4cc517b-d3b5-4517-a28b-efb8aba24496
md"""
The numerical issues associated with the Laplace approximation and the evaluation of the predictive class distribution are discussed in detail in the following mini-lecture.
"""

# ╔═╡ 33b859f2-9ea8-4f8b-b0f8-08a19c6a96fc
NotebookCard("https://bmlip.github.io/course/minis/Laplace%20Approximation.html")

# ╔═╡ 38b4854f-be02-4696-802f-2106481e3aea
md"""
## Bayesian Processing of Uncertainties

We now make an important observation: According to Eq. B-4.143, the posterior covariance matrix ``S_N`` of the weight vector depends on both the prior variance ``S_0`` and the distribution of the training data ``\{(x_n, y_n)\}_{n=1}^N``. In regions with limited training data and/or an uninformative prior (i.e., large ``S_0``), the posterior uncertainty about the weights remains high. This increased uncertainty raises ``\sigma_a^2`` in Eq. B-4.150, **causing the posterior class probability in Eq. B-4.152 to approach ``0.5``** (since ``\Phi(0) = 0.5``, see [Gaussian CDF image](https://en.wikipedia.org/wiki/Normal_distribution#/media/File:Normal_Distribution_CDF.svg)), thereby reflecting greater uncertainty in the prediction.

In other words, if you draw a new feature ``x_\bullet`` from a region with little training data, then the predictive class probability ``p(y_\bullet | x_\bullet, D)`` naturally tends toward ``0.5``, a built-in expression of uncertainty, courtesy of the Bayesian framework.

In contrast, if you eliminate uncertainty by representing the weights as fixed-point estimates (i.e., as mere numbers), then the model becomes overconfident. It will still produce sharp predictions, even in regions where it has seen no data, exactly when it should be most uncertain. 

"""

# ╔═╡ 0045e569-dc3c-4998-86da-9d96f599c599
md"""
# Maximum Likelihood Estimation

"""

# ╔═╡ 25f365e2-d294-11ef-300e-9914333b1233
md"""

## MLE Parameter Estimation

Rather than the computationally involved Laplace approximation, in practice, discriminative classification is often executed through maximum likelihood estimation. 

With the usual 1-of-K encoding scheme for classes,

```math 
y_{nk} = \begin{cases} 1 & \text{if } x_n \in \mathcal{C}_k \\ 
  0 & \text{otherwise}  \,,\end{cases}
```

the log-likelihood for a ``K``-dimensional discriminative classifier evaluates to

```math
\begin{align*}
    \mathrm{L}(w) &= \log \prod_n \prod_k {p(\mathcal{C}_k|x_n,w)}^{y_{nk}} \\
    &= \sum_n \sum_k y_{kn} \log \bigg( \underbrace{\frac{e^{w_k^T x_n}}{ \sum_j e^{w_j^T x_n}}}_{=\text{softmax}(w_k^T x_n)} \bigg)
     \end{align*}
```

The gradient ``\nabla_{w_k} \mathrm{L}(w)`` to the weight  ``w_k`` can be worked out to
```math
\nabla_{w_k} \mathrm{L}(w) = \sum_n \underbrace{\big( \underbrace{y_{nk}}_{\text{target}} - \underbrace{\frac{e^{w_k^T x_n}}{ \sum_j e^{w_j^T x_n}}}_{\text{prediction}} \big)}_{\text{prediction error}}\cdot x_n  .
```

"""

# ╔═╡ 3b24b142-2239-4951-9177-ff87b5da4b68
details("Click for proof", 
	   md"""
The Log-likelihood is 

```math
\mathrm{L}(w) = \log \prod_n \prod_k {\underbrace{p(y_{nk}=1|x_n,w)}_{p_{nk}}}^{y_{nk}} = \sum_{n,k} y_{nk} \log p_{nk}
```

Use the fact that the softmax ``\phi_k \equiv e^{a_k} / {\sum_j e^{a_j}}`` has analytical derivative,

```math
 \begin{align*}
 \frac{\partial \phi_k}{\partial a_j} &= \frac{(\sum_j e^{a_j})e^{a_k}\delta_{kj}-e^{a_j}e^{a_k}}{(\sum_j e^{a_j})^2} = \frac{e^{a_k}}{\sum_j e^{a_j}}\delta_{kj} - \frac{e^{a_j}}{\sum_j e^{a_j}} \frac{e^{a_k}}{\sum_j e^{a_j}}\\
     &= \phi_k \cdot(\delta_{kj}-\phi_j) \,.
 \end{align*}
```

Take the derivative of ``\mathrm{L}(w)`` (or: how to spend an hour ...)

```math
\begin{align*} 
\nabla_{w_j} \mathrm{L}(w) &= \sum_{n,k} \frac{\partial \mathrm{L}_{nk}}{\partial p_{nk}} \cdot\frac{\partial p_{nk}}{\partial a_{nj}}\cdot\frac{\partial a_{nj}}{\partial w_j} \\
  &= \sum_{n,k} \frac{y_{nk}}{p_{nk}} \cdot p_{nk} (\delta_{kj}-p_{nj}) \cdot x_n \\
  &= \sum_n \Big( y_{nj} (1-p_{nj}) -\sum_{k\neq j} y_{nk} p_{nj} \Big) \cdot x_n \\
  &= \sum_n \left( y_{nj} - p_{nj} \right)\cdot x_n \\
  &= \sum_n \Big( \underbrace{y_{nj}}_{\text{target}} - \underbrace{\frac{e^{w_j^T x_n}}{\sum_{j^\prime} e^{w_{j^\prime}^T x_n}}}_{\text{prediction}} \Big)\cdot x_n 
\end{align*}
```

		
		""")

# ╔═╡ ff31d8c1-db35-4c85-a609-67fc40e9e78d
md"""

The parameter vector ``w`` for logistic regression can then be estimated through iterative gradient-based adaptation. For instance, start with a random weight ``\hat{w} = w_0``, and iterate through

```math
\hat{w}^{(i+1)} =  \hat{w}^{(i)} + \eta \cdot \left. \nabla_w   \mathrm{L}(w)  \right|_{w = \hat{w}^{(i)}}
```
until convergence. 

"""

# ╔═╡ 7932fff4-0568-49de-b34c-711e51487ae3
section_outline("Challenge Revisited:", "Bayesian Logistic Regression for Difficult Class-conditional Data Distributions" , color= "green" )

# ╔═╡ 25f3bef2-d294-11ef-1438-e9f7e469336f
md"""

Let us perform ML estimation of ``w`` on the data set from the introduction. To allow an offset in the discrimination boundary, we add a constant 1 to the feature vector ``x``. We only have to specify the (negative) log-likelihood and the gradient w.r.t. ``w``. Then, we use an off-the-shelf optimisation library to minimize the negative log-likelihood.

We plot the resulting maximum likelihood discrimination boundary. For comparison we also plot the ML discrimination boundary obtained from the [code example in the generative Gaussian classifier lesson](https://bmlip.github.io/course/lectures/Generative%20Classification.html#code-generative-classification-example).

"""

# ╔═╡ aaf764da-cf1b-4bc7-83ea-6d25a80ca3ab
N_bond

# ╔═╡ 25f3ee5e-d294-11ef-1fb4-e9d84b1e1ec6
md"""
The generative model gives a bad result because the feature distribution of one class is clearly non-Gaussian: the model does not fit the data well. 

The discriminative approach does not suffer from this problem because it makes no assumptions about the feature distribution ``p(x)``. Rather, it just estimates the conditional class distribution ``p(y|x)`` directly.

"""

# ╔═╡ 1f2bfcf4-fef4-4612-8683-d5c86a326eef
md"""
# Closing Thoughts
"""

# ╔═╡ 25f3ff84-d294-11ef-0031-63b23d23324d
md"""
## Why be Bayesian?

Why should you embrace the Bayesian approach to logistic regression? After all, Maximum Likelihood for logistic regression seems simpler.

Still, consider the following:

  * Bayesian logistic regression with the Laplace approximation ultimately leads to very simple analytic rules. Moreover, modern probabilistic programming languages and packages are able to automate the above inference derivations. (We just do them here to gain insight into a difficult inference process.)

  * Bayesian logistic regression allows for the computation of model evidence, enabling principled comparison of model performance across alternative models.

  * Perhaps most importantly, Bayesian logistic regression processes uncertainties, e.g., in places where almost no data is observed, the posterior class probability will pull back to the prior class probability rather than predicting some arbitrary probability.

"""

# ╔═╡ 25f41118-d294-11ef-13a8-3fa6587c1bf3
@mdx """
## Recap Classification

Let us recapitulate the differences between the generative and discriminative approaches to classification in a table:


<table> <tr> <td></td><td style="text-align:center"><b>Generative</b></td> <td style="text-align:center"><b>Discriminative (ML)</b></td> </tr> 

<tr> <td>1</td><td>Like <b>density estimation</b>, model joint prob.

```math
p(\\mathcal{C}_k) p(x|\\mathcal{C}_k) = \\pi_k \\mathcal{N}(\\mu_k,\\Sigma)
```

</td> <td>Like (linear) <b>regression</b>, model conditional

```math
p(\\mathcal{C}_k|x,\\theta)
```

</td> </tr>

<tr> <td>2</td><td>Leads to <b>softmax</b> posterior class probability

```math
 p(\\mathcal{C}_k|x,\\theta ) = e^{\\theta_k^T x}/Z
```

with <b>structured</b> ``\\theta``</td> <td> <b>Choose</b> also softmax posterior class probability

```math
 p(\\mathcal{C}_k|x,\\theta ) = e^{\\theta_k^T x}/Z
```

but now with 'free' ``\\theta``</td> </tr>

<tr> <td>3</td><td>

For Gaussian ``p(x|\\mathcal{C}_k)`` and multinomial priors,

```math
\\hat \\theta_k  = \\left[ {\\begin{array}{c}
   { - \\frac{1}{2} \\mu_k^T \\sigma^{-1} \\mu_k  + \\log \\pi_k}  \\\\
   {\\sigma^{-1} \\mu_k }  \\\\
\\end{array}} \\right]
```

<b>in one shot</b>.</td> <td>Find ``\\hat\\theta_k`` through gradient-based adaptation

```math
\\nabla_{\\theta_k}\\mathrm{L}(\\theta) = \\sum_n \\Big( y_{nk} - \\frac{e^{\\theta_k^T x_n}}{\\sum_{k^\\prime} e^{\\theta_{k^\\prime}^T x_n}} \\Big)\\, x_n
```

</td> </tr> </table>

"""

# ╔═╡ 25f19230-d294-11ef-2dfd-6d4927e86f57
md"""
## Discriminative Training or Discriminative Models?

In this lecture series, we presented two approaches to classification, namely the generative and the discriminative approach.  

While the discriminative approach is intuitive and effective for many tasks, it sits somewhat uncomfortably with the [Bayesian modeling approach](https://bmlip.github.io/course/lectures/Bayesian%20Machine%20Learning.html#The-Bayesian-Modeling-Approach) outlined in the Bayesian Machine Learning lecture. Specifically, the discriminative approach does not define a full joint model over all variables in the system. Instead, it focuses only on modeling the conditional distribution ``p(y | x)``, effectively ignoring the input distribution ``p(x)`` that would normally be part of a fully generative Bayesian model.

In a short paper by [T. Minka (2005)](https://github.com/bmlip/course/blob/main/assets/files/Minka-2005-Discriminative-models-not-discriminative-training.pdf), the model assumptions underlying discriminative classification are reinterpreted as arising from a special case of a generative model. This effectively restores discriminative approaches as fully compatible with the Bayesian modeling framework. (Note: the Minka paper is not required reading for the exam.)

"""

# ╔═╡ a00c545c-2274-4086-94ca-319d1436fa26
md"""
# Exercises
"""

# ╔═╡ b94644f8-725d-49bf-9641-3dad8b647f45
md"""

#### Discrimination boundaries (*)

Show that for logistic regression with ``p(y_n =1 \,|\, x_n, w) = \sigma(w^T x_n)``, the discrimination boundary, which can be computed by

```math
\frac{p(y_n\in\mathcal{C}_1|x_n)}{p(y_n\in\mathcal{C}_0|x_n)} \overset{!}{=} 1
```

is a straight line.
"""

# ╔═╡ 9554ed0b-69dd-443c-9538-03a4117eeb78
details("Click for solution",
md"""
		
```math
\begin{align}
\frac{ p(y_n \in \mathcal{C}_1 |x_n) }{ p(y_n\in\mathcal{C}_0|x_n) } &= \frac{ \sigma(w^T x_n)}{1 - \sigma(w^T x_n)} \\  
&= \frac{ \frac{1}{1+\exp(-w^T x_n)} }{ 1 -\frac{1}{1+\exp(-w^T x_n)} } \\  
&= \frac{1}{1+\exp(-w^T x_n) - 1}   \\   
&= \exp(w^T x) 
\end{align}
```	
Setting ``\exp(w^T x) \overset{!}{=} 1``, leads to 
```math
		w^T x = 0
```
for the discrimination boundary, which is a line. 
		""")

# ╔═╡ 6eee35ee-fd55-498f-9441-f18c2508de19
md"""
# Appendix
"""

# ╔═╡ 1128cb07-68c8-4b80-8fb4-ee9fcc76c050
md"""
## Behind the scenes
"""

# ╔═╡ fcec3c3a-8b0b-4dfd-b010-66abbf330069
function generate_dataset(N::Int64)
	Random.seed!(1234)
    # Generate dataset {(x1,y1),...,(xN,yN)}
    # x is a 2d feature vector [x1;x2]
    # y ∈ {false,true} is a binary class label
    # p(x|y) is multi-modal (mixture of uniform and Gaussian distributions)
	# srand(123)
    X = Matrix{Float64}(undef,2,N); y = Vector{Bool}(undef,N)
    for n=1:N
        if (y[n]=(rand()>0.6)) # p(y=true) = 0.6
            # Sample class 1 conditional distribution
            if rand()<0.5
                X[:,n] = [6.0; 0.5] .* rand(2) .+ [3.0; 6.0]
            else
                X[:,n] = sqrt(0.5) * randn(2) .+ [5.5, 0.0]
            end
        else
            # Sample class 2 conditional distribution
            X[:,n] = randn(2) .+ [1., 4.]
        end
    end

    return (X, y)
end

# ╔═╡ e3474a09-11ec-43e8-900f-f4fb31283f46
X, y = generate_dataset(N) # Generate data set, collect in matrix X and vector y

# ╔═╡ 48c200f1-9363-4ae4-ab9c-b000071aa9d6
X_c1 = X[:,findall(.!y)]'

# ╔═╡ 47ee9e9f-427c-42e1-867a-b6d2c9438d76
X_c2 = X[:,findall(y)]'

# ╔═╡ a65ca01a-0e9a-42cb-b1d7-648102a77eb5
function plot_dataset()
    result = scatter(X_c1[:,1], X_c1[:,2],markersize=4, label=L"y=0", xlabel=L"x_1", ylabel=L"x_2", xlims=(-1.6, 9), ylims=(-2, 7))
    scatter!(X_c2[:,1], X_c2[:,2],markersize=4, label=L"y=1")
    scatter!([X_test[1]], [X_test[2]], markersize=7, marker=:star, label=L"y=?") 
	plot!(legend=:bottomright)
    return result  
end

# ╔═╡ d29ccc9e-d4a6-46ae-b907-2bc68c8d99bc
	plot_dataset()

# ╔═╡ 56598859-2824-4242-a894-684bf1ad1f6e
y_1 = ifelse.(y, 1.0, 0.0)

# ╔═╡ 6f483978-29f0-4165-bd8f-650c403e3512
# Extend X with a row of ones to allow an offset in the discrimination boundary
X_ext = vcat(X, ones(1, length(y)))

# ╔═╡ a89af0df-c39b-406e-a30a-4706ad2ea043
function negative_log_likelihood(θ::Vector)
	# P(C1|X,θ)
    p_1 = 1.0 ./ (1.0 .+ exp.(-X_ext' * θ))
	
	# negative log-likelihood
    return -sum(log.( (y_1 .* p_1) + ((1 .- y_1).*(1 .- p_1))) ) 
end

# ╔═╡ a75d69e1-c1e9-45b4-9924-4c2fe59413dc
# Use Optim.jl optimiser to minimize the negative log-likelihood function w.r.t. θ
θ = let
	results = optimize(negative_log_likelihood, zeros(3), LBFGS())
	results.minimizer
end

# ╔═╡ 00488cbb-75c6-4df9-9924-fada8f79a6f1
function build_generative_discrimination_boundary(X::Matrix, y::Vector{Bool})
    # Generate discrimination boundary function x[2] = boundary(x[1]) for a Gaussian generative model:
    # X = [x_1,...,x_N]
    # y = [y_1;...;y_N]
    # x is a 2-d real (feature) vector
    # y ∈ {false,true}
    # x|y ~ 𝓝(x|μ_y, Σ_y)
    # We find the class-conditional Gaussian distributions by MLE
    # See lesson (generative classification) for more details
    (size(X,1)==2) || error("The columns of X should have length 2")

    # MLE of p(y)
    p_1_est = sum(y.==true) / length(y)
    π_hat = [p_1_est; 1 .- p_1_est]

    # MLE of class-conditional multivariate Gaussian densities
    X_cls1 = X[:,y.==true]
    X_cls2 = X[:,y.==false]
    d1 = fit_mle(FullNormal, X_cls1)  # MLE density estimation d1 = N(μ₁, Σ₁)
    d2 = fit_mle(FullNormal, X_cls2)  # MLE density estimation d2 = N(μ₂, Σ₂)
    Σ = π_hat[1] * cov(d1) + π_hat[2] * cov(d2) # Combine Σ₁ and Σ₂ into Σ

    conditionals = [MvNormal(mean(d1), Σ); MvNormal(mean(d2), Σ)] # p(x|C)

    # Discrimination boundary of the posterior (p(apple|x;D) = p(peach|x;D) = 0.5)
    β(k) = inv(Σ)* mean(conditionals[k])
    γ(k) = -0.5 * mean(conditionals[k])' * inv(Σ) * mean(conditionals[k]) + log(π_hat[k])
    function discriminant_x2(x1)
        # Solve discriminant equation for x2
        
        β12 = β(1) .- β(2)
        γ12 = (γ(1) .- γ(2))[1,1]
        return -1 ./ β12[2]*(β12[1]*x1 .+ γ12) 
    end

    return discriminant_x2
end

# ╔═╡ 7ad2f815-9d19-448c-bb7e-044a955f82e0
let
	# Plot the data set and ML discrimination boundary
	plot_dataset()
	disc_boundary(x1) = -1 / θ[2] * (θ[1]*x1 + θ[3])
	generative_boundary = build_generative_discrimination_boundary(X, y)
	
	
	p_1(x) = 1.0 / (1.0 + exp(-([x;1.]' * θ)))
	x_test = [3.75;1.0]
	@debug("P(C1|x•,θ) = $(p_1(x_test))")
	
	plot!([-2., 10.], disc_boundary; label="Discr. boundary", linewidth=2)
	plot!([-2.,10.], generative_boundary; label="Gen. boundary", linewidth=2)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
MarkdownLiteral = "736d6165-7244-6769-4267-6b50796e6954"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Distributions = "~0.25.120"
LaTeXStrings = "~1.4.0"
MarkdownLiteral = "~0.1.2"
Optim = "~1.13.2"
Plots = "~1.40.17"
PlutoTeachingTools = "~0.4.4"
PlutoUI = "~0.7.69"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "3af9d070de921b89310e964c2b511252864a26e8"

[[deps.ADTypes]]
git-tree-sha1 = "60665b326b75db6517939d0e1875850bc4a54368"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.17.0"

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

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

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

[[deps.CommonMark]]
deps = ["PrecompileTools"]
git-tree-sha1 = "351d6f4eaf273b753001b2de4dffb8279b100769"
uuid = "a80b9123-70ca-4bc0-993e-6e3bcb318db6"
version = "0.9.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

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

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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
git-tree-sha1 = "76b3b7c3925d943edf158ddb7f693ba54eb297a5"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.0"

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
git-tree-sha1 = "53970db0989d231937cd96c0b0ace67f38da274e"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.5"

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

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "31fd32af86234b6b71add76229d53129aa1b87a9"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.28.1"

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
git-tree-sha1 = "910febccb28d493032495b7009dce7d7f7aee554"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.0.1"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

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

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "4adee99b7262ad2a1a4bbbc59d993d24e55ea96f"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.4.0"

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

[[deps.MarkdownLiteral]]
deps = ["CommonMark", "HypertextLiteral"]
git-tree-sha1 = "f7d73634acd573bf3489df1ee0d270a5d6d3a7a3"
uuid = "736d6165-7244-6769-4267-6b50796e6954"
version = "0.1.2"

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
git-tree-sha1 = "fcfec547342405c7a8529ea896f98c0ffcc4931d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.70"

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

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

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

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

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
# ╟─25eefb10-d294-11ef-0734-2daf18636e8e
# ╟─e7c45ff8-9fa2-4ea3-a06f-5769d877540e
# ╟─25ef12bc-d294-11ef-1557-d98ba829a804
# ╟─fe66a986-2f55-4417-a71d-b3b99f6369cc
# ╟─25ef2806-d294-11ef-3cb6-0f3e76b9177e
# ╟─4ceede48-a4d5-446b-bb34-26cec4af357a
# ╟─d29ccc9e-d4a6-46ae-b907-2bc68c8d99bc
# ╟─cf829697-6283-4d2f-b0dd-bbfbd689a145
# ╟─a65ca01a-0e9a-42cb-b1d7-648102a77eb5
# ╠═e3474a09-11ec-43e8-900f-f4fb31283f46
# ╟─b5bc379c-5a0e-4b13-99fe-92632250a35e
# ╠═48c200f1-9363-4ae4-ab9c-b000071aa9d6
# ╠═47ee9e9f-427c-42e1-867a-b6d2c9438d76
# ╠═ddb2dc0d-3ef9-4add-93e9-ee575aabd81a
# ╠═a759653c-0da4-40b7-9e9e-1e3d2e4df4ea
# ╟─d1bbdc6a-e5ff-4cd6-9175-860b5ec04f3c
# ╟─25ef6ece-d294-11ef-270a-999c8d457b24
# ╟─25ef7f54-d294-11ef-3f05-0d85fe6e7a17
# ╟─25efa2fe-d294-11ef-172f-9bb09277f59e
# ╟─25efbe42-d294-11ef-3e4e-cfea366757da
# ╟─25efd6b6-d294-11ef-3b21-6363ef531eb5
# ╟─25f02ac6-d294-11ef-26c4-f142b8ac4b5f
# ╟─25f0adde-d294-11ef-353e-4b4773df9ff5
# ╟─7d8f376c-b43f-46f4-87bc-c4d7f94d60e1
# ╟─25f0f618-d294-11ef-0d94-bf80c8e2957b
# ╟─25f12528-d294-11ef-0c65-97c61935e9c2
# ╟─25f14226-d294-11ef-369f-e545d5fe2700
# ╟─25f14f82-d294-11ef-02fb-2dc632b8f118
# ╟─25f15e0a-d294-11ef-3737-79a68c9b3c61
# ╟─25f19ed8-d294-11ef-3298-efa16dda1dde
# ╟─25f1390c-d294-11ef-364d-17e4c93b9a57
# ╟─bda07a2e-3769-4ffe-9bc5-2b8a515247f6
# ╟─25f1ab08-d294-11ef-32ed-493792e121b7
# ╟─25f1b404-d294-11ef-1c3a-a5a8142bb202
# ╟─25f1c2a0-d294-11ef-009c-69b64e87e5fb
# ╟─3422dd29-6da9-4e0f-a4ab-646f223c2244
# ╟─8b0bb225-bdc1-45ec-bd34-68d674d6f08d
# ╟─ae2b23f0-853e-4237-aab2-81c961f52cf6
# ╟─e4cc517b-d3b5-4517-a28b-efb8aba24496
# ╟─33b859f2-9ea8-4f8b-b0f8-08a19c6a96fc
# ╟─38b4854f-be02-4696-802f-2106481e3aea
# ╟─0045e569-dc3c-4998-86da-9d96f599c599
# ╟─25f365e2-d294-11ef-300e-9914333b1233
# ╟─3b24b142-2239-4951-9177-ff87b5da4b68
# ╟─ff31d8c1-db35-4c85-a609-67fc40e9e78d
# ╟─7932fff4-0568-49de-b34c-711e51487ae3
# ╟─25f3bef2-d294-11ef-1438-e9f7e469336f
# ╠═aaf764da-cf1b-4bc7-83ea-6d25a80ca3ab
# ╟─7ad2f815-9d19-448c-bb7e-044a955f82e0
# ╠═56598859-2824-4242-a894-684bf1ad1f6e
# ╠═6f483978-29f0-4165-bd8f-650c403e3512
# ╠═a89af0df-c39b-406e-a30a-4706ad2ea043
# ╠═a75d69e1-c1e9-45b4-9924-4c2fe59413dc
# ╟─25f3ee5e-d294-11ef-1fb4-e9d84b1e1ec6
# ╠═6a20aa94-e2fa-45ab-9889-62d44cbfc1ba
# ╟─1f2bfcf4-fef4-4612-8683-d5c86a326eef
# ╟─25f3ff84-d294-11ef-0031-63b23d23324d
# ╟─25f41118-d294-11ef-13a8-3fa6587c1bf3
# ╟─25f19230-d294-11ef-2dfd-6d4927e86f57
# ╟─a00c545c-2274-4086-94ca-319d1436fa26
# ╟─b94644f8-725d-49bf-9641-3dad8b647f45
# ╟─9554ed0b-69dd-443c-9538-03a4117eeb78
# ╟─6eee35ee-fd55-498f-9441-f18c2508de19
# ╟─1128cb07-68c8-4b80-8fb4-ee9fcc76c050
# ╠═ad196ae6-c65e-4aaa-b0cc-bd72daa41952
# ╠═616e84d7-063d-4d9d-99e4-56aecf3c7ee4
# ╠═fcec3c3a-8b0b-4dfd-b010-66abbf330069
# ╠═00488cbb-75c6-4df9-9924-fada8f79a6f1
# ╠═e379cc2a-43f8-432f-84fc-a88fd4f3ad0a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
