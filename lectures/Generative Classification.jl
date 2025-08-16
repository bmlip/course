### A Pluto.jl notebook ###
# v0.20.13

#> [frontmatter]
#> description = "Can you teach a computer to tell apples from peaches? Discover generative classification!"
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

# ╔═╡ f1a40378-a27c-4aa0-a62c-600ffde0032f
using PlutoUI, PlutoTeachingTools

# ╔═╡ 1b304964-6833-4cae-b84e-a5073f9586cd
using Markdown

# ╔═╡ 05ccf8cf-0711-4751-b378-5b0953eeedd0
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).

# ╔═╡ 6631c0e4-4941-442e-8dd4-fa307ee7a8c0
using Random

# ╔═╡ f1575443-c9fb-4674-bbce-bf3a5a6d5a8d
using Plots, Distributions, HypertextLiteral

# ╔═╡ 23c689fc-d294-11ef-086e-47c4f871bed2
md"""
# Generative Classification

"""

# ╔═╡ fe9d4fbc-f264-459b-8fbe-26663500f6c5
PlutoUI.TableOfContents()

# ╔═╡ 23c6997e-d294-11ef-09a8-a50563e5975b
md"""
## Preliminaries

##### Goal 

  * Introduction to linear generative classification with a Gaussian-categorical generative model

##### Materials        

  * Mandatory

      * These lecture notes
  * Optional

      * [Bishop PRML book](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) (2006),  pp. 196-202 (section 4.2 focuses on binary classification, whereas in these lecture notes we describe generative classification for multiple classes).

"""

# ╔═╡ f7a19975-a919-4659-9b6a-d8963a1cd6d9
section_outline("Challenge:", "Apple or Peach?" , color= "red" )

# ╔═╡ 51a46b5e-0c35-4841-a4f3-413d5d294805
md"""

You're given the numerical values for two features (let's say, _sugar content_ and _acidity_) of a bunch of fruits. Each piece of fruit is either an apple or a peach. Generate these data yourself by selecting the total number of fruits with the slider.

"""

# ╔═╡ 876f47d8-b272-4e23-b5ec-5c7d615ff618
begin
	N_bond = @bindname N Slider(1:250; show_value=true, default=50)
end

# ╔═╡ e774041a-672d-40f3-ac8f-fc5dbf1bfc59
md"""
In the scatter plot, the two features are represented along the two ``x``-coordinates, while the fruit label ``y \in \{\text{apple}, \text{peach}\}`` is encoded by the marker style.

You are also given a test fruit, shown as a yellow marker, which has known feature values but an unknown fruit label. 

##### problem

  - Based on the observed data, what is the probability that the test fruit is an apple? 

##### solution

  - Later in this lecture.
"""

# ╔═╡ 5730758d-80cd-4d95-b16c-399c38cf585b
md"""
# Bayesian Generative Classification
"""

# ╔═╡ 23c73302-d294-11ef-0c12-571686b202a9
md"""
The plan for generative classification is as follows:
We begin by constructing a model for the joint distribution 

```math
p(x, y) = p(x | y)\, p(y),
```
which combines a **class-conditional likelihood** ``p(x | y)`` with a **prior** ``p(y)`` over classes.
Then, we apply Bayes rule to compute the posterior class probabilities,

```math
p(y|x) = \frac{p(x|y) p(y)}{\sum_{y^\prime} p(x|y^\prime) p(y^\prime)} \propto p(x|y)\,p(y)
```
This posterior can then be used for classification by selecting the class with the highest probability.

Next, we discuss the three modeling stages: (1) model specification, (2) parameter learning, (3) application (classification).

"""

# ╔═╡ 23c73b54-d294-11ef-0ef8-8d9159139a1b
md"""
## Model Specification

#### Representation

The above data set will be represented as  ``D = \{(x_1,y_1),\dotsc,(x_N,y_N)\}``, where 
  * inputs ``x_n \in \mathbb{R}^M`` are called **features**.
  * outputs ``y_n \in \mathcal{C}_k``, with ``k=1,\ldots,K``; The **discrete** targets ``\mathcal{C}_k`` are called **classes**.

Similar to our representation of the categorical distribution, we will again use the 1-of-K (or one-hot) encoding to represent the discrete class labels. We define binary **class selection variables**

```math
y_{nk} = \begin{cases} 1 & \text{if  } \, y_n \in \mathcal{C}_k\\
0 & \text{otherwise} \end{cases}
```

Hence, the notations ``y_{nk}=1`` and ``y_n \in \mathcal{C}_k`` mean the same thing.

"""

# ╔═╡ 0d52466f-b092-4569-8c2d-b43c725887ae
md"""

#### Likelihood

Assume a Gaussian **class-conditional data-generating distribution** with **equal covariance matrix** across the classes,

```math
 p(x_n|\mathcal{C}_{k}) = \mathcal{N}(x_n|\mu_k,\Sigma) \tag{1}
 
```

with notational shorthand: ``\mathcal{C}_{k} \triangleq (y_n \in \mathcal{C}_{k})``.

"""

# ╔═╡ 23c74748-d294-11ef-2170-bf45b6379e4d
md"""
#### Prior

We use a categorical distribution for the class labels ``y_{nk}``: 

```math
p(\mathcal{C}_{k}) = \pi_k \tag{2}
```

"""

# ╔═╡ 23c75dc8-d294-11ef-3c57-614e75f06d8f
md"""
We will refer to this model (specified by Eqs. 1 and 2) as the **Gaussian-Categorical Model** ($(HTML("<span id='GCM'>GCM</span>"))). 

  * N.B. In the literature, this model (with possibly unequal ``\Sigma_k`` across classes) is often called the Gaussian Discriminant Analysis  model and the special case with equal covariance matrices ``\Sigma_k=\Sigma`` is also called Linear Discriminant Analysis. We think these names are a bit unfortunate as it may lead to confusion with the [discriminative method for classification](https://bmlip.github.io/course/lectures/Discriminative%20Classification.html).

"""

# ╔═╡ 23c763ce-d294-11ef-015b-736be1a5e9d6
md"""
As usual, once the model has been specified, the rest (inference for parameters and application) can be executed through straight probability theory.

"""

# ╔═╡ 23c7779a-d294-11ef-2e2c-6ba6cadb1381
md"""
## Parameter Estimation

In principle, a full Bayesian treatment requires us to specify prior distributions over the model parameters ``\theta = \{ \{\pi_k\}, \{\mu_k\}, \Sigma \}``, and then apply Bayes rule to obtain the corresponding posterior distributions. While this is certainly possible, the mathematics quickly becomes bewildering. Therefore, we opt for maximum likelihood estimation of the parameters as a more practical alternative.


"""

# ╔═╡ ffc80e65-a454-4b45-a9b7-76b01c7e96c0
section_outline("Exercise:", "Evaluate log-likelihood" , color= "yellow", header_level=4 )

# ╔═╡ 2e1ccf78-6097-4097-8bc8-1f1ec2d9c3ff
md"""

Show that the log-likelihood for the parameters evaluates to

```math
\log\, p(D|\theta) = \sum_{n,k} y_{nk} \underbrace{ \log\mathcal{N}(x_n|\mu_k,\Sigma) }_{ \text{see Gaussian lecture} } + \underbrace{ \sum_k m_k \log \pi_k }_{ \text{see multinomial lecture} } \tag{3}
```

where we used ``m_k \triangleq \sum_n y_{nk}``.

"""

# ╔═╡ 32cb67f6-1ed2-4d30-8493-e4eed9651526
details("Click for answer", 
md"""	   
```math
\begin{align*}
\log\, p(D|\theta) &= \log \prod_n p(x_n,y_n|\theta ) \quad \text{(assume IID data)} \\
  &= \sum_n \log p(x_n,y_n|\theta ) \\
  &= \sum_n \log \prod_k p(x_n,y_{nk}=1\,|\,\theta)^{y_{nk}} \\
  &=  \sum_{n,k} y_{nk} \log p(x_n,y_{nk}=1\,|\,\theta) \\
     &=  \sum_{n,k} y_{nk}  \log p(x_n|y_{nk}=1)  +  \sum_{n,k} y_{nk} \log p(y_{nk}=1) \\
   &=  \sum_{n,k} y_{nk}  \log\mathcal{N}(x_n|\mu_k,\Sigma)  +  \sum_{n,k} y_{nk} \log \pi_k \\
   &=  \sum_{n,k} y_{nk} \log\mathcal{N}(x_n|\mu_k,\Sigma)+ \sum_k m_k \log \pi_k 
\end{align*}
```
"""
	   )

# ╔═╡ 23c78d3e-d294-11ef-0309-ff10f58f0252
md"""
####  Maximization of log-likelihood 

Maximization of the LLH for the GDA model breaks down into

  * **Gaussian density estimation** for parameters ``\mu_k, \Sigma``, since the first term contains exactly the log-likelihood for MVG density estimation. We've already done this, see the [Gaussian distribution lesson](https://bmlip.github.io/course/lectures/The%20Gaussian%20Distribution.html#ML-for-Gaussian).
  * **Multinomial density estimation** for class priors ``\pi_k``, since the second term holds exactly the log-likelihood for multinomial density estimation, see the [Multinomial distribution lesson](https://bmlip.github.io/course/lectures/The%20Multinomial%20Distribution.html#ML-for-multinomial).

"""

# ╔═╡ 23c798ce-d294-11ef-0190-f342f30e2266
md"""
The ML for multinomial class prior (we've done this before!)

```math
\begin{align*}   
\hat \pi_k = \frac{m_k}{N} 
\end{align*}
```

"""

# ╔═╡ 23c7a54c-d294-11ef-0252-ef7a043e995c
md"""
Now group the data into separate classes and do MVG ML estimation for class-conditional parameters (we've done this before as well):

```math
\begin{align*}
 \hat \mu_k &= \frac{ \sum_n y_{nk} x_n} { \sum_n y_{nk} } = \frac{1}{m_k} \sum_n y_{nk} x_n \\
 \hat \Sigma  &= \frac{1}{N} \sum_{n,k} y_{nk} (x_n-\hat \mu_k)(x_n-\hat \mu_k)^T \\
  &= \sum_k \hat \pi_k \cdot \underbrace{ \left( \frac{1}{m_k} \sum_{n} y_{nk} (x_n-\hat \mu_k)(x_n-\hat \mu_k)^T  \right) }_{ \text{class-cond. variance} } \\
  &= \sum_k \hat \pi_k \cdot \hat \Sigma_k
\end{align*}
```

where ``\hat \pi_k``, ``\hat{\mu}_k`` and ``\hat{\Sigma}_k`` are the sample proportion, sample mean and sample variance for the ``k``th class, respectively.

"""

# ╔═╡ 23c7ab20-d294-11ef-1926-afae49e79923
md"""
Note that the binary class selection variable ``y_{nk}`` groups data from the same class.

"""

# ╔═╡ 23c7baa4-d294-11ef-22c1-31b0d86f5586
md"""
## Application: Class prediction for new Data

##### the posterior class probability

Let's apply the trained model to predict the class for a "new" input ``x_\bullet``:

```math
\begin{align*}
p(\mathcal{C}_k|x_\bullet,D ) &= \int p(\mathcal{C}_k|x_\bullet,\theta ) p(\theta|D) \mathrm{d}\theta \\
  &= \sigma\left( \beta_k^T x_\bullet + \gamma_k\right) \tag{4}
\end{align*}
```

where  
```math 
\sigma(a)_k \triangleq \frac{\exp(a_k)}{\sum_{k^\prime}\exp(a_{k^\prime})}
``` 
is $(HTML("<span id='softmax'>called a</span>")) [**softmax**](https://en.wikipedia.org/wiki/Softmax_function) (a.k.a., **normalized exponential**) function, and

```math
\begin{align*}
\beta_k &= \hat{\Sigma}^{-1} \hat{\mu}_k \\
\gamma_k &= - \frac{1}{2} \hat{\mu}_k^T \hat{\Sigma}^{-1} \hat{\mu}_k  + \log \hat{\pi}_k 
\end{align*}
```

"""

# ╔═╡ 84353cd1-e4fb-4689-9e90-d8995cbe2e9b
details("Click for proof of (4)", 
md""" ```math
\begin{align*}
p(\mathcal{C}_k|x_\bullet,D ) &= \int p(\mathcal{C}_k|x_\bullet,\theta ) \underbrace{p(\theta|D)}_{=\delta(\theta - \hat{\theta})} \mathrm{d}\theta \\
&= p(\mathcal{C}_k|x_\bullet,\hat{\theta} ) \\
&\propto p(\mathcal{C}_k)\,p(x_\bullet|\mathcal{C}_k) \\
&= \hat{\pi}_k \cdot \mathcal{N}(x_\bullet | \hat{\mu}_k, \hat{\Sigma}) \\
  &\propto \hat{\pi}_k \exp \left\{ { - {\frac{1}{2}}(x_\bullet - \hat{\mu}_k )^T \hat{\Sigma}^{ - 1} (x_\bullet - \hat{\mu}_k )} \right\}\\
  &=\exp \Big\{ \underbrace{-\frac{1}{2}x_\bullet^T \hat{\Sigma}^{ - 1} x_\bullet}_{\text{not a function of }k} + \underbrace{\hat{\mu}_k^T \hat{\Sigma}^{-1}}_{\beta_k^T} x_\bullet \underbrace{- {\frac{1}{2}}\hat{\mu}_k^T \hat{\Sigma}^{ - 1} \hat{\mu}_k  + \log \hat{\pi}_k }_{\gamma_k} \Big\}  \\
  &\propto  \frac{1}{Z}\exp\{\beta_k^T x_\bullet + \gamma_k\} \\
  &= \sigma\left( \beta_k^T x_\bullet + \gamma_k\right)
\end{align*}
``` """ )
		

# ╔═╡ 23c7c920-d294-11ef-1b6d-d98dd54dcbe3
md"""

##### The softmax function

The softmax function can be viewed as a smooth approximation to the maximum function.
Importantly, we did not impose the softmax posterior by assumption; rather, it emerged naturally by applying Bayes rule to our chosen prior and likelihood models.
"""

# ╔═╡ 23c7d700-d294-11ef-1268-c1441a3301a4
md"""
Note the following properties of the softmax function ``\sigma(a)_k``:

  * ``\sigma(a)_k`` is monotonically ascending function and hence it preserves the order of ``a_k``. That is, if ``a_j>a_k`` then ``\sigma(a)_j > \sigma(a)_k``.
  
  * ``\sigma(a)`` is always a proper probability distribution, since ``\forall_k \sigma(a)_k>0`` and ``\sum_k \sigma(a)_k = 1``.

"""

# ╔═╡ 23c82154-d294-11ef-0945-c9c94fc2a44d
md"""
#### making a decision

How should we classify a new input ``x_\bullet``?

The Bayesian answer is to compute the posterior distribution over classes, 
```math 
p(\mathcal{C}_k | x_\bullet,D)\,,
```
and this completes the classification task: the posterior encapsulates all available information about class membership given the input and data. 

If a definite classification **must** be made, a natural choice is the class with the highest posterior probability:

```math
\begin{align*}
k^* &= \arg\max_k p(\mathcal{C}_k|x_\bullet,D) \\
  &= \arg\max_k \left( \beta _k^T x_\bullet + \gamma_k \right)
\end{align*}
```
This corresponds to a maximum a posteriori (MAP) decision rule, which is both simple and effective in many practical settings.

"""

# ╔═╡ 23c7e4a0-d294-11ef-16e9-6f96a41baf97
md"""
## Discrimination Boundaries

The class log-posterior ``\log p(\mathcal{C}_k|x) \propto \beta_k^T x + \gamma_k`` is a linear function of the input features.

"""

# ╔═╡ 23c7f170-d294-11ef-1340-fbdf4ce5fd44
md"""
Therefore, the contours of equal probability (also known as **discriminant functions**, or **decision boundaries**), given by

```math
\log \frac{{p(\mathcal{C}_k|x,D )}}{{p(\mathcal{C}_j|x,D )}} \overset{!}{=} 0 \,,
```
are lines (hyperplanes) in the feature space.


"""

# ╔═╡ 5c746070-19a9-464b-aedc-401d016dfdb6
section_outline("Exercise:", "Discrimination boundaries" , color= "yellow", header_level=4 )

# ╔═╡ 8d78f9d3-7ba8-46b0-8d6f-231e681caa49
md"""
Show that the discrimination boundaries for the posterior class probabilities in Eq. (4) evaluates to a line (or hyperplane).
"""

# ╔═╡ 25e18c78-9cac-4faa-bb7c-ac036d0eac90
details("Click for answer",
md"""
```math
\begin{align}
&\log \frac{{p(\mathcal{C}_k|x,D )}}{{p(\mathcal{C}_j|x,D )}} \overset{!}{=} 0 \\
\implies &\frac{\beta_k^T x + \gamma_k}{\beta_j^T x + \gamma_j} = 0 \\
\implies &\beta_k^T x + \gamma_k = \beta_j^T x + \gamma_j \\
\implies &\beta_{kj}^T x + \gamma_{kj} = 0 \quad \text{(this is a line)}\,,
\end{align}
```

where we defined ``\beta_{kj} \triangleq \beta_k - \beta_j`` and similarly for ``\gamma_{kj}``.
"""	   
	   
	   )

# ╔═╡ a8adaf31-bee2-40e9-8d9b-bb9f1ad996ca
md"""
Now assume that the class-conditional feature distributions are modeled with class-dependent covariance matrices, i.e.,
```math
 p(x_n|\mathcal{C}_{k}) = \mathcal{N}(x_n|\mu_k,\Sigma_k) 
 
```
What do the decision boundaries look like in this case?
"""

# ╔═╡ b01a4a56-bed2-4a06-991a-831adc84aa3e
details("Click for answer",
md""" 
Following the same derivation as above (in the cell "Click for proof of (4)"), the posterior class probability evaluates to		
		```math
\begin{align*}
p(\mathcal{C}_k|x_\bullet,D ) \propto \exp \Big\{ \underbrace{-\frac{1}{2}x_\bullet^T \hat{\Sigma}_k^{ - 1} x_\bullet}_{\text{now a function of }k} + \underbrace{\hat{\mu}_k^T \hat{\Sigma}_k^{-1}}_{\beta_k^T} x_\bullet \underbrace{- {\frac{1}{2}}\hat{\mu}_k^T \hat{\Sigma}_k^{ - 1} \hat{\mu}_k  + \log \hat{\pi}_k }_{\gamma_k} \Big\}  \,.
\end{align*}
```		
Because the quadratic term ``x_\bullet^T \hat{\Sigma}_k^{-1} x_\bullet`` is now class-dependent, the decision boundaries are given by quadratic equations in ``x``. Hence, the decision boundaries are generally (hyper-)parabolic surfaces.
		
"""	)

# ╔═╡ 1a890e4b-b8a9-4a6e-b1f3-17863e1416d7
section_outline("Challenge Revisited:", "Apple or Peach", header_level=2, color="green")

# ╔═╡ 23c82e10-d294-11ef-286a-ff6fee0f2805
md"""

We'll apply the above results to solve the "apple or peach" example problem.

"""

# ╔═╡ 4481b38d-dc67-4c1f-ac0b-b348f0aea461
md"""
#### Multinomial (in this case binomial) density estimation
"""

# ╔═╡ 5092090d-cfac-4ced-b61e-fb7107a4c638
md"""
#### Estimate class-conditional multivariate Gaussian densities
"""

# ╔═╡ 90b862a5-d5bc-4122-a942-f01062daa86a
md"""
#### Posterior class probability of ``x_∙`` (prediction)
"""

# ╔═╡ 3791ac2a-8dc2-4d9a-8310-beae13d5a694
md"""
#### Discrimination boundary of the posterior 
Given by the condition ``p(apple|x;D) = p(peach|x;D) = 0.5``
"""

# ╔═╡ 21602809-d98b-43d7-8c41-80dc8de6da57
md"""
# Closing Thoughts
"""

# ╔═╡ 23c85d90-d294-11ef-375e-7101d4d3cbfa
md"""
## Why Be Bayesian?

A student in one of the previous years posed the following question at Piazza: 

> "After re-reading topics regarding generative classification, this question popped into my mind: Besides the sole purpose of the lecture, which is getting to know the concepts of generative classification and how to implement them, are there any advantages of using this instead of using deep neural nets (DNN), as they seem simpler and more powerful?"


The following answer was provided: 

If you are only interested in approximating a function, and you have lots of examples of desired behavior, then often a non-probabilistic DNN is a fine approach. However, if you are willing to formulate your models in a probabilistic framework, you can frequently improve on the deterministic approach in many ways. We list a few below:

1.	Bayesian Evidence as a Performance Metric
  - Model performance is evaluated using the evidence ``p(D|m)`` for a model, which inherently balances fit and complexity. This enables the use of the entire dataset for learning: there is no need for arbitrary splits into training and test sets.

2.	Parameter Uncertainty Enables Active Learning
  - By maintaining uncertainty over model parameters, Bayesian models support active learning, i.e., the selection of data points that are expected to be most informative (See the [lesson on intelligent agents](https://bmlip.github.io/course/lectures/Intelligent%20Agents%20and%20Active%20Inference.html)). This allows learning from smaller datasets, unlike deterministic deep networks, which often require massive amounts of labeled data.

3.	Predictions with Confidence Bounds
  - Bayesian models naturally yield predictive distributions, enabling uncertainty quantification (e.g., confidence intervals) around predictions.

4.	Explicit and Modular Assumptions
  - Priors, likelihoods, and structural assumptions are explicitly specified and can be independently modified, promoting transparency and model modularity.

5.	Unified Treatment of Accuracy and Complexity
  - Both data fit and model complexity are scored in the same probabilistic units. In contrast, how would you penalize overparameterized architectures (e.g., deep networks) in a deterministic framework?

6.	Data-Dependent, Optimal Learning Rates
  - Learning rates emerge naturally from Bayesian updates. Contrast this with the trial-and-error tuning needed in standard optimization.
  - Example: The Kalman gain is an optimal learning rate based on current uncertainty.

7.	Principled Knowledge Transfer
  - Bayesian inference enables posterior-to-prior propagation: results from one experiment (posterior) can inform the next (as a prior). This provides a principled mechanism for sequential learning and integration of heterogeneous information sources.


Admittedly, the probabilistic approach can be challenging to grasp at first, but the effort often pays off. It provides a principled, flexible, and robust framework for reasoning under uncertainty.


"""

# ╔═╡ 23c8698e-d294-11ef-2ae8-83bebd89d6c0
md"""
## Recap Generative Classification

Gaussian-Categorical Model specification:  

```math
p(x,\mathcal{C}_k|\,\theta) = \pi_k \cdot \mathcal{N}(x|\mu_k,\Sigma)
```

"""

# ╔═╡ 23c87654-d294-11ef-3aaf-595b207054a5
md"""
If the class-conditional distributions are Gaussian with equal covariance matrices across classes (``\Sigma_k = \Sigma``), then   the discriminant functions are hyperplanes in feature space.

"""

# ╔═╡ 23c88284-d294-11ef-113b-f57800a10e5d
md"""
ML estimation for ``\{\pi_k,\mu_k,\Sigma\}`` in the GCM model breaks down to simple density estimation for Gaussian and multinomial/categorical distributions.

"""

# ╔═╡ 23c88ec8-d294-11ef-3e0d-8de1377a14bf
md"""
Posterior class probability is a softmax function

```math
 p(\mathcal{C}_k|x,\theta ) \propto \exp\{\beta_k^T x + \gamma_k\}
```

where ``\beta _k= \Sigma^{-1} \mu_k`` and ``\gamma_k=- \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k  + \log \pi_k``.

"""

# ╔═╡ ca11db2d-aa15-4bf1-b949-529c7487d11d
md"""
# Exercises
"""

# ╔═╡ 24a08e5c-c2c1-4f1f-a2c1-998b30147e61
md"""

#### Fanta or Orangina? (**)

You have a machine that measures property ``x``, the "orangeness" of liquids. You wish to discriminate between ``C_1 = \text{`Fanta'}`` and ``C_2 = \text{`Orangina'}``. It is known that

```math
\begin{align*}
p(x|C_1) &= \begin{cases} 10 & 1.0 \leq x \leq 1.1\\
    0 & \text{otherwise}
    \end{cases}\\
p(x|C_2) &= \begin{cases} 200(x - 1) & 1.0 \leq x \leq 1.1\\
0 & \text{otherwise}
\end{cases}
\end{align*}
```

The prior probabilities ``p(C_1) = 0.6`` and ``p(C_2) = 0.4`` are also known from experience. 

- (a) A "Bayes Classifier" is given by

```math
 \text{Decision} = \begin{cases} C_1 & \text{if } p(C_1|x)>p(C_2|x) \\
                               C_2 & \text{otherwise}
                 \end{cases}
```

Derive the optimal Bayes classifier.  

- (b) The probability of making the wrong decision, given ``x``, is

```math
p(\text{error}|x)= p(C_1|\text{we-decide-} C_2, x) +  p(C_2|\text{we-decide-}C_1, x)
```

Compute the **total** error probability  ``p(\text{error})`` for the Bayes classifier in this example. 

"""

# ╔═╡ 66172ab6-7df8-4068-a748-b33b3f345d6d
details("Click for solution",
md"""
- (a) We choose ``C_1`` if ``p(C_1|x)/p(C_2|x) > 1``. This condition can be worked out as


```math
\frac{p(C_1|x)}{p(C_2|x)} = \frac{p(x|C_1)p(C_1)}{p(x|C_2)p(C_2)} = \frac{10 \times 0.6}{200(x-1)\times 0.4}>1 
```

which evaluates to choosing

```math
\mathrm{Decision} = \begin{cases}
C_1 & \text{ if $1.0\leq x < 1.075$} \\ 
C_2 & \text{ if $1.075 \leq x \leq 1.1$ } 
\end{cases}
```

The probability that ``x`` falls outside the interval ``[1.0,1.1]`` is zero.


- (b) The total probability of error is

```math
p(\text{error})=\int_x p(\text{error}|x)p(x) \mathrm{d}{x} \,.
```

We can work this out as


```math
\begin{align*}
p(\text{error}) &= \int_x p(\text{error}|x)p(x)\mathrm{d}{x}\\
&= \int_{1.0}^{1.075} p(C_2|x)p(x) \mathrm{d}{x} + \int_{1.075}^{1.1} p(C_1|x)p(x) \mathrm{d}{x}\\
&= \int_{1.0}^{1.075} p(x|C_2)p(C_2) \mathrm{d}{x} + \int_{1.075}^{1.1} p(x|C_1)p(C_1) \mathrm{d}{x}\\
&= \int_{1.0}^{1.075}0.4\cdot 200(x-1) \mathrm{d}{x} + \int_{1.075}^{1.1} 0.6\cdot 10 \mathrm{d}{x}\\
&=80\cdot[x^2/2-x]_{1.0}^{1.075} + 6\cdot[x]_{1.075}^{1.1}\\
&=0.225 + 0.15\\
&=0.375
\end{align*}
```

""")

# ╔═╡ 7e5213d6-4a68-4843-ab4c-77ea3ed8b0cd
md"""
#### [Bishop exercise 4.8](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf#page=241) (**)

Using (4.57) and (4.58) (from Bishop's book), derive the result (4.65) for the posterior class probability in the two-class generative model with Gaussian densities, and verify the results (4.66) and (4.67) for the parameters ``w`` and ``w0``.

"""

# ╔═╡ ef1e5885-7153-4b55-9f97-1e984c2504e6
details("Click for solution",
md"""
Substitute 4.64 into 4.58 to get

```math
\begin{align*}
a &= \log \left( \frac{ \frac{1}{(2\pi)^{D/2}} \cdot \frac{1}{|\Sigma|^{1/2}} \cdot \exp\left( -\frac{1}{2}(x-\mu_1)^T \Sigma^{-1} (x-\mu_1)\right) \cdot p(C_1)}{\frac{1}{(2\pi)^{D/2}} \cdot \frac{1}{|\Sigma|^{1/2}}\cdot  \exp\left( -\frac{1}{2}(x-\mu_2)^T \Sigma^{-1} (x-\mu_2)\right) \cdot p(C_2)}\right) \\
&= \log \left(  \exp\left(-\frac{1}{2}(x-\mu_1)^T \Sigma^{-1} (x-\mu_1) + \frac{1}{2}(x-\mu_2)^T \Sigma^{-1} (x-\mu_2) \right) \right) + \log \frac{p(C_1)}{p(C_2)} \\
&\qquad\vdots \\
&=( \mu_1-\mu_2)^T\Sigma^{-1}x - 0.5\left(\mu_1^T\Sigma^{-1}\mu_1 - \mu_2^T\Sigma^{-1} \mu_2\right)+ \log \frac{p(C_1)}{p(C_2)} 
\end{align*}
```

Substituting this into the right-most form of (4.57) we obtain (4.65), with ``w`` and ``w0`` given by (4.66) and (4.67), respectively.

""")

# ╔═╡ e65e0e33-3e4f-4765-84ea-a4fb5d43269e
md"""
# Appendix
"""

# ╔═╡ 3804c03c-6769-4258-806a-62e3d18221b5
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

# ╔═╡ 86f217fc-379c-46a0-a720-de956d456b2a
md"""
## Code
"""

# ╔═╡ a4463d74-04ea-428a-b5a5-504d96432a0a
md"""
Our data is structured as follows:
- ``y`` is whether a data point is an apple (`true`) or a peach (`false`)
- ``x`` has one column of data per fruit
"""

# ╔═╡ 3842654e-6dd7-427c-bb77-8b35a2f324fb
const Σ_secret = [0.2 0.1; 0.1 0.3];

# ╔═╡ 156d7866-00e1-47d8-ac38-52d72158f4d8
y = let
		p_apple = 0.7
		y = rand(MersenneTwister(23), Bernoulli(p_apple), N)
    end

# ╔═╡ cc8144d9-9ecf-4cbd-aea9-0c7a2fca2d94
p_apple_est = sum(y.==true) / length(y)

# ╔═╡ 19360d53-93d8-46fe-82d5-357015e75e22
π_hat = [p_apple_est; 1-p_apple_est]

# ╔═╡ eac2821e-b25c-4605-857a-cd3bd06303c1
X = let
		Σ = Σ_secret
		p_given_apple = MvNormal([1.0, 1.0], Σ) # p(X|y=apple)
		p_given_peach = MvNormal([1.7, 2.5], Σ) # p(X|y=peach)
	
		# Apple or peach?
		X = Matrix{Float64}(undef,2,N);
	
		rng = MersenneTwister(76)
		
		for n in 1:N
			X[:,n] = rand(rng, y[n] ? p_given_apple : p_given_peach)
		end # for
	    X
	end

# ╔═╡ 24d3c1f4-432f-419f-8854-69d8bfc135f8
X_apples, X_peaches =  X[:,findall(y)]', X[:,findall(.!y)]'

# ╔═╡ 10bfb9ea-46a6-4f4d-980e-ed2afce7b39a
d1 = fit_mle(FullNormal, X_apples')  # MLE density estimation d1 = N(μ₁, Σ₁)

# ╔═╡ cd310392-aabd-40e0-b06f-f8297c7eed6f
d2 = fit_mle(FullNormal, X_peaches') # MLE density estimation d2 = N(μ₂, Σ₂)

# ╔═╡ ba9fa93f-093c-4783-988f-27f4ba228e88
Σ_computed = π_hat[1]*cov(d1) + π_hat[2]*cov(d2) # Combine Σ₁ and Σ₂ into Σ

# ╔═╡ 46d2d5e9-bb6b-409a-acdc-cdffd1a6f797
conditionals = [
	MvNormal(mean(d1), Σ_computed)
	MvNormal(mean(d2), Σ_computed)
] # p(x|C)

# ╔═╡ 33d5d6e7-1208-4c5b-b651-429b3b6ad50b
function predict_class(k, X) # calculate p(Ck|X)
    norm = π_hat[1]*pdf(conditionals[1],X) + π_hat[2]*pdf(conditionals[2],X)
    return π_hat[k]*pdf(conditionals[k], X) ./ norm
end

# ╔═╡ b06c93fa-3439-4ed1-84ed-befc1ab7e40b
β(k) = inv(Σ)*mean(conditionals[k]);

# ╔═╡ 8610196d-2e0b-4a7f-96b2-2ca09078ffd6
γ(k) = -0.5 * mean(conditionals[k])' * inv(Σ) * mean(conditionals[k]) + 
		log(π_hat[k]);

# ╔═╡ 25002ffd-79c9-44bf-85d8-28c87df6c9df
function discriminant_x2(x1::Real)
    # Solve discriminant equation for x2
    β12 = β(1) .- β(2)
    γ12 = (γ(1) .- γ(2))[1,1]
    return -(β12[1]*x1 + γ12) / β12[2]
end;

# ╔═╡ d9efe8bb-c32c-40f4-89d9-8ace7a0665ba
x_test = [2.3; 1.5] # Features of 'new' data point

# ╔═╡ 69732524-90fd-46f4-9706-c07ce6226d2b
let
	# plot training data
	scatter(X_apples[:,1], X_apples[:,2], label="apples", marker=:x, markerstrokewidth=3)
	scatter!(X_peaches[:,1], X_peaches[:,2], label="peaches", marker=:+,  markerstrokewidth=3)
	plot!(; xlim=(-0.5, 2.5), ylim=(-0.5, 3.5))

	# plot test point
	scatter!([x_test[1]], [x_test[2]], label="unknown", c=:yellow, ms=9)
end # let

# ╔═╡ 723e09fc-ec63-4c47-844c-d821515ce0f4
@debug("p(apple|x=x∙) = $(predict_class(1,x_test))")

# ╔═╡ d5a342ff-6c5c-45af-affb-baf66ac7a7c1
let
	scatter(X_apples[:,1], X_apples[:,2], label="apples", marker=:x, markerstrokewidth=3)
	scatter!(X_peaches[:,1], X_peaches[:,2], label="peaches", marker=:+,  markerstrokewidth=3)
	scatter!([x_test[1]], [x_test[2]], label="unknown") # 'new' unlabelled data point

	# Discrimination boundary
	x1 = range(-1,length=10,stop=3)
	plot!(x1, discriminant_x2, color="black", label="")
	plot!(x1, discriminant_x2, fillrange=-10, alpha=0.2, color=:blue, label="")
	plot!(x1, discriminant_x2, fillrange=10, alpha=0.2, color=:red, xlims=(-0.5, 3), ylims=(-1, 4), label="")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
InteractiveUtils = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Distributions = "~0.25.120"
HypertextLiteral = "~0.9.5"
Plots = "~1.40.17"
PlutoTeachingTools = "~0.4.4"
PlutoUI = "~0.7.62"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "def3744cfb9ac0d0939e72d808a6a7f037d39060"

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
git-tree-sha1 = "3db9167c618b290a05d4345ca70de6d95304a32a"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.17"

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
git-tree-sha1 = "d0f6e09433d14161a24607268d89be104e743523"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.4"

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

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

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
git-tree-sha1 = "0fc001395447da85495b7fef1dfae9789fdd6e31"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.11"

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
# ╟─23c689fc-d294-11ef-086e-47c4f871bed2
# ╟─fe9d4fbc-f264-459b-8fbe-26663500f6c5
# ╟─23c6997e-d294-11ef-09a8-a50563e5975b
# ╟─f7a19975-a919-4659-9b6a-d8963a1cd6d9
# ╟─51a46b5e-0c35-4841-a4f3-413d5d294805
# ╟─876f47d8-b272-4e23-b5ec-5c7d615ff618
# ╟─69732524-90fd-46f4-9706-c07ce6226d2b
# ╟─e774041a-672d-40f3-ac8f-fc5dbf1bfc59
# ╟─5730758d-80cd-4d95-b16c-399c38cf585b
# ╟─23c73302-d294-11ef-0c12-571686b202a9
# ╟─23c73b54-d294-11ef-0ef8-8d9159139a1b
# ╟─0d52466f-b092-4569-8c2d-b43c725887ae
# ╟─23c74748-d294-11ef-2170-bf45b6379e4d
# ╟─23c75dc8-d294-11ef-3c57-614e75f06d8f
# ╟─23c763ce-d294-11ef-015b-736be1a5e9d6
# ╟─23c7779a-d294-11ef-2e2c-6ba6cadb1381
# ╟─ffc80e65-a454-4b45-a9b7-76b01c7e96c0
# ╟─2e1ccf78-6097-4097-8bc8-1f1ec2d9c3ff
# ╟─32cb67f6-1ed2-4d30-8493-e4eed9651526
# ╟─23c78d3e-d294-11ef-0309-ff10f58f0252
# ╟─23c798ce-d294-11ef-0190-f342f30e2266
# ╟─23c7a54c-d294-11ef-0252-ef7a043e995c
# ╟─23c7ab20-d294-11ef-1926-afae49e79923
# ╟─23c7baa4-d294-11ef-22c1-31b0d86f5586
# ╟─84353cd1-e4fb-4689-9e90-d8995cbe2e9b
# ╟─23c7c920-d294-11ef-1b6d-d98dd54dcbe3
# ╟─23c7d700-d294-11ef-1268-c1441a3301a4
# ╟─23c82154-d294-11ef-0945-c9c94fc2a44d
# ╟─23c7e4a0-d294-11ef-16e9-6f96a41baf97
# ╟─23c7f170-d294-11ef-1340-fbdf4ce5fd44
# ╟─5c746070-19a9-464b-aedc-401d016dfdb6
# ╟─8d78f9d3-7ba8-46b0-8d6f-231e681caa49
# ╟─25e18c78-9cac-4faa-bb7c-ac036d0eac90
# ╟─a8adaf31-bee2-40e9-8d9b-bb9f1ad996ca
# ╟─b01a4a56-bed2-4a06-991a-831adc84aa3e
# ╠═1a890e4b-b8a9-4a6e-b1f3-17863e1416d7
# ╟─23c82e10-d294-11ef-286a-ff6fee0f2805
# ╟─4481b38d-dc67-4c1f-ac0b-b348f0aea461
# ╠═cc8144d9-9ecf-4cbd-aea9-0c7a2fca2d94
# ╠═19360d53-93d8-46fe-82d5-357015e75e22
# ╟─5092090d-cfac-4ced-b61e-fb7107a4c638
# ╠═10bfb9ea-46a6-4f4d-980e-ed2afce7b39a
# ╠═cd310392-aabd-40e0-b06f-f8297c7eed6f
# ╠═ba9fa93f-093c-4783-988f-27f4ba228e88
# ╠═46d2d5e9-bb6b-409a-acdc-cdffd1a6f797
# ╟─90b862a5-d5bc-4122-a942-f01062daa86a
# ╠═33d5d6e7-1208-4c5b-b651-429b3b6ad50b
# ╠═723e09fc-ec63-4c47-844c-d821515ce0f4
# ╟─3791ac2a-8dc2-4d9a-8310-beae13d5a694
# ╠═b06c93fa-3439-4ed1-84ed-befc1ab7e40b
# ╠═8610196d-2e0b-4a7f-96b2-2ca09078ffd6
# ╠═25002ffd-79c9-44bf-85d8-28c87df6c9df
# ╠═d5a342ff-6c5c-45af-affb-baf66ac7a7c1
# ╟─21602809-d98b-43d7-8c41-80dc8de6da57
# ╟─23c85d90-d294-11ef-375e-7101d4d3cbfa
# ╟─23c8698e-d294-11ef-2ae8-83bebd89d6c0
# ╟─23c87654-d294-11ef-3aaf-595b207054a5
# ╟─23c88284-d294-11ef-113b-f57800a10e5d
# ╟─23c88ec8-d294-11ef-3e0d-8de1377a14bf
# ╟─ca11db2d-aa15-4bf1-b949-529c7487d11d
# ╟─24a08e5c-c2c1-4f1f-a2c1-998b30147e61
# ╟─66172ab6-7df8-4068-a748-b33b3f345d6d
# ╟─7e5213d6-4a68-4843-ab4c-77ea3ed8b0cd
# ╟─ef1e5885-7153-4b55-9f97-1e984c2504e6
# ╟─e65e0e33-3e4f-4765-84ea-a4fb5d43269e
# ╠═f1a40378-a27c-4aa0-a62c-600ffde0032f
# ╠═1b304964-6833-4cae-b84e-a5073f9586cd
# ╠═05ccf8cf-0711-4751-b378-5b0953eeedd0
# ╠═3804c03c-6769-4258-806a-62e3d18221b5
# ╠═6631c0e4-4941-442e-8dd4-fa307ee7a8c0
# ╠═f1575443-c9fb-4674-bbce-bf3a5a6d5a8d
# ╠═86f217fc-379c-46a0-a720-de956d456b2a
# ╟─a4463d74-04ea-428a-b5a5-504d96432a0a
# ╠═3842654e-6dd7-427c-bb77-8b35a2f324fb
# ╠═eac2821e-b25c-4605-857a-cd3bd06303c1
# ╠═156d7866-00e1-47d8-ac38-52d72158f4d8
# ╠═24d3c1f4-432f-419f-8854-69d8bfc135f8
# ╠═d9efe8bb-c32c-40f4-89d9-8ace7a0665ba
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
