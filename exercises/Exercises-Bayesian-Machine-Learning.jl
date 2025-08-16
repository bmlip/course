### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 616f9a3c-6e1b-11f0-1402-1bb9cceda105
md"""
# Bayesian Machine Learning

  * **[1]** (#) (a) Explain shortly the relation between machine learning and Bayes rule.       (b) How are Maximum a Posteriori (MAP) and Maximum Likelihood (ML) estimation related to Bayes rule and machine learning?

  * **[2]** (#) What are the four stages of the Bayesian design approach?

  * **[3]** (##) The Bayes estimate is a summary of a posterior distribution by a delta distribution on its mean, i.e.,

```math
\hat \theta_{bayes}  = \int \theta \, p\left( \theta |D \right)
\,\mathrm{d}{\theta}
```

Proof that the Bayes estimate minimizes the mean-squared error, i.e., proof that

```math
\hat \theta_{bayes} = \arg\min_{\hat \theta} \int_\theta (\hat \theta -\theta)^2 p \left( \theta |D \right) \,\mathrm{d}{\theta}
```

  * **[4]** (##) We consider the coin toss example from the notebook and use a conjugate prior for a Bernoulli likelihood function.     (a) Derive the Maximum Likelihood estimate.     (b) Derive the MAP estimate.           (c) Do these two estimates ever coincide (if so under what circumstances)?

  * **[5]** (##) A model ``m_1`` is described by a single parameter ``\theta``, with ``0 \leq \theta \leq1 $. The system can produce data $x \in \{0,1\}``. The sampling distribution and prior are given by

```math
\begin{aligned}
p(x|\theta,m_1) &=  \theta^x (1-\theta)^{(1-x)} \\
p(\theta|m_1) &= 6\theta(1-\theta)
\end{aligned}
```

(a) Work out the probability ``p(x=1|m_1)``.           (b) Determine the posterior ``p(\theta|x=1,m_1)``.        

Now consider a second model ``m_2`` with the following sampling distribution and prior on ``0 \leq \theta \leq 1``:

```math
\begin{aligned}
p(x|\theta,m_2) &= (1-\theta)^x \theta^{(1-x)} \\
p(\theta|m_2) &= 2\theta
\end{aligned}
```

(c) ​Determine the probability ``p(x=1|m_2)``.          

Now assume that the model priors are given by

```math
\begin{aligned}
    p(m_1) &= 1/3  \\
    p(m_2) &= 2/3
    \end{aligned}
```

(d) Compute the probability ``p(x=1)`` by "Bayesian model averaging", i.e., by weighing the predictions of both models appropriately.            (e) Compute the fraction of posterior model probabilities ``\frac{p(m_1|x=1)}{p(m_2|x=1)}``.             (f) Which model do you prefer after observation ``x=1``?

​ 

"""

# ╔═╡ Cell order:
# ╟─616f9a3c-6e1b-11f0-1402-1bb9cceda105
