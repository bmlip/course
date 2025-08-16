### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 925177f6-6e1b-11f0-0ade-3324e1236594
md"""
# Bayesian Machine Learning

  * **[1]** (#) (a) Explain shortly the relation between machine learning and Bayes rule.       (b) How are Maximum a Posteriori (MAP) and Maximum Likelihood (ML) estimation related to Bayes rule and machine learning?

> (a) Machine learning is inference over models (hypotheses, parameters, etc.) from a given data set. *Bayes rule* makes this statement precise. Let ``\theta \in \Theta`` and ``D`` represent a model parameter vector and the given data set, respectively. Then, Bayes rule,


```math
p(\theta|D) = \frac{p(D|\theta)}{p(D)} p(\theta)
```

relates the information that we have about ``\theta`` before we saw the data (i.e., the distribution ``p(\theta)``) to what we know after having seen the data, ``p(\theta|D)``.      

> (b) The *Maximum a Posteriori* (MAP) estimate picks a value ``\hat\theta`` for which the posterior distribution ``p(\theta|D)`` is maximal, i.e.,


```math
 \hat\theta_{MAP} = \arg\max_\theta p(\theta|D)
```

In a sense, MAP estimation approximates Bayesian learning, since we approximated ``p(\theta|D)`` by ``\delta(\theta-\hat\theta_{\text{MAP}})``. Note that, by Bayes rule, 

```math
\arg\max_\theta p(\theta|D) = \arg\max_\theta p(D|\theta)p(\theta)
```

If we further assume that prior to seeing the data all values for ``\theta`` are equally likely (i.e., ``p(\theta)=\text{const.}``), then the MAP estimate reduces to the *Maximum Likelihood* estimate,

```math
 \hat\theta_{ML} = \arg\max_\theta p(D|\theta)
```

"""

# ╔═╡ 925198b2-6e1b-11f0-043d-972f67c6ca2d
md"""
  * **[2]** (#) What are the four stages of the Bayesian design approach?

> (1) Model specification, (2) parameter estimation, (3) model evaluation and (4) application of the model to tasks.


"""

# ╔═╡ 9251bf04-6e1b-11f0-3ae4-ddf699a02532
md"""
  * **[3]** (##) The Bayes estimate is a summary of a posterior distribution by a delta distribution on its mean, i.e.,

```math
\hat \theta_{bayes}  = \int \theta \, p\left( \theta |D \right)
\,\mathrm{d}{\theta}
```

Proof that the Bayes estimate minimizes the mean-squared error, i.e., proof that

```math
\hat \theta_{bayes} = \arg\min_{\hat \theta} \int_\theta (\hat \theta -\theta)^2 p \left( \theta |D \right) \,\mathrm{d}{\theta}
```

> To minimize the expected mean-squared error we will look for ``\hat{\theta}`` that makes the gradient of the integral with respect to ``\hat{\theta}`` vanish.


```math
\begin{align*}
  \nabla_{\hat{\theta}}  \int_\theta (\hat \theta -\theta)^2 p \left( \theta |D \right) \,\mathrm{d}{\theta} &= 0 \\
  \int_\theta \nabla_{\hat{\theta}}  (\hat \theta -\theta)^2 p \left( \theta |D \right) \,\mathrm{d}{\theta} &= 0 \\
  \int_\theta  2(\hat \theta -\theta) p \left( \theta |D \right) \,\mathrm{d}{\theta} &= 0 \\
  \int_\theta  \hat \theta p \left( \theta |D \right) \,\mathrm{d}{\theta} &= \int_\theta  \theta p \left( \theta |D \right) \,\mathrm{d}{\theta} \\
  \hat \theta \underbrace{\int_\theta p \left( \theta |D \right) \,\mathrm{d}{\theta}}_{1} &= \int_\theta  \theta p \left( \theta |D \right) \,\mathrm{d}{\theta} \\
  \Rightarrow \hat \theta &= \int_\theta  \theta p \left( \theta |D \right) \,\mathrm{d}{\theta}
\end{align*}
```

"""

# ╔═╡ 9252298c-6e1b-11f0-3366-e5d85bcc9f93
md"""
  * **[4]** (##) We consider the coin toss example from the notebook and use a conjugate prior for a Bernoulli likelihood function.     (a) Derive the Maximum Likelihood estimate.     (b) Derive the MAP estimate.           (c) Do these two estimates ever coincide (if so under what circumstances)?

> (a) The likelihood is given by ``p(D|\mu) = \mu^n\cdot (1-\mu)^{(N-n)}``. It follows that


```math
\begin{align*}
    \nabla \log p(D|\mu) &= 0 \\
    \nabla \left( n\log \mu + (N-n)\log(1-\mu)\right) &= 0\\
    \frac{n}{\mu} - \frac{N-n}{1-\mu} &= 0 \\
    \rightarrow \hat{\mu}_{\text{ML}} &= \frac{n}{N}
  \end{align*}
```

> (b) Assuming a beta prior ``\mathcal{B}(\mu|\alpha,\beta)``, we can write the posterior as as


```math
\begin{align*}
   p(\mu|D) &\propto p(D|\mu)p(\mu) \\
      &\propto \mu^n (1-\mu)^{N-n} \mu^{\alpha-1} (1-\mu)^{\beta-1} \\
      &\propto \mathcal{B}(\mu|n+\alpha,N-n+\beta)
   \end{align*}
```

> The MAP estimate for a beta distribution ``\mathcal{B}(a,b)`` is located at ``\frac{a - 1}{a+b-2}``, see [wikipedia](https://en.wikipedia.org/wiki/Beta_distribution). Hence,


```math
\begin{align*}
\hat{\mu}_{\text{MAP}} &= \frac{(n+\alpha)-1}{(n+\alpha) + (N-n+\beta) -2} \\
  &= \frac{n+\alpha-1}{N + \alpha +\beta -2}
\end{align*}
```

> (c) As ``N`` gets larger, the MAP estimate approaches the ML estimate. In the limit the MAP solution converges to the ML solution.


"""

# ╔═╡ 9252e21c-6e1b-11f0-364b-59b837678333
md"""
  * **[5]** (##) A model ``m_1`` is described by a single parameter ``\theta``, with ``0 \leq \theta \leq1 $. The system can produce data $x \in \{0,1\}``. The sampling distribution and prior are given by

```math
\begin{align*}
p(x|\theta,m_1) &=  \theta^x (1-\theta)^{(1-x)} \\
p(\theta|m_1) &= 6\theta(1-\theta)
\end{align*}
```

(a) Work out the probability ``p(x=1|m_1)``.    

```math
\begin{align*}
  p(x=1|m_1) &= \int_0^1 p(x=1|\theta,m_1) p(\theta|m_1) \mathrm{d}\theta \\
  &= \int \theta \cdot 6\theta (1-\theta) \mathrm{d}\theta \\
  &= 6 \cdot \left(\frac{1}{3}\theta^3 - \frac{1}{4}\theta^4\right) \bigg|_0^1 \\
  &= 6 \cdot (\frac{1}{3} - \frac{1}{4}) = \frac{1}{2}
\end{align*}
```

(b) Determine the posterior ``p(\theta|x=1,m_1)``.     

```math
\begin{align*}
  p(\theta|x=1,m_1) &= \frac{p(x=1|\theta) p(\theta|m_1)}{p(x=1|m_1)} \\
  &= 2\cdot \theta \cdot 6\theta (1-\theta) \\
  &= \begin{cases} 12 \theta^2 (1-\theta) & \text{if }0 \leq \theta \leq 1 \\
  0 & \text{otherwise} \end{cases}
  \end{align*}
```

Now consider a second model ``m_2`` with the following sampling distribution and prior on ``0 \leq \theta \leq 1``:

```math
\begin{align*}
p(x|\theta,m_2) &= (1-\theta)^x \theta^{(1-x)} \\
p(\theta|m_2) &= 2\theta
\end{align*}
```

(c) ​Determine the probability ``p(x=1|m_2)``.    

```math
\begin{align*}
  p(x=1|m_2) &= \int_0^1 p(x=1|\theta,m_2) p(\theta|m_2) \mathrm{d}\theta \\
  &= \int (1-\theta) \cdot 2\theta \mathrm{d}\theta \\
  &= 2 \cdot \left( \frac{1}{2}\theta^2 - \frac{1}{3}\theta^3 \right) \bigg|_0^1 \\
  &= 2 \cdot (\frac{1}{2} - \frac{1}{3}) = \frac{1}{3}
  \end{align*}
```

Now assume that the model priors are given by

```math
\begin{align*}
    p(m_1) &= 1/3  \\
    p(m_2) &= 2/3
    \end{align*}
```

(d) Compute the probability ``p(x=1)`` by "Bayesian model averaging", i.e., by weighing the predictions of both models appropriately.  

```math
\begin{align*}
    p(x=1) &= \sum_{k=1}^2 p(x=1|m_k) p(m_k)  \\
    &= \frac{1}{2} \cdot \frac{1}{3} + \frac{1}{3} \cdot \frac{2}{3} = \frac{7}{18} 
    \end{align*}
```

(e) Compute the fraction of posterior model probabilities ``\frac{p(m_1|x=1)}{p(m_2|x=1)}``.     

```math
\frac{p(m_1|x=1)}{p(m_2|x=1)} = \frac{p(x=1|m_1) p(m_1)}{p(x=1|m_2) p(m_2)} = \frac{\frac{1}{2} \cdot \frac{1}{3}}{\frac{1}{3} \cdot \frac{2}{3}} =\frac{3}{4}
```

(f) Which model do you prefer after observation ``x=1``?

> In principle, the observation ``x=1`` favors model ``m_2``, since ``p(m_2|x=1) = \frac{4}{3} \times p(m_1|x=1)``. However, note that ``\log_{10} \frac{3}{4} \approx -0.125``, so the extra evidence for ``m_2`` relative to ``m_1`` is very low. At this point, after 1 observation, we have no preference for a model yet.


​

"""

# ╔═╡ Cell order:
# ╟─925177f6-6e1b-11f0-0ade-3324e1236594
# ╟─925198b2-6e1b-11f0-043d-972f67c6ca2d
# ╟─9251bf04-6e1b-11f0-3ae4-ddf699a02532
# ╟─9252298c-6e1b-11f0-3366-e5d85bcc9f93
# ╟─9252e21c-6e1b-11f0-364b-59b837678333
