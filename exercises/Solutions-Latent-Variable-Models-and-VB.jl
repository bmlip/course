### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ d9cdddf4-6e1b-11f0-374a-05f3a76d1180
md"""
# Latent Variable Models and Variational Bayes

  * **[1]** (##) For a Gaussian mixture model, given by generative equations

```math
p(x,z) = \prod_{k=1}^K (\underbrace{\pi_k \cdot \mathcal{N}\left( x | \mu_k, \Sigma_k\right) }_{p(x,z_{k}=1)})^{z_{k}} \,,
```

proof that the marginal distribution for observations ``x_n`` evaluates to 

```math
p(x) = \sum_{j=1}^K \pi_k \cdot \mathcal{N}\left( x | \mu_j, \Sigma_j \right) \,.
```

> We start with the reverse sum rule,


```math
\begin{align*}
  p(x) &= \sum_{z} p(x,z) \\
  &= \sum_{z} \prod_{k=1}^K \left(\pi_k \cdot \mathcal{N}\left( x | \mu_k, \Sigma_k\right) \right)^{z_{k}}
  \end{align*}
```

> Exploiting the one-hot coding scheme for ``z``, we can re-write the RHS as


```math
\begin{equation*}
  \sum_{j=1}^K \prod_{k=1}^K \left(\pi_k \cdot \mathcal{N}\left( x | \mu_k, \Sigma_k\right) \right)^{I_{kj}} = \sum_{j=1}^K  \pi_j \cdot \mathcal{N}\left( x | \mu_j, \Sigma_j\right) 
  \end{equation*}
```

> where ``I_{kj} = 1`` if ``k=j`` and ``0`` otherwise.


"""

# ╔═╡ d9ce6ef6-6e1b-11f0-11aa-6f4340ad12bd
md"""
  * **[2]** (#) Given the free energy functional ``F[q] = \sum_z q(z) \log \frac{q(z)}{p(x,z)}``, proof the [EE, DE and AC decompositions](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/Latent-Variable-Models-and-VB.ipynb#fe-decompositions). 

    > The Energy-Entropy decomposition follows simply from  ``\log \frac{a}{b} = \log(a) - \log(b)``. The Divergence-Evidence decomposition follows from ``p(x,z) = p(z|x)p(x)`` and the Complexity-Accuracy decomposition follows from substituting ``p(x,z)=p(x|z)p(z)``. Altogether, leading to

```math
\begin{align*}
\mathrm{F}[q] &= \underbrace{-\sum_z q(z) \log p(x,z)}_{\text{energy}} - \underbrace{\sum_z q(z) \log \frac{1}{q(z)}}_{\text{entropy}} \tag{EE} \\
&= \underbrace{\sum_z q(z) \log \frac{q(z)}{p(z|x)}}_{\text{KL divergence}\geq 0} - \underbrace{\log p(x)}_{\text{log-evidence}} \tag{DE}\\
&= \underbrace{\sum_z q(z)\log\frac{q(z)}{p(z)}}_{\text{complexity}} - \underbrace{\sum_z q(z) \log p(x|z)}_{\text{accuracy}}  \tag{CA}
\end{align*}
```

  * **[3]** (#) The Free energy functional ``\mathrm{F}[q] = -\sum_z q(z) \log p(x,z) - \sum_z q(z) \log \frac{1}{q(z)}`` decomposes into "Energy minus Entropy". So apparently the entropy of the posterior ``q(z)`` is maximized. This entropy maximization may seem puzzling at first because inference should intuitively lead to *more* informed posteriors, i.e., posterior distributions whose entropy is smaller than the entropy of the prior. Explain why entropy maximization is still a reasonable objective. 

    > Note that Free Energy minimization is a balancing act: FE minimization implies entropy maximization *and at the same time* energy minimization. Minimizing the energy term leads to aligning ``q(z)`` with ``\log p(x,z)``, ie, it tries to move the bulk of the function ``q(z)`` to areas in ``z``-space where ``p(x,z)`` is large (``p(x,z)`` is here just a function of ``z``, since x is observed). However, aside from aligning with ``p(x,z)``, we want ``q(z)`` to be as uninformative as possible. Everything that can be inferred should be represented in ``p(x,z)`` (which is prior times likelihood). We don't want to learn anything that is not in either the prior or the likelihood. The entropy term balances the energy term by favoring distributions that are as uninformative as possible.
  * **[4]** (#) Explain the following update rule for the mean of the Gaussian cluster-conditional data distribution (from the example about mean-field updating of a Gaussian mixture model):

```math
m_k = \frac{1}{\beta_k} \left( \beta_0 m_0 + N_k \bar{x}_k \right) \tag{B-10.61} 
```

> We see here an example of "precision-weighted means add" when two sources of information are fused, just like precision-weighted means add when two Gaussians are multiplied, eg a prior and likelihood. In this case, the prior is ``m_0`` and the likelihood estimate is ``\bar{x}``. ``\beta_0`` can be interpreted as the number of pseudo-observations in the prior.


  * **[5]** (##) Consider a model ``p(x,z|\theta)``, where ``D=\{x_1,x_2,\ldots,x_N\}`` is observed, ``z`` are unobserved variables and ``\theta`` are parameters. The EM algorithm estimates the parameters by iterating over the following two equations (``i`` is the iteration index):

```math
\begin{align*}
q^{(i)}(z) &= p(z|D,\theta^{(i-1)}) \\
\theta^{(i)} &= \arg\max_\theta \sum_z q^{(i)}(z) \cdot \log p(D,z|\theta)
\end{align*}
```

Proof that this algorithm minimizes the Free Energy functional 

```math
\begin{align*}
F[q,\theta] =  \sum_z q(z) \log \frac{q(z)}{p(D,z|\theta)} 
\end{align*}
```

> Let's start with a prior estimate ``\theta^{(i-1)}`` and we want to minimize the free energy functional wrt ``q``. This leads to


```math
\begin{align*}
q^{(i)}(z) &= \arg\min_q F[q,\theta^{(i-1)}] \\
  &= \arg\min_q \sum_z q(z) \log \frac{q(z)}{p(D,z|\theta^{(i-1)})} \\
  &= \arg\min_q \sum_z q(z) \log \frac{q(z)}{p(z|D,\theta^{(i-1)}) \cdot p(D|\theta^{(i-1)})} \\
  &= p(z|D,\theta^{(i-1)})
\end{align*}
```

Next, we use ``q^{(i)}(z)=p(z|D,\theta^{(i-1)})`` and minimize the free energy w.r.t. ``\theta``, leading to

```math
\begin{align*}
  \theta^{(i)} &= \arg\min_\theta F[q^{(i)}(z),\theta] \\
  &= \arg\min_\theta \sum_z p(z|D,\theta^{(i-1)}) \log \frac{p(z|D,\theta^{(i-1)})}{p(D,z|\theta)} \\
  &= \arg\max_\theta \sum_z \underbrace{p(z|D,\theta^{(i-1)})}_{q^{(i)}(z)} \log p(D,z|\theta)
\end{align*}
```

  * **[6]** (###) Consult the internet on what *overfitting* and *underfitting* is and then explain how FE minimization finds a balance between these two (unwanted) extremes.

    > Overfitting relates to learning a posterior that "listens" too much to the data (and not enough to the prior). Underfitting does the opposite. The CA decompostion

```math
\begin{equation*}  
\underbrace{\sum_z q(z)\log\frac{q(z)}{p(z)}}_{\text{complexity}} - \underbrace{\sum_z q(z) \log p(x|z)}_{\text{accuracy}}  \tag{CA}
\end{equation*}
```

exposes this dilemma nicely. The complrxity term tries to keep the posterior ``q(z)`` near the prior ``p(z)`` whereas the accuracy term tries to align the posterior ``q(z)`` with the likelihood ``p(x|z)``. Thus, minimizing Free Energy keeps an eye on avoiding both under- and over-fitting.  

  * **[7]** (##) Consider a model ``p(x,z|\theta) = p(x|z,\theta) p(z|\theta)`` where ``x`` and ``z`` relate to observed and unobserved variables, respectively. Also available is an observed data set ``D=\left\{x_1,x_2,\ldots,x_N\right\}``. One iteration of the EM-algorithm for estimating the parameters ``\theta`` is described by (``m`` is the iteration counter)

```math
\hat{\theta}^{(m+1)} :=  \arg \max_\theta \left(\sum_z p(z|x=D,\hat{\theta}^{(m)}) \log p(x=D,z|\theta) \right) \,.
```

(a) Apparently, in order to execute EM, we need to work out an expression for the 'responsibility' ``p(z|x=D,\hat{\theta}^{(m)})``. Use Bayes rule to show how we can compute the responsibility that allows us to execute an EM step.    

> Use Bayes rule:


```math
p(z|x=D,\hat{\theta}^{(m)}) = \frac{p(x=D|z,\hat{\theta}^{(m)}) \,p(z|\hat{\theta}^{(m)})}{\int p(x=D|z,\hat{\theta}^{(m)}) \,p(z|\hat{\theta}^{(m)}) \,\mathrm{d}z}
```

Note that the RHS is an expression in ``z`` since ``D`` and ``\hat{\theta}`` are given. If you want to evaluate the RHS, you need to make a specific choice for your model 

```math
p(x,z|\theta) = \underbrace{p(x|z,\theta)}_{\text{likelihood}} \underbrace{p(z|\theta)}_{\text{prior}}
```

(b) Why do we need multiple iterations in the EM algorithm?     

> We must have a parameter estimate in order to compute the responsibilities, and vice versa, we need responsibilities to update the parameter estimate. Thus, in the EM algorithm, we iterate between updating responsibilities (beliefs about ``z``) and parameter estimates (beliefs about ``\theta``).


(c) Why can't we just use simple maximum log-likelihood to estimate parameters, as described by 

```math
\hat{\theta} := \arg \max_\theta  \log p(x=D,z|\theta) \,?
```

> Because ``z`` is not observed.


  * **[8]** In a particular model with hidden variables, the log-likelihood can be worked out to the following expression:

```math
 L(\theta) = \sum_n \log \left(\sum_k \pi_k\,\mathcal{N}(x_n|\mu_k,\Sigma_k)\right)
```

Do you prefer a gradient descent or EM algorithm to estimate maximum likelihood values for the parameters?  Explain your answer. (No need to work out the equations.)

> Since this expression does not degenerate into simple MVGs, the EM approach is in practice preferred.


"""

# ╔═╡ Cell order:
# ╟─d9cdddf4-6e1b-11f0-374a-05f3a76d1180
# ╟─d9ce6ef6-6e1b-11f0-11aa-6f4340ad12bd
