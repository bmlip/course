### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 8e180218-6e1b-11f0-3ec2-fff499c26634
md"""
# Latent Variable Models and Variational Bayes

  * **[1]** (##) For a Gaussian mixture model, given by generative equations

```math
p(x,z) = \prod_{k=1}^K (\underbrace{\pi_k \cdot \mathcal{N}\left( x | \mu_k, \Sigma_k\right) }_{p(x,z_{k}=1)})^{z_{k}} 
```

proof that the marginal distribution for observations ``x_n`` evaluates to 

```math
p(x) = \sum_{j=1}^K \pi_k \cdot \mathcal{N}\left( x | \mu_j, \Sigma_j \right) 
```

  * **[2]** (#) Given the free energy functional ``F[q] = \sum_z q(z) \log \frac{q(z)}{p(x,z)}``, proof the [EE, DE and AC decompositions](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/Latent-Variable-Models-and-VB.ipynb#fe-decompositions).

  * **[3]** (#) The Free energy functional ``\mathrm{F}[q] = -\sum_z q(z) \log p(x,z) - \sum_z q(z) \log \frac{1}{q(z)}`` decomposes into "Energy minus Entropy". So apparently the entropy of the posterior ``q(z)`` is maximized. This entropy maximization may seem puzzling at first because inference should intuitively lead to *more* informed posteriors, i.e., posterior distributions whose entropy is smaller than the entropy of the prior. Explain why entropy maximization is still a reasonable objective.
  * **[4]** (#) Explain the following update rule for the mean of the Gaussian cluster-conditional data distribution (from the example about mean-field updating of a Gaussian mixture model):

```math
m_k = \frac{1}{\beta_k} \left( \beta_0 m_0 + N_k \bar{x}_k \right) \tag{B-10.61} 
```

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

  * **[6]** (###) Consult the internet on what *overfitting* and *underfitting* is and then explain how FE minimization finds a balance between these two (unwanted) extremes.
  * **[7]** (##) Consider a model ``p(x,z|\theta) = p(x|z,\theta) p(z|\theta)`` where ``x`` and ``z`` relate to observed and unobserved variables, respectively. Also available is an observed data set ``D=\left\{x_1,x_2,\ldots,x_N\right\}``. One iteration of the EM-algorithm for estimating the parameters ``\theta`` is described by (``m`` is the iteration counter)

```math
\hat{\theta}^{(m+1)} :=  \arg \max_\theta \left(\sum_z p(z|x=D,\hat{\theta}^{(m)}) \log p(x=D,z|\theta) \right) \,.
```

(a) Apparently, in order to execute EM, we need to work out an expression for the 'responsibility' ``p(z|x=D,\hat{\theta}^{(m)})``. Use Bayes rule to show how we can compute the responsibility that allows us to execute an EM step.    

(b) Why do we need multiple iterations in the EM algorithm?      

(c) Why can't we just use simple maximum log-likelihood to estimate parameters, as described by 

```math
\hat{\theta} := \arg \max_\theta  \log p(x=D,z|\theta) \,?
```

  * **[8]** In a particular model with hidden variables, the log-likelihood can be worked out to the following expression:

```math
 L(\theta) = \sum_n \log \left(\sum_k \pi_k\,\mathcal{N}(x_n|\mu_k,\Sigma_k)\right)
```

Do you prefer a gradient descent or EM algorithm to estimate maximum likelihood values for the parameters?  Explain your answer. (No need to work out the equations.)

"""

# ╔═╡ Cell order:
# ╟─8e180218-6e1b-11f0-3ec2-fff499c26634
