### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 90d96708-6e1b-11f0-0f81-99329f26acc4
md"""
# Discrete Data and the Multinomial Distribution

  * **[1]** (##) We consider IID data ``D = \{x_1,x_2,\ldots,x_N\}`` obtained from tossing a ``K``-sided die. We use a *binary selection variable*

```math
x_{nk} \equiv \begin{cases} 1 & \text{if $x_n$ lands on $k$th face}\\
    0 & \text{otherwise}
\end{cases}
```

with probabilities ``p(x_{nk} = 1)=\theta_k``.         (a) Write down the probability for the ``n``th observation ``p(x_n|\theta)`` and derive the log-likelihood ``\log p(D|\theta)``.        (b) Derive the maximum likelihood estimate for ``\theta``.

  * **[2]** (#) In the notebook, Laplace's generalized rule of succession (the probability that we throw the ``k``th face at the next toss) was derived as

```math
\begin{align*}
p(x_{\bullet,k}=1|D) = \frac{m_k + \alpha_k }{ N+ \sum_k \alpha_k}
\end{align*}
```

Provide an interpretation of the variables ``m_k,N,\alpha_k,\sum_k\alpha_k``.

  * **[3]** (##) Show that Laplace's generalized rule of succession can be worked out to a prediction that is composed of a prior prediction and data-based correction term.

  * **[4]** (#) Verify that     (a) the categorial distribution is a special case of the multinomial for ``N=1``.     (b) the Bernoulli is a special case of the categorial distribution for ``K=2``.     (c) the binomial is a special case of the multinomial for ``K=2``.



  * **[5]** (###) Consider a data set of binary variables ``D=\{x_1,x_2,\ldots,x_N\}`` with a Bernoulli distribution ``\mathrm{Ber}(x_k|\mu)`` as data generating distribution and a Beta prior for ``\mu``. Assume that you make ``n`` observations with ``x=1`` and ``N-n`` observations with ``x=0``. Now consider a new draw ``x_\bullet``. We are interested in computing ``p(x_\bullet|D)``. Show that the mean value for ``p(x_\bullet|D)`` lies in between the prior mean and Maximum Likelihood estimate.

  * **[6]** Consider a data set ``D = \{(x_1,y_1), (x_2,y_2),\dots,(x_N,y_N)\}`` with one-hot encoding for the ``K`` discrete classes, i.e.,  ``y_{nk} = 1`` if and only if ``y_n \in \mathcal{C}_k``, else ``y_{nk} = 0``. Also given are the class-conditional distribution ``p(x_n| y_{nk}=1,\theta) = \mathcal{N}(x_n|\mu_k,\Sigma)`` and multinomial prior ``p(y_{nk}=1) = \pi_k``.        (a) Proof that the joint log-likelihood is given by

```math
\begin{equation*}
\log p(D|\theta) =  \sum_{n,k} y_{nk} \log \mathcal{N}(x_n|\mu_k,\Sigma) + \sum_{n,k} y_{nk} \log \pi_k
\end{equation*}
```

(b) Show now that the MLE of the *class-conditional* mean is given by

```math
\begin{equation*}
 \hat \mu_k = \frac{\sum_n y_{nk} x_n}{\sum_n y_{nk}} 
\end{equation*}
```





"""

# ╔═╡ Cell order:
# ╟─90d96708-6e1b-11f0-0f81-99329f26acc4
