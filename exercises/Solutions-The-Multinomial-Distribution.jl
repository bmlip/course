### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 94e6dd58-6e1b-11f0-1ac5-fbc0cf7bc2b6
md"""
# Discrete Data and the Multinomial Distribution

  * **[1]** (##) We consider IID data ``D = \{x_1,x_2,\ldots,x_N\}`` obtained from tossing a ``K``-sided die. We use a *binary selection variable*

```math
x_{nk} \equiv \begin{cases} 1 & \text{if $x_n$ lands on $k$th face}\\
    0 & \text{otherwise}
\end{cases}
```

with probabilities ``p(x_{nk} = 1)=\theta_k``.         (a) Write down the probability for the ``n``th observation ``p(x_n|\theta)`` and derive the log-likelihood ``\log p(D|\theta)``.        (b) Derive the maximum likelihood estimate for ``\theta``.

> See lecture notes (on class homepage).        (a)


```math
p(x_n|\theta) = \prod_k \theta_k^{x_{nk}} \quad \text{subject to} \quad \sum_k \theta_k = 1 \,.
```

```math
\ell(\theta)  = \sum_k m_k \log \theta_k
```

> where ``m_k = \sum_n x_{nk}``.       (b)


```math
\hat \theta = \frac{m_k}{N}\,,
```

which is the *sample proportion*.

  * **[2]** (#) In the notebook, Laplace's generalized rule of succession (the probability that we throw the ``k``th face at the next toss) was derived as

```math
\begin{align*}
p(x_{\bullet,k}=1|D) = \frac{m_k + \alpha_k }{ N+ \sum_k \alpha_k}
\end{align*}
```

Provide an interpretation of the variables ``m_k,N,\alpha_k,\sum_k\alpha_k``.

> ``m_k`` is the total number of occurances that we threw ``k`` eyes, ``\alpha_k`` is the prior pseudo counts representing the number of observations in the ``k``th that we assume to have seen already. :\sum*k m*k = N $ is the total number of rolls and :\sum*k \alpha*k $ is the total number of prior pseudo rolls.


  * **[3]** (##) Show that Laplace's generalized rule of succession can be worked out to a prediction that is composed of a prior prediction and data-based correction term.

```math
\begin{align*}
p(x_{\bullet,k}=1|D) &= \frac{m_k + \alpha_k }{ N+ \sum_k \alpha_k} \\
&= \frac{m_k}{N+\sum_k \alpha_k}  + \frac{\alpha_k}{N+\sum_k \alpha_k}\\
&= \frac{m_k}{N+\sum_k \alpha_k} \cdot \frac{N}{N} + \frac{\alpha_k}{N+\sum_k \alpha_k}\cdot \frac{\sum_k \alpha_k}{\sum_k\alpha_k} \\
&= \frac{N}{N+\sum_k \alpha_k} \cdot \frac{m_k}{N} + \frac{\sum_k \alpha_k}{N+\sum_k \alpha_k} \cdot \frac{\alpha_k}{\sum_k\alpha_k} \\
&= \frac{N}{N+\sum_k \alpha_k} \cdot \frac{m_k}{N} + \bigg( \frac{\sum_k \alpha_k}{N+\sum_k \alpha_k} + \underbrace{\frac{N}{N+\sum_k \alpha_k} - \frac{N}{N+\sum_k \alpha_k}}_{0}\bigg) \cdot \frac{\alpha_k}{\sum_k\alpha_k} \\
&= \frac{N}{N+\sum_k \alpha_k} \cdot \frac{m_k}{N} + \bigg( 1 - \frac{N}{N+\sum_k \alpha_k}\bigg) \cdot \frac{\alpha_k}{\sum_k\alpha_k} \\
&= \underbrace{\frac{\alpha_k}{\sum_k\alpha_k}}_{\text{prior prediction}} + \underbrace{\frac{N}{N+\sum_k \alpha_k} \cdot \underbrace{\left(\frac{m_k}{N} - \frac{\alpha_k}{\sum_k\alpha_k}\right)}_{\text{prediction error}}}_{\text{data-based correction}}
\end{align*}
```

  * **[4]** (#) Verify that     (a) the categorial distribution is a special case of the multinomial for ``N=1``.     (b) the Bernoulli is a special case of the categorial distribution for ``K=2``.     (c) the binomial is a special case of the multinomial for ``K=2``.

> (a) The probability mass function of a multinomial distribution is ``p(D_m|\mu) =\frac{N!}{m_1! m_2!\ldots m_K!} \,\prod_k \mu_k^{m_k}`` over the data frequencies ``D_m=\{m_1,\ldots,m_K\}`` with the constraint that ``\sum_k \mu_k = 1`` and ``\sum_k m_k=N``. Setting ``N=1`` we see that ``p(D_m|\mu) \propto \prod_k \mu_k^{m_k}`` with ``\sum_k m_k=1``, making the sample space one-hot coded given by the categorical distribution.       (b) When ``K=2``, the constraint for the categorical distribution takes the form ``m_1=1-m_2`` leading to ``p(D_m|\mu) \propto \mu_1^{m_1}(1-\mu_1)^{1-m_1}`` which is associated with the Bernoulli distribution.       (c) Plugging ``K=2`` into the multinomial distribution leads to ``p(D_m|\mu) =\frac{N!}{m_1! m_2!}\mu_1^{m_1}\left(\mu_2^{m_2}\right)`` with the constraints ``m_1+m_2=N`` and ``\mu_1+\mu_2=1``. Then plugging the constraints back in we obtain ``p(D_m|\mu) = \frac{N!}{m_1! (N-m1)!}\mu_1^{m_1}\left(1-\mu_1\right)^{N-m_1}`` as the binomial distribution.




  * **[5]** (###) Consider a data set of binary variables ``D=\{x_1,x_2,\ldots,x_N\}`` with a Bernoulli distribution ``\mathrm{Ber}(x_k|\mu)`` as data generating distribution and a Beta prior for ``\mu``. Assume that you make ``n`` observations with ``x=1`` and ``N-n`` observations with ``x=0``. Now consider a new draw ``x_\bullet``. We are interested in computing ``p(x_\bullet|D)``. Show that the mean value for ``p(x_\bullet|D)`` lies in between the prior mean and Maximum Likelihood estimate.

> In the lectures we have seen that ``p(x_\bullet =1|D) = \frac{a+n}{a+b+N}``, where ``a`` and ``b`` are parameters of the Beta prior. The ML estimate is ``\frac{n}{N}`` and the prior mean is ``\frac{a}{a+b}``. To show that the prediction lies in between ML and prior estimate, we will try to write the prediction as a convex combination of the latter two. That is we want to solve for ``\lambda``.


```math
\begin{align*}
(1-\lambda) \frac{n}{N} + \lambda\frac{a}{a+b} &= \frac{a+n}{a+b+N} \\
\lambda &= \frac{1}{1+\frac{N}{a+b}}  
\end{align*}
```

> Since ``a,b`` and ``N`` are positive, it follows that ``0<\lambda <1``. This means the prediction is a convex combination of prior and ML estimates and thus lies in between the two.


  * **[6]** Consider a data set ``D = \{(x_1,y_1), (x_2,y_2),\dots,(x_N,y_N)\}`` with one-hot encoding for the ``K`` discrete classes, i.e.,  ``y_{nk} = 1`` if and only if ``y_n \in \mathcal{C}_k``, else ``y_{nk} = 0``. Also given are the class-conditional distribution ``p(x_n| y_{nk}=1,\theta) = \mathcal{N}(x_n|\mu_k,\Sigma)`` and multinomial prior ``p(y_{nk}=1) = \pi_k``.       .        (a) Proof that the joint log-likelihood is given by

```math
\begin{equation*}
\log p(D|\theta) =  \sum_{n,k} y_{nk} \log \mathcal{N}(x_n|\mu_k,\Sigma) + \sum_{n,k} y_{nk} \log \pi_k
\end{equation*}
```

```math
\begin{align*}
 \log p(D|\theta) &= \sum_n \log \prod_k p(x_n,y_{nk}|\theta)^{y_{nk}} \\
  &=  \sum_{n,k} y_{nk} \log p(x_n,y_{nk}|\theta)\\
  &=  \sum_{n,k} y_{nk} \log \mathcal{N}(x_n|\mu_k,\Sigma) + \sum_{n,k} y_{nk} \log \pi_k
\end{align*}
```

(b) Show now that the MLE of the *class-conditional* mean is given by

```math
\begin{equation*}
 \hat \mu_k = \frac{\sum_n y_{nk} x_n}{\sum_n y_{nk}} 
\end{equation*}
```





"""

# ╔═╡ Cell order:
# ╟─94e6dd58-6e1b-11f0-1ac5-fbc0cf7bc2b6
