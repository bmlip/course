### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ dad449de-6e1b-11f0-122f-5ffd7030f2f9
md"""
# Generative Classification

  * **[1]** You have a machine that measures property ``x``, the "orangeness" of liquids. You wish to discriminate between ``C_1 = \text{`Fanta'}`` and ``C_2 = \text{`Orangina'}``. It is known that

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

(a) (##) A "Bayes Classifier" is given by

```math
 \text{Decision} = \begin{cases} C_1 & \text{if } p(C_1|x)>p(C_2|x) \\
                               C_2 & \text{otherwise}
                 \end{cases}
```

Derive the optimal Bayes classifier.  

> We choose ``C_1`` if ``p(C_1|x)/p(C_2|x) > 1``. This condition can be worked out as


```math
\frac{p(C_1|x)}{p(C_2|x)} = \frac{p(x|C_1)p(C_1)}{p(x|C_2)p(C_2)} = \frac{10 \times 0.6}{200(x-1)\times 0.4}>1 
```

> which evaluates to choosing


```math
\begin{align*}
C_1 &\quad \text{ if $1.0\leq x < 1.075$}\\ 
C_2 &\quad \text{ if $1.075 \leq x \leq 1.1$ }
\end{align*}
```

> The probability that ``x`` falls outside the interval ``[1.0,1.1]`` is zero.


(b) (###) The probability of making the wrong decision, given ``x``, is

```math
p(\text{error}|x)= p(C_1|x,\text{we-decide-}C_2) +  p(C_2|x,\text{we-decide-}C_1)
```

Compute the **total** error probability  ``p(\text{error})`` for the Bayes classifier in this example. 

> (b) The total probability of error ``p(\text{error})=\int_x p(\text{error}|x)p(x) \mathrm{d}{x}``. We can work this out as


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

"""

# ╔═╡ dad46bf0-6e1b-11f0-1d60-0323d5d7c1ad
md"""
  * **[2]** (#) (see Bishop exercise 4.8): Using (4.57) and (4.58) (from Bishop's book), derive the result (4.65) for the posterior class probability in the two-class generative model with Gaussian densities, and verify the results (4.66) and (4.67) for the parameters ``w`` and ``w0``.

> Substitute 4.64 into 4.58 to get


```math
\begin{align*}
a &= \log \left( \frac{ \frac{1}{(2\pi)^{D/2}} \cdot \frac{1}{|\Sigma|^{1/2}} \cdot \exp\left( -\frac{1}{2}(x-\mu_1)^T \Sigma^{-1} (x-\mu_1)\right) \cdot p(C_1)}{\frac{1}{(2\pi)^{D/2}} \cdot \frac{1}{|\Sigma|^{1/2}}\cdot  \exp\left( -\frac{1}{2}(x-\mu_2)^T \Sigma^{-1} (x-\mu_2)\right) \cdot p(C_2)}\right) \\
&= \log \left(  \exp\left(-\frac{1}{2}(x-\mu_1)^T \Sigma^{-1} (x-\mu_1) + \frac{1}{2}(x-\mu_2)^T \Sigma^{-1} (x-\mu_2) \right) \right) + \log \frac{p(C_1)}{p(C_2)} \\
&= ... \\
&=( \mu_1-\mu_2)^T\Sigma^{-1}x - 0.5\left(\mu_1^T\Sigma^{-1}\mu_1 - \mu_2^T\Sigma^{-1} \mu_2\right)+ \log \frac{p(C_1)}{p(C_2)} 
\end{align*}
```

> Substituting this into the right-most form of (4.57) we obtain (4.65), with ``w`` and ``w0`` given by (4.66) and (4.67), respectively.




"""

# ╔═╡ dad48d74-6e1b-11f0-1c40-3dffee205c13
md"""
  * **[3]** (##) (see Bishop exercise 4.10).

> We can write the log-likelihood as


```math
\begin{align*}
\log p(\{\phi_n,t_n\}|\{\pi_k\}) \propto -0.5\sum_n\sum_kt_{nk}\left(\log |\Sigma|+(\phi_n-\mu_k)^T\Sigma^{-1}(\phi-\mu)\right)
\end{align*}
```

> The derivatives of the likelihood with respect to mean and shared covariance are respectively


```math
\begin{align*}
\nabla_{\mu_k}\log p(\{\phi_n,t_n\}|\{\pi_k\}) &= \sum_n t_{nk}\Sigma^{-1}\left(\phi_n-\mu_k\right)  \\
&= \sum_n t_{nk}\left(\phi_n-\mu_k\right)  \\
\text{set deriv. to $0$} \Rightarrow \mu_k &= \frac{1}{N_k}\sum_n t_{nk}\phi_n  \\
\nabla_{\Sigma}\log p(\{\phi_n,t_n\}|\{\pi_k\})&=\sum_n\sum_k t_{nk}\left(\Sigma - (\phi_n-\mu_k)(\phi_n-\mu_k)^T\right)  \\
\text{set deriv. to $0$} \Rightarrow\Sigma &=  \frac{1}{N}\sum_k\sum_n t_{nk}(\phi_n-\mu_k)(\phi_n-\mu_k)^T 
\end{align*}
```

> where we used ``\sum_n t_{nk}=N_k`` and  ``\sum_{n,k} t_{nk}=N``.




"""

# ╔═╡ dad53d96-6e1b-11f0-0751-ff28377e58cc
md"""
# Discriminative Classification

  * **[1]**  Given a data set ``D=\{(x_1,y_1),\ldots,(x_N,y_N)\}``, where ``x_n \in \mathbb{R}^M`` and ``y_n \in \{0,1\}``. The probabilistic classification method known as *logistic regression* attempts to model these data as

```math
p(y_n=1|x_n) = \sigma(\theta^T x_n + b)
```

where ``\sigma(x) = 1/(1+e^{-x})`` is the *logistic function*. Let's introduce shorthand notation ``\mu_n=\sigma(\theta^T x_n + b)``. So, for every input ``x_n``, we have a model output ``\mu_n`` and an actual data output ``y_n``.                      (a) Express ``p(y_n|x_n)`` as a Bernoulli distribution in terms of ``\mu_n`` and ``y_n``.            (b) If furthermore is given that the data set is IID, show that the log-likelihood is given by

```math
L(\theta) \triangleq \log p(D|\theta) = \sum_n \left\{y_n \log \mu_n  + (1-y_n)\log(1-\mu_n)\right\}
```

(c) Prove that the derivative of the logistic function is given by

```math
\sigma^\prime(\xi) = \sigma(\xi)\cdot\left(1-\sigma(\xi)\right)
```

(d) Show that the derivative of the log-likelihood is

```math
\nabla_\theta L(\theta) = \sum_{n=1}^N \left( y_n - \sigma(\theta^T x_n +b)\right)x_n
```

(e) Design a gradient-ascent algorithm for maximizing ``L(\theta)`` with respect to ``\theta``.     

> (a)


```math
p(y_n|x_n) = p(y_n=1|x_n)^{y_n} p(y_n=0|x_n)^{1-y_n} = \mu_n^{y_n}(1-\mu_n)^{1-y_n}
```

> (b) The log-likelihood is given by


```math
\begin{align*} L(\theta) &= \log p(D|\theta) = \sum_n \log p(y_n|x_n,\theta)\\
&= \sum_n \left\{y_n \log \mu_n + (1-y_n)\log(1-\mu_n)\right\}
\end{align*}
```

> (c)


```math
\begin{align*}
\frac{d{}}{d{x}}\left( \frac{1}{1+e^{-x}}\right) &= \frac{(1+e^{-x})\cdot 0 - (-e^{-x}\cdot 1)}{(1+e^{-x})^2}\\
&= \frac{e^{-x}}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}}\cdot \frac{e^{-x}}{1+e^{-x}}\\
&=\sigma(x)\left( 1-\sigma(x)\right)
\end{align*}
```

> (d)


```math
\begin{align*}
\nabla_\theta L(\theta) &= \sum_n \frac{\partial{L}}{\partial{\mu_n}}\cdot \frac{\partial{\mu_n}}{\partial{(\theta^T x_n +b)}} \cdot \frac{\partial{(\theta^T x_n +b)}}{\partial{\theta}}\\
&= \sum_n  \left(\frac{y_n}{\mu_n} - \frac{1-y_n}{1-\mu_n} \right) \cdot \mu_n(1-\mu_n) \cdot x_n\\
&= \sum_n \frac{y_n - \mu_n}{\mu_n(1-\mu_n)} \cdot \mu_n(1-\mu_n) \cdot x_n\\
&= \sum_n (y_n - \mu_n) \cdot x_n
\end{align*}
```

> (e)


```math
 \theta^{(t+1)} = \theta^{(t)} + \rho \sum_n (y_n - \mu_n^{(t)})x_n
```

  * **[2]** Describe shortly the similarities and differences between the discriminative and generative approach to classification.

> Both aim to build an algorithm for ``p(y|x)`` where ``y`` is a discrete class label and ``x`` is a vector of real (or possibly discretely valued) variables. In the discriminative approach, we propose a model ``p(y|x,\theta)`` and use a training data set ``D=\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\}`` to infer good values for the parameters. For instance, in a maximum likelihood setting, we choose the parameters ``\hat{\theta}`` that maximize ``p(D|\theta)``. The classification algorithm is now given by


```math
p(y|x) = p(y|x,\hat{\theta})\,.
```

In the generative approach, we also aim to design an algorithm ``p(y|x)`` through a parametric model that is now given by ``p(y,x|\theta) = p(x|y,\theta)p(y|\theta)``. Again, we use the data set to train the parameters, eg, ``\hat{\theta} = \arg\max_\theta p(D|\theta)``, and the classification algorithm is now given by Bayes rule: 

>


```math
  p(y|x) \propto p(x|y,\hat{\theta})\cdot p(y|\hat{\theta})
  
```

  * **[3]** (Bishop ex.4.7) (#) Show that the logistic sigmoid function ``\sigma(a) = \frac{1}{1+\exp(-a)}`` satisfies the property ``\sigma(-a) = 1-\sigma(a)`` and that its inverse is given by ``\sigma^{-1}(y) = \log\{y/(1-y)\}``.

```math
\begin{align*} 
  1- \sigma(a) &= 1 - \frac{1}{1 + \exp(-a)} = \frac{1+\exp(-a) - 1}{1+\exp(-a)} \\
  &= \frac{\exp(-a)}{1+\exp(-a)} = \frac{1}{\exp(a)+1} = \sigma(-a)\end{align*}
```

> Regarding the inverse,


```math
\begin{align*} 
  y = \sigma(a) &= \frac{1}{1+\exp(-a)} \\
  \Rightarrow \frac1y - 1 &= \exp(-a) \\
  \Rightarrow \log\left( \frac{1-y}{y}\right) &= -a \\
  \Rightarrow \log\left( \frac{y}{1-y}\right) &= a = \sigma^{-1}(y)
\end{align*}
```



  * **[4]** (###) Let ``X`` be a real valued random variable with probability density

```math
p_X(x) = \frac{e^{-x^2/2}}{\sqrt{2\pi}},\quad\text{for all $x$}.
```

Also ``Y`` is a real valued random variable with conditional density

```math
p_{Y|X}(y|x) = \frac{e^{-(y-x)^2/2}}{\sqrt{2\pi}},\quad\text{for all $x$ and $y$}. 
```

(a) Give an (integral) expression for ``p_Y(y)``. Do not try to evaluate the integral.       (b) Approximate ``p_Y(y)`` using the Laplace approximation.  Give the detailed derivation, not just the answer. Hint: You may use the following results. Let 

```math
g(x) = \frac{e^{-x^2/2}}{\sqrt{2\pi}}
```

and

```math
h(x) = \frac{e^{-(y-x)^2/2}}{\sqrt{2\pi}}
```

for some real value ``y``. Then:

```math
\begin{align*}
\frac{\partial}{\partial x} g(x) &= -xg(x) \\
\frac{\partial^2}{\partial x^2} g(x) &= (x^2-1)g(x) \\
\frac{\partial}{\partial x} h(x) &= (y-x)h(x) \\
\frac{\partial^2}{\partial x^2} h(x) &= ((y-x)^2-1)h(x) 
\end{align*}
```

> (a)


```math
 p_Y(y) = \int_{-\infty}^{\infty} p_X(x)p_{Y|X}(y|x)\,dx =
\int_{-\infty}^{\infty} \frac{e^{-\frac12(x^2+(y-x)^2)}}{2\pi}\,dx
```

> (b) Using the hint we determine the first derivative of


```math
\begin{align*}
        f(x) &= g(x)h(x), \\
        \frac{\partial}{\partial x} f(x) &= \frac{\partial}{\partial x} g(x)\cdot h(x) = -xg(x)h(x)+g(x)(y-x)h(x) = (y-2x)f(x)
\end{align*}
```

> Setting this to zero gives


```math
\begin{align*}
        y-2x&= 0; \quad \text{so}\quad x=\frac12y. \\
        \frac{\partial}{\partial x} \ln f(x) &= \frac{\frac{\partial}{\partial x} f(x)}{f(x)} = (y-2x) \\
        \frac{\partial^2}{\partial x^2} \ln f(x) &= \frac{\partial}{\partial x} (y-2x) = -2.
\end{align*}
```

> So, we find ``A=2``, see lecture notes, and thus


```math
\begin{align*}
p_Y(y) &= \int_{-\infty}^{\infty}f(x)\,dx\approx f(\frac{y}{2})\sqrt{\frac{2\pi}{A}} \\
&= g(\frac{y}{2})h(\frac{y}{2})\sqrt{\frac{2\pi}{A}} \\
&= \frac{1}{\sqrt{2\pi\cdot2}}e^{-y^2/4}.
\end{align*}
```

> So ``Y`` is a Gaussian with mean ``m=0`` and variance ``\sigma^2=2``.


"""

# ╔═╡ Cell order:
# ╟─dad449de-6e1b-11f0-122f-5ffd7030f2f9
# ╟─dad46bf0-6e1b-11f0-1d60-0323d5d7c1ad
# ╟─dad48d74-6e1b-11f0-1c40-3dffee205c13
# ╟─dad53d96-6e1b-11f0-0751-ff28377e58cc
