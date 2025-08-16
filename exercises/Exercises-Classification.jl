### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 87bbaf50-6e1b-11f0-1f57-114d7462bd47
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

The prior probabilities ``p(C_1) = 0.6`` and ``p(C_2) = 0.4`` are also known from experience.          (a) (##) A "Bayes Classifier" is given by

```math
 \text{Decision} = \begin{cases} C_1 & \text{if } p(C_1|x)>p(C_2|x) \\
                               C_2 & \text{otherwise}
                 \end{cases}
```

Derive the optimal Bayes classifier.         (b) (###) The probability of making the wrong decision, given ``x``, is

```math
p(\text{error}|x)= p(C_1|x,\text{we-decide-}C_2) +  p(C_2|x,\text{we-decide-}C_1)
```

Compute the **total** error probability  ``p(\text{error})`` for the Bayes classifier in this example. 

  * **[2]** (#) (see Bishop exercise 4.8): Using (4.57) and (4.58) (from Bishop's book), derive the result (4.65) for the posterior class probability in the two-class generative model with Gaussian densities, and verify the results (4.66) and (4.67) for the parameters ``w`` and ``w0``.

  * **[3]** (##) (see Bishop exercise 4.10).



"""

# ╔═╡ 87bc9a8c-6e1b-11f0-0db9-c54b6e718575
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

  * **[2]** Describe shortly the similarities and differences between the discriminative and generative approach to classification.

  * **[3]** (Bishop ex.4.7) (#) Show that the logistic sigmoid function ``\sigma(a) = \frac{1}{1+\exp(-a)}`` satisfies the property ``\sigma(-a) = 1-\sigma(a)`` and that its inverse is given by ``\sigma^{-1}(y) = \log\{y/(1-y)\}``.



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

"""

# ╔═╡ Cell order:
# ╟─87bbaf50-6e1b-11f0-1f57-114d7462bd47
# ╟─87bc9a8c-6e1b-11f0-0db9-c54b6e718575
