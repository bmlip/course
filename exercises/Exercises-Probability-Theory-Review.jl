### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 8f3201ee-6e1b-11f0-1c9d-41d394d82d7a
md"""
# Probability Theory Review

  * **[1]** (a) (#) Proof that the "elementary" sum rule ``p(A) + p(\bar{A}) = 1`` follows from the (general) sum rule

```math
p(A+B) = p(A) + p(B) - p(A,B)\,.
```

(b) (###) Conversely, derive the general sum rule     ``p(A + B) = p(A) + p(B) - p(A,B)`` from the elementary sum rule ``p(A) + p(\bar A) = 1`` and the product rule. Here, you may make use of the (Boolean logic) fact that ``A + B = \overline {\bar A \bar B }``.      

  * **[2]** Box 1 contains 8 apples and 4 oranges. Box 2 contains 10 apples and 2 oranges. Boxes are chosen with equal probability.      (a) (#) What is the probability of choosing an apple?         (b) (##) If an apple is chosen, what is the probability that it came from box 1?

  * **[3]** (###) The inhabitants of an island tell the truth one third of the time. They lie with probability ``2/3``. On an occasion, after one of them made a statement, you ask another "was that statement true?" and he says "yes". What is the probability that the statement was indeed true?

  * **[4]** (##) A bag contains one ball, known to be either white or black. A white ball is put in, the bag is shaken, and a ball is drawn out, which proves to be white. What is now the chance of drawing a white ball? (Note that the state of the bag, after the operations, is exactly identical to its state before.)

  * **[5]**  A dark bag contains five red balls and seven green ones.       (a) (#) What is the probability of drawing a red ball on the first draw?       (b) (##) Balls are not returned to the bag after each draw. If you know that on the second draw the ball was a green one, what is now the probability of drawing a red ball on the first draw?

  * **[6]**  (#) Is it more correct to speak about the likelihood of a *model* (or model parameters) than about the likelihood of an *observed data set*. And why?

  * **[7]** (##) Is a speech signal a 'probabilistic' (random) or a deterministic signal?

  * **[8]** (##) $(HTML("<span id='distribution-of-sum'>Proof</span>")) that, for any distribution of ``x`` and ``y`` and ``z=x+y``

```math
\begin{align*}
    \mathbb{E}[z] &= \mathbb{E}[x] + \mathbb{E}[y] \\
    \mathbb{V}[z] &= \mathbb{V}[x] + \mathbb{V}[y] + 2\mathbb{V}[x,y] 
\end{align*}
```

where ``\mathbb{E}[\cdot]``, ``\mathbb{V}[\cdot]`` and ``\mathbb{V}[\cdot,\cdot]`` refer to the expectation (mean), variance and covariance operators respectively. You may make use of the more general theorem that the mean and variance of any distribution ``p(x)`` is processed by a linear tranformation as

```math
\begin{align*}
\mathbb{E}[Ax +b] &= A\mathbb{E}[x] + b \\
\mathbb{V}[Ax +b] &= A\,\mathbb{V}[x]\,A^T 
\end{align*}
```

"""

# ╔═╡ Cell order:
# ╟─8f3201ee-6e1b-11f0-1c9d-41d394d82d7a
