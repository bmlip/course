### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 943f2554-6e1b-11f0-3bc0-35f9414b2d0f
md"""
# Probability Theory Review

  * **[1]** (a) (#) Proof that the "elementary" sum rule ``p(A) + p(\bar{A}) = 1`` follows from the (general) sum rule

```math
p(A+B) = p(A) + p(B) - p(A,B)\,.
```

(b) (###) Conversely, derive the general sum rule     ``p(A + B) = p(A) + p(B) - p(A,B)`` from the elementary sum rule ``p(A) + p(\bar A) = 1`` and the product rule. Here, you may make use of the (Boolean logic) fact that ``A + B = \overline {\bar A \bar B }``.      

```math
\begin{align*}
  p\left( {A + B} \right)  &\underset{\mathrm{bool}}{=}  p\left( {\overline {\bar A \bar B } } \right) \\
    &\underset{\mathrm{sum}}{=} 1 - p\left( {\bar A \bar B } \right) \\
    &\underset{\mathrm{prod}}{=} 1 - p\left( {\bar A |\bar B } \right)p\left( {\bar B } \right) \\
    &\underset{\mathrm{sum}}{=} 1 - \left( {1 - p\left( {A|\bar B } \right)} \right)\left( {1 - p\left( B \right)} \right) \\
    &= p(B) + \left( {1 - p\left( B \right)} \right)p\left( {A|\bar B } \right)  \\
    &\underset{\mathrm{prod}}{=} p(B) + \left( {1 - p\left( B \right)} \right)p\left( {\bar B |A} \right)\frac{{p\left( A \right)}}
{{p\left( {\bar B } \right)}} \\
    &\underset{\mathrm{sum}}{=} p(B) + p\left( {\bar B |A} \right)p\left( A \right) \\
    &\underset{\mathrm{sum}}{=} p(B) + \left( {1 - p\left( {B|A} \right)} \right)p\left( A \right)  \\
    &\underset{\mathrm{sum}}{=} p\left( A \right) + p(B) - p\left( {A,B} \right) 
\end{align*}
```

Note that, aside from the first boolean rewrite, everything follows straight application of sum and product rules. 

  * **[2]** Box 1 contains 8 apples and 4 oranges. Box 2 contains 10 apples and 2 oranges. Boxes are chosen with equal probability.      (a) (#) What is the probability of choosing an apple?         (b) (##) If an apple is chosen, what is the probability that it came from box 1?

> The following probabilities are given in the problem statement,


```math
\begin{align*}
p(b_1) &= p(b_2) = 1/2\\
p(a|b_1) &= 8/12,  \quad p(a|b_2)=10/12\\
p(o|b_1) &= 4/12,  \quad p(o|b_2)=2/12
\end{align*}
```

> (a) ``p(a) = \sum_i p(a,b_i) = \sum_i p(a|b_i)p(b_i)=\frac{8}{12}\cdot\frac{1}{2} + \frac{10}{12}\cdot\frac{1}{2} = \frac{3}{4}``        (b) ``p(b_1|a) = \frac{p(a,b_1)}{p(a)} = \frac{p(a|b_1)p(b_1)}{p(a)} = \frac{\frac{8}{12}\cdot\frac{1}{2}}{\frac{3}{4}} = \frac{4}{9}``


  * **[3]** (###) The inhabitants of an island tell the truth one third of the time. They lie with probability ``2/3``. On an occasion, after one of them made a statement, you ask another "was that statement true?" and he says "yes". What is the probability that the statement was indeed true?

> We use variables ``S_1 \in \{\text{t},\text{f}\}`` and ``S_2 \in \{\text{y},\text{n}\}`` for statements 1 and 2 and shorthand "y", "n", "t" and "f" for "yes", "no", "true and "false", respectively. The problem statement provides us with the following probabilities,


```math
\begin{align*}
p(S_1=\text{t})&= 1/3\\
p(S_1=\text{f})&= 1 - p(S_1=\text{t})= 2/3\\
p(S_2=\text{y}|S_1=\text{t})&= 1/3 \\
p(S_2=\text{y}|S_1=\text{f})&= 2/3
\end{align*}
```

We are asked to compute ``p(S_1=\text{t}|S_2=\text{y})``. Use Bayes rule,

```math
\begin{align*}
p(S_1=\text{t}|S_2=\text{y}) &= \frac{p(S_1=\text{t},S_2=\text{y})}{p(S_2=\text{y})}\\
&=\frac{\overbrace{p(S_2=\text{y}|S_1=\text{t})p(S_1=\text{t})}^{\text{both speak the truth}}}{\underbrace{p(S_2=\text{y}|S_1=\text{t})p(S_1=\text{t})}_{\text{both speak the truth}}+\underbrace{p(S_2=\text{y}|S_1=\text{f})p(S_1=\text{f})}_{\text{both lie}}}\\
&= \frac{\frac{1}{3}\cdot\frac{1}{3}}{\frac{1}{3}\cdot\frac{1}{3}+\frac{2}{3}\cdot\frac{2}{3}} = \frac{1}{5}
\end{align*}
```

  * **[4]** (##) A bag contains one ball, known to be either white or black. A white ball is put in, the bag is shaken, and a ball is drawn out, which proves to be white. What is now the chance of drawing a white ball? (Note that the state of the bag, after the operations, is exactly identical to its state before.)

> There are two hypotheses: let ``H = 0`` mean that the original ball in the bag was white and ``H = 1`` that is was black.


Assume the prior probabilities are equal. The data is that when a randomly selected ball was drawn from the bag, which contained a white one and the unknown one, it turned out to be white. The probability of this result according to each hypothesis is:

```math
 P(D|H =0) = 1,\quad P(D|H =1) = 1/2
```

So by Bayes theorem, 

```math
\begin{align*}
P(H=0|D) &= \frac{P(H=0,D)}{P(D)} \\
&= \frac{P(D|H=0) P(H=0)}{P(D|H=0) P(H=0) + P(D|H=1) P(H=1)} \\
&= \frac{1 \cdot \frac{1}{2}}{1 \cdot \frac{1}{2} + \frac{1}{2} \cdot \frac{1}{2}} \\
&= \frac{2}{3}
\end{align*}
```

and consequently,

```math
P(H =1|D) = 1 - P(H =0|D) = \frac{1}{3}\,.
```

  * **[5]**  A dark bag contains five red balls and seven green ones.       (a) (#) What is the probability of drawing a red ball on the first draw?       (b) (##) Balls are not returned to the bag after each draw. If you know that on the second draw the ball was a green one, what is now the probability of drawing a red ball on the first draw?

> (a) ``p(S_1=R) = \frac{N_R}{N_R+N_G}= \frac{5}{12}``         (b) The outcome of the ``n``th draw is referred to by variable ``S_n``. Use Bayes rule to get


```math
\begin{align*}
p(S_1=\text{R}|S_2=\text{G}) &=\frac{p(S_2=\text{G}|S_1=\text{R})p(S_1=\text{R})}{p(S_2=\text{G}|S_1=\text{R})p(S_1=\text{R})+p(S_2=\text{G}|S_1=\text{G})p(S_1=\text{G})}\\
&= \frac{\frac{7}{11}\cdot\frac{5}{12}}{\frac{7}{11}\cdot\frac{5}{12}+\frac{6}{11}\cdot\frac{7}{12}} = \frac{5}{11}
\end{align*}
```

  * **[6]**  (#) Is it more correct to speak about the likelihood of a *model* (or model parameters) than about the likelihood of an *observed data set*. And why?

> When a data generating distribution is considered as a function of the model *parameters* for given data, i.e. ``L(\theta) \triangleq \log p(D|\theta)``, it is called a likelihood. It is more correct to speak about the likelihood of a *model* (or of the likelihood of the parameters).


  * **[7]** (##) Is a speech signal a 'probabilistic' (random) or a deterministic signal?

> That depends. The term 'probabilistic' refers to a state-of-knowledge (or beliefs) about something (in this case, about the values of a speech signal). The fundamental issue here is to realize that the signal itself is not probabilistic (nor deterministic), but rather that these attributes reflect a state-of-knowledge. If you had a perfect microphone and recorded a speech signal perfectly at its source, then you would know all the signal values perfectly. You could say that the signal is deterministic since there is no uncertainty. However, before you would record the signal, how would you describe your state-of-knowledge about the signal values that your are going to record? There is uncertainty, so you would need to describe that speech signal by a probability distribution over all possible values.


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

> Define ``A = [I, I]``, ``w = [x;y]`` (where the notation ";" stacks the columns of ``x`` and ``y`` and ``I`` is the identity matrix). Then ``z = A w``. Now apply the formula for the mean and variance of a RV after a linear transformation.


```math
\begin{align*}
\mathbb{E}[z] &= \mathbb{E}[Aw] \\ 
    &= \mathbb{E}[x+y] \\
    &= \mathbb{E}[x]+\mathbb{E}[y]\\
    \mathbb{V}[z] &= \mathbb{V}[Aw] \\
    &= A\mathbb{V}[w]A^T \\
    &= \begin{bmatrix}I & I \end{bmatrix}\begin{bmatrix} \mathbb{V}[x] & \mathbb{V}[x,y] \\ \mathbb{V}[x,y] & \mathbb{V}[y]\end{bmatrix}\begin{bmatrix}I \\ I \end{bmatrix} \\
    &= \mathbb{V}[x]+\mathbb{V}[y]+2\mathbb{V}[x,y]
\end{align*}
```

"""

# ╔═╡ Cell order:
# ╟─943f2554-6e1b-11f0-3bc0-35f9414b2d0f
