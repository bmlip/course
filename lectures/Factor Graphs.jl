### A Pluto.jl notebook ###
# v0.20.13

#> [frontmatter]
#> image = "https://github.com/bmlip/course/blob/v2/assets/figures/ffg-example-1.png?raw=true"
#> description = "Introduction to Forney-style factor graphs and message passing-based inference."
#> 
#>     [[frontmatter.author]]
#>     name = "BMLIP"
#>     url = "https://github.com/bmlip"

using Markdown
using InteractiveUtils

# ╔═╡ 965a08f4-d294-11ef-0604-1586ff37c0d4
using Plots, LinearAlgebra, LaTeXStrings

# ╔═╡ 2cb7d369-e7fd-4d66-8321-66a9197a26bd
using RxInfer, Random

# ╔═╡ 5a8dcadb-f0c2-4fb0-b8cd-db8cf49cc292
using PlutoUI, PlutoTeachingTools

# ╔═╡ 981b08cc-7fb4-4880-8e8a-0b60a5dd72a2
using HypertextLiteral

# ╔═╡ 96547560-d294-11ef-0fa7-6b6489f7baba
md"""
# Factor Graphs

"""

# ╔═╡ af24aa27-b0a1-4c9b-aee0-0e5143d2f47e
PlutoUI.TableOfContents()

# ╔═╡ 9654ea3e-d294-11ef-335c-657af1ceaf19
md"""
## Preliminaries

##### Goal 

  * Introduction to Forney-style factor graphs and message passing-based inference

##### Materials        

  * Mandatory

      * These lecture notes
      * Loeliger (2007), [The factor graph approach to model based signal processing](https://github.com/bmlip/course/blob/main/assets/files/Loeliger-2007-The-factor-graph-approach-to-model-based-signal-processing.pdf), pp. 1295-1302 (until section V)
  * Optional

      * Frederico Wadehn (2015), [Probabilistic graphical models: Factor graphs and more](https://www.youtube.com/watch?v=Fv2YbVg9Frc&t=31) video lecture (**recommended**)
  * References

      * Forney (2001), [Codes on graphs: normal realizations](https://github.com/bmlip/course/blob/main/assets/files/Forney-2001-Codes-on-graphs-normal-realizations.pdf)

"""

# ╔═╡ 96552348-d294-11ef-16d8-b53563054687
md"""
## Why Factor Graphs?


A probabilistic inference task derives most of its computational complexity from the need to perform marginalization, i.e., integrating (or summing) over latent or nuisance variables. 

For example, for a model ``p(x_1,x_2,x_3,x_4,x_5)``, the inference task ``p(x_2|x_3)`` involves computing

```math
p(x_2|x_3) = \frac{p(x_2,x_3)}{p(x_3)} = \frac{\int \cdots \int p(x_1,x_2,x_3,x_4,x_5) \, \mathrm{d}x_1  \mathrm{d}x_4 \mathrm{d}x_5}{\int \cdots \int p(x_1,x_2,x_3,x_4,x_5) \, \mathrm{d}x_1  \mathrm{d}x_2 \mathrm{d}x_4 \mathrm{d}x_5}
```

Because marginalization operations (sums or integrals) scale poorly with the number of variables, a phenomenon known as the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality), we must often leverage the model’s conditional independence structure to reduce computational complexity and obtain feasible inference procedures.

Factor graphs provide a computationally efficient approach to solving inference problems **if the probabilistic model can be factorized**. 

"""


# ╔═╡ 9655959e-d294-11ef-0ca6-5f20aa579e91
md"""
 $(HTML("<span id='factorization-helps'>Factorization helps.</span>")) For instance, if ``p(x_1,x_2,x_3,x_4,x_5) = p(x_1)p(x_2,x_3)p(x_4)p(x_5|x_4)``, then

```math
\begin{align}
p(x_2|x_3) &= \frac{\int \cdots \int p(x_1)p(x_2,x_3)p(x_4)p(x_5|x_4) \, \mathrm{d}x_1  \mathrm{d}x_4 \mathrm{d}x_5}{\int \cdots \int p(x_1)p(x_2,x_3)p(x_4)p(x_5|x_4) \, \mathrm{d}x_1  \mathrm{d}x_2 \mathrm{d}x_4 \mathrm{d}x_5} \\
  &= \frac{p(x_2,x_3)}{\int p(x_2,x_3) \mathrm{d}x_2}
\end{align}
```

which is computationally much cheaper than the general case above.

In this lesson, we discuss how computationally efficient inference in *factorized* probability distributions can be automated by message passing-based inference in factor graphs.

"""

# ╔═╡ 05db1eab-5b63-4ab9-8b4a-ab2cde554295
md"""
# Forney-style Factor Graphs
"""

# ╔═╡ 9655b2c2-d294-11ef-057f-9b3984064411
md"""
## Factor Graph Construction Rules

Consider a function 

```math
f(x_1,x_2,x_3,x_4,x_5) = f_a(x_1,x_2,x_3) \cdot f_b(x_3,x_4,x_5) \cdot f_c(x_4)
```

"""

# ╔═╡ 9655c1ae-d294-11ef-061a-991947cee620
md"""
The factorization of this function can be graphically represented by a **Forney-style Factor Graph** (FFG):

![](https://github.com/bmlip/course/blob/v2/assets/figures/ffg-example-1.png?raw=true)

"""

# ╔═╡ 9655d360-d294-11ef-0f06-ab58e2ad0e5f
md"""
An FFG is an **undirected** graph subject to the following construction rules ([Forney, 2001](https://github.com/bmlip/course/blob/main/assets/files/Forney-2001-Codes-on-graphs-normal-realizations.pdf))

1. A **node** for every factor;
2. An **edge** (or **half-edge**) for every variable;
3. Node ``f_\bullet`` is connected to edge ``x`` **iff** variable ``x`` appears in factor ``f_\bullet``.

"""

# ╔═╡ 9655e06c-d294-11ef-0393-9355d6e20afb
md"""
A **configuration** is an assignment of values to all variables. A configuration ``\omega=(x_1,x_2,x_3,x_4,x_5)`` is said to be **valid** iff ``f(\omega) \neq 0``

"""

# ╔═╡ 9655ed6e-d294-11ef-370f-937b590036f3
md"""
## Equality Nodes for Branching Points

Note that a variable can appear in maximally two factors in an FFG (since an edge has only two end points).

"""

# ╔═╡ 9655fb88-d294-11ef-1ceb-91585012d142
md"""
Consider the factorization (where ``x_2`` appears in three factors) 

```math
 f(x_1,x_2,x_3,x_4) = f_a(x_1,x_2)\cdot f_b(x_2,x_3) \cdot f_c(x_2,x_4)
```

"""

# ╔═╡ 965606f2-d294-11ef-305b-870427879e50
md"""
For the factor graph representation, we will instead consider the function ``g``, defined as

```math
\begin{align*}
 g(x_1,x_2&,x_2^\prime,x_2^{\prime\prime},x_3,x_4) 
  = f_a(x_1,x_2)\cdot f_b(x_2^\prime,x_3) \cdot f_c(x_2^{\prime\prime},x_4) \cdot f_=(x_2,x_2^\prime,x_2^{\prime\prime})\,,
\end{align*}
```


"""

# ╔═╡ 2d5cd42e-26ea-4ee4-979f-3a4e2c6271b6
TwoColumn(
md"""
where 
```math
f_=(x_2,x_2^\prime,x_2^{\prime\prime}) \triangleq \delta(x_2-x_2^\prime)\, \delta(x_2-x_2^{\prime\prime})
```
is a so-called **equality** (or branching) node. 
""", 
md"""
![](https://github.com/bmlip/course/blob/v2/assets/figures/ffg-wEquality-node.png?raw=true)
""")

# ╔═╡ 96561594-d294-11ef-1590-198382927808
md"""
Note that through introduction of auxiliary variables ``x_2^{\prime}`` and ``x_2^{\prime\prime}`` and a factor ``f_=(x_2,x_2^\prime,x_2^{\prime\prime})``, each variable in ``g`` appears in maximally two factors.

The constraint ``f_=(x,x^\prime,x^{\prime\prime})`` enforces that ``x=x^\prime=x^{\prime\prime}`` **for every valid configuration**.

Since ``f`` is a marginal of ``g``, i.e., 

```math
f(x_1,x_2,x_3,x_4) = \iint g(x_1,x_2,x_2^\prime,x_2^{\prime\prime},x_3,x_4)\, \mathrm{d}x_2^\prime \mathrm{d}x_2^{\prime\prime}
```

it follows that any inference problem on ``f`` can be executed by a corresponding inference problem on ``g``, e.g.,

```math
\begin{align*}
f(x_1 \mid x_2) &\triangleq \frac{\iint f(x_1,x_2,x_3,x_4) \,\mathrm{d}x_3 \mathrm{d}x_4 }{ \int\cdots\int f(x_1,x_2,x_3,x_4) \,\mathrm{d}x_1 \mathrm{d}x_3 \mathrm{d}x_4} \\
  &= \frac{\int\cdots\int g(x_1,x_2,x_2^\prime,x_2^{\prime\prime},x_3,x_4) \,\mathrm{d}x_2^\prime \mathrm{d}x_2^{\prime\prime} \mathrm{d}x_3 \mathrm{d}x_4 }{ \int\cdots\int g(x_1,x_2,x_2^\prime,x_2^{\prime\prime},x_3,x_4) \,\mathrm{d}x_1 \mathrm{d}x_2^\prime \mathrm{d}x_2^{\prime\prime} \mathrm{d}x_3 \mathrm{d}x_4} \\
  &= g(x_1 \mid x_2)
\end{align*}
```
"""

# ╔═╡ 9656cf72-d294-11ef-03aa-b715dd686c09
md"""
## Probabilistic Models as Factor Graphs

FFGs can be used to express conditional independence (factorization) in probabilistic models. 


For example, the (previously shown) graph for 

```math
f_a(x_1,x_2,x_3) \cdot f_b(x_3,x_4,x_5) \cdot f_c(x_4)
```

could represent the probabilistic model

```math
p(x_1,x_2,x_3,x_4,x_5) = p(x_1,x_2|x_3) \cdot p(x_3,x_5|x_4) \cdot p(x_4)
```

"""

# ╔═╡ 27b9e811-4542-4fe8-86a5-f8ba09074761
TwoColumn(
md"""
where we identify 
```math
\begin{align*}
f_a(x_1,x_2,x_3) &= p(x_1,x_2|x_3) \\
f_b(x_3,x_4,x_5) &= p(x_3,x_5|x_4) \\
f_c(x_4) &= p(x_4)
\end{align*}
```
""", 
@htl """

<img src="https://github.com/bmlip/course/blob/v2/assets/figures/ffg-example-prob-model.png?raw=true" alt=" " style="display: block; width: 100%; margin: 0 auto;">

""")

# ╔═╡ 9656d850-d294-11ef-21a1-474b07ea7729
md"""
This factorized probability distribution is represented by the above FFG. 

"""

# ╔═╡ 9658329c-d294-11ef-0d03-45e6872c4985
md"""
## Terminating an FGG

Consider a model 

```math
f(x_1,x_2,y) = f_a(x_1) f_b(x_1,x_2,y) \,.
```

In this model, the variables ``x_2`` and ``y`` appear in only one factor. In the corresponding FFG, edges that only connect to one factor are called a **half-edges**. Half-edges typically represent inputs or outputs of the graph, such as observed variables and external control signals.

In general, any half-edge can be terminated by a node ``f(\cdot) = 1``, since the model

```math
f_a(x_1) f_b(x_1,x_2,y) \underbrace{f_c(y)}_{=1} \underbrace{f_d(x_2)}_{=1}\,,
```
is the same model as ``f_a(x_1) f_b(x_1,x_2,y)``.


![](https://github.com/bmlip/course/blob/v2/assets/figures/ffg-terminal-node.png?raw=true)

An FFG without half-edges is called a Terminated FFG (TFFG).   

"""

# ╔═╡ f0181b53-a604-489f-a89e-db6fc58571dd
md"""
## Representing Observations

An observation, say ``y=3``, can be represented by a **delta node** ``f(y)=\delta(y−3)`` to terminate the half-edge for variable ``y``.

In an FFG, we visualize a delta node by a small black box,

![](https://github.com/bmlip/course/blob/v2/assets/figures/ffg-observation-y-3.png?raw=true)

where we identify 
```math
\begin{align*}
f_a(x_1,x_2,x_3) &= p(x_1,x_2|x_3) \\
f_b(x_3,x_4,x_5) &= p(x_3,x_5|x_4) \\
f_c(x_4) &= p(x_4)
\end{align*}
```

"""

# ╔═╡ ea4a720f-a644-46a0-ad35-b215780e0928
keyconcept("",md"Any factorized probabilistic model, including a set of observations for that model, can be represented by a Terminated Forney-style factor graph.")

# ╔═╡ 00c69a22-feb5-4d1e-9ab5-a136435d7d22
md"""
# Message Passing-based Inference
"""

# ╔═╡ 9656e606-d294-11ef-1daa-312623552a5b
md"""
## Inference in Factorized Models

Factorizations offer opportunities to reduce the computational cost of inference by exploiting the conditional independence structure of the model.  

"""

# ╔═╡ 9656ee62-d294-11ef-38f4-7bc8031df7ee
md"""
Assume we wish to compute the marginal

```math
\bar{f}(x_3) \triangleq \sum\limits_{x_1,x_2,x_4,x_5,x_6,x_7}f(x_1,x_2,\ldots,x_7) 
```

for a model ``f`` with given factorization 

```math
f(x_1,x_2,\ldots,x_7) = f_a(x_1) f_b(x_2) f_c(x_1,x_2,x_3) f_d(x_4) f_e(x_3,x_4,x_5) f_f(x_5,x_6,x_7) f_g(x_7)
```

"""

# ╔═╡ 9656fae2-d294-11ef-10d8-ff921d5956bd
md"""
Note that, if each variable ``x_i`` can take on ``10`` values, then computing the marginal ``\bar{f}(x_3)`` takes about ``10^6`` (=``1`` million) additions. 

"""

# ╔═╡ b33b2aef-e672-490c-bdf4-a5f655fa4695
md"""
We draw here the FFG for the factorized distribution:

![](https://github.com/bertdv/BMLIP/blob/master/lessons/notebooks/figures/ffg-message-passing.png?raw=true)


"""

# ╔═╡ cb3df230-6c7e-41b9-ba13-3c5f8a7fbb62
md"""
Note that we drew *directed edges* to distinguish between intermediate results ("messages") ``\overrightarrow{\mu}_\bullet(\cdot)`` that flow in the same direction as the arrow of the edge (later to be called: forward messages) from intermediate results ``\overleftarrow{\mu}_\bullet(\cdot)`` that flow in opposite direction (later to be called: backward messages).  This is just a notational convenience since an FFG is computationally an undirected graph. For now, only consider the nodes and edges. The messages ``\overrightarrow{\mu}_\bullet(\cdot)`` will be discussed later. 

"""

# ╔═╡ 96570d3e-d294-11ef-0178-c34dda717495
md"""
Due to the factorization of ``f(x_1,x_2,\ldots,x_7)`` and the [Generalized Distributive Law](https://en.wikipedia.org/wiki/Generalized_distributive_law), we can decompose the marginalization operation to the following product-of-sums:

```math
\begin{align*}
\bar{f}(x_3) = 
  &\underbrace{ \Bigg( \sum_{x_1,x_2} \underbrace{f_a(x_1)}_{\overrightarrow{\mu}_{X_1}(x_1)}\, \underbrace{f_b(x_2)}_{\overrightarrow{\mu}_{X_2}(x_2)}\,f_c(x_1,x_2,x_3)\Bigg) }_{\overrightarrow{\mu}_{X_3}(x_3)} \\
  &\quad\underbrace{ \cdot\Bigg( \sum_{x_4,x_5} \underbrace{f_d(x_4)}_{\overrightarrow{\mu}_{X_4}(x_4)}\,f_e(x_3,x_4,x_5) \cdot \underbrace{ \big( \sum_{x_6,x_7} f_f(x_5,x_6,x_7)\,\underbrace{f_g(x_7)}_{\overleftarrow{\mu}_{X_7}(x_7)}\big) }_{\overleftarrow{\mu}_{X_5}(x_5)} \Bigg) }_{\overleftarrow{\mu}_{X_3}(x_3)}
\end{align*}
```

which, in case ``x_i`` has ``10`` values, requires a few hundred additions and is therefore computationally (much!) lighter than executing the full sum ``\sum_{x_1,\ldots,x_7}f(x_1,x_2,\ldots,x_7)``


"""

# ╔═╡ 9657b088-d294-11ef-3017-e95c4c69b62b
md"""
As an afterthought, note that applying the distributive law in an FFG for inference is analogous to replacing the sum-of-products

```math
ac + ad + bc + bd
```

by the following product-of-sums:

```math
(a + b)(c + d) \,.
```

Which of these two computations is cheaper to execute?

"""

# ╔═╡ 0afe3cdc-15ed-4d9a-848a-d1977d051866
md"""
## Closing-the-Box and Message Passing Interpretations
"""

# ╔═╡ 96571c34-d294-11ef-11ef-29beeb1f96c2
md"""
Note that the intermediate result ``\overrightarrow{\mu}_{X_3}(x_3)`` is obtained by multiplying all enclosed factors (``f_a``, ``f_b, f_c``) by the red dashed box, followed by marginalization (summing) over all enclosed variables (``x_1``, ``x_2``),

```math
\overrightarrow{\mu}_{X_3}(x_3) = \underbrace{\sum_{x_1}\sum_{x_2}}_{\text{enclosed variables}}\underbrace{f_a(x_1)f_b(x_2) f_c(x_1,x_2,x_3) }_{\text{enclosed factors}}
```

This operation is known as **Closing-the-Box**. The result is a new **composite node** that holds the factor ``\overrightarrow{\mu}_{X_3}(x_3)``, and is visually represented by the red dashed box in the factor graph. The composite node ``\overrightarrow{\mu}_{X_3}(x_3)`` depends only on the variable(s) that cross the boundary of the box (in this case ``x_3``) and effectively replaces the internal subgraph contained within the red box.
"""

# ╔═╡ 253d4703-03d6-4961-8c3b-b70d2cbc0710
md"""
When closing the box around a terminal node, the result is simply the factor associated with that node, since there are no internal variables that need to be marginalized out.
"""

# ╔═╡ a7b1f559-3c34-491e-83e7-ba95c8c22c80
md"""

The Closing-the-box operation can alternatively be interpreted as **passing a message** from the newly created composite node to the rest of the graph. For instance, ``\overrightarrow{\mu}_{X_3}(x_3)`` can be understood in two equivalent ways: 
  * as a factor associated with the composite node that encloses the subgraph inside the red box.
  * as a message sent from this composite node to the variable ``x_3``. 

In both interpretations, the internal details of the subgraph are abstracted away, and the composite node effectively summarizes its contribution to the overall inference process.


"""

# ╔═╡ 70736e62-2b6c-4b3a-ab59-7e51522d620b
md"""

The complete inference process for computing ``\bar{f}(x_3)`` can be interpreted as a **message passing process**. It begins by sending messages from the terminal nodes and proceeds by propagating messages through the internal nodes of the factor graph. This continues until both the forward and backward messages for ``x_3`` have been computed. The final result, ``\bar{f}(x_3)``, is obtained by multiplying the forward and backward messages,

```math
\bar{f}(x_3) = \overrightarrow{\mu}_{X_3}(x_3) \cdot \overleftarrow{\mu}_{X_3}(x_3)
```

This message-based interpretation enables modular, local inference that scales efficiently with the structure of the factor graph.


"""


# ╔═╡ 96575dd4-d294-11ef-31d6-b39b4c4bdea1
md"""
## Sum-Product Messages

Let's continue with the message passing interpretation of inference in an FFG. Closing the red box around ``f_a``, ``f_b`` and ``f_c`` leads to an outgoing message ``\overrightarrow{\mu}_{X_3}(x_3)`` for node ``f_c``, given by

```math
\begin{align}
\underbrace{\overrightarrow{\mu}_{X_3}(x_3)}_{\substack{ \text{outgoing} \\ \text{message} }} 
&= \sum_{x_1}\sum_{x_2} \underbrace{\overrightarrow{\mu}_{X_1}(x_1) \overrightarrow{\mu}_{X_2}(x_2)}_{\substack{\text{incoming} \\ \text{messages}}} \underbrace{f_c(x_1,x_2,x_3)}_{\text{factor}} 
\end{align}
```

This recipe holds generally. For a node ``f(y,x_1,\ldots,x_n)`` with incoming messages ``\overrightarrow{\mu}_{X_1}(x_1), \overrightarrow{\mu}_{X_1}(x_1), \ldots,\overrightarrow{\mu}_{X_n}(x_n)``, the outgoing message is given by ([Loeliger (2007), pg.1299](https://github.com/bmlip/course/blob/main/assets/files/Loeliger-2007-The-factor-graph-approach-to-model-based-signal-processing.pdf)): 

```math
\underbrace{\overrightarrow{\mu}_{Y}(y)}_{\substack{ \text{outgoing}\\ \text{message}}} = \sum_{x_1,\ldots,x_n} \underbrace{\overrightarrow{\mu}_{X_1}(x_1)\cdots \overrightarrow{\mu}_{X_n}(x_n)}_{\substack{\text{incoming} \\ \text{messages}}} \cdot \underbrace{f(y,x_1,\ldots,x_n)}_{\substack{\text{node}\\ \text{function}}} \tag{SP}
```

"""

# ╔═╡ 5cc2016e-0383-448c-bd33-5b3a687b7436
TwoColumn(
md"""
Equation (SP) is called a **Sum-Product** message, so named because the computation involves evaluating a sum-of-products. Note that all SP messages in an FFG can be computed from information that is **locally available** at each node.
""", 
@htl """

<img src="https://github.com/bmlip/course/blob/main/assets/figures/ffg-sum-product.png?raw=true" alt=" " style="display: block; width: 100%; margin: 0 auto;">

""")

# ╔═╡ f65f5d0e-2583-4b88-b9f2-5fee15257c05
md"""


If the factor graph for the whole model has no cycles, i.e., the FFG is a tree, then the process of passing SP message from the terminal nodes to the internal (latent) variables yields exact Bayesian marginals for all hidden variables. This inference method is known as the **Sum-Product** (SP) algorithm.

However, if the graph contains cycles, one can conceptually view the graph as an infinite tree by “unrolling” the cycles. In this loopy setting, SP-based inference is not guaranteed to yield exact marginals. Nevertheless, in practice, if we run the SP algorithm for a limited number of iterations (i.e., a finite unrolling), we often obtain high-quality approximate marginals that are sufficient for many inference tasks.
"""

# ╔═╡ 91f81188-727c-4754-9a07-e754eef8bbe0
md"""
## Example: Sum-Product Messages for the Equality Node
"""

# ╔═╡ 0633afea-5e92-4bad-8402-d159c534af81
TwoColumn(
md"""
As an example, let´s evaluate the SP messages for the **equality node** 

```math 
f_=(x,y,z) = \delta(z-x)\delta(z-y) \,.
``` 
""", 
@htl """

<img src="https://github.com/bmlip/course/blob/v2/assets/figures/ffg-equality-node.png?raw=true" alt=" " style="display: block; width: 90%; margin: 0 auto;">

""")

# ╔═╡ f11564db-aafc-4df9-b494-4e5ced9bfcfe
md"""

Given incoming messages ``\overrightarrow{\mu}_{X}(x)`` and ``\overrightarrow{\mu}_{Y}(y)``, the outgoing SP message ``\overrightarrow{\mu}_{Z}(z)`` to edge ``z`` is given by
```math
\begin{align*}
\overrightarrow{\mu}_{Z}(z) &= \iint  \overrightarrow{\mu}_{X}(x) \overrightarrow{\mu}_{Y}(y) \,\delta(z-x)\delta(z-y) \,\mathrm{d}x \mathrm{d}y \\
   &=  \overrightarrow{\mu}_{X}(z)  \int  \overrightarrow{\mu}_{Y}(y) \,\delta(z-y) \,\mathrm{d}y \\
   &=  \overrightarrow{\mu}_{X}(z) \overrightarrow{\mu}_{Y}(z) 
\end{align*}
```

By symmetry, this also implies (for the same equality node) that

```math
\begin{align*}
\overleftarrow{\mu}_{X}(x) &= \overrightarrow{\mu}_{Y}(x) \overleftarrow{\mu}_{Z}(x) \\
\overleftarrow{\mu}_{Y}(y) &= \overrightarrow{\mu}_{X}(y) \overleftarrow{\mu}_{Z}(y)\,.
\end{align*}
```

It follows that message passing through an equality node is analogous to applying Bayes’ rule: two information sources are fused by multiplying their corresponding probability distributions.

"""

# ╔═╡ 9651f976-b834-4b81-8810-649f0290969d
md"""
# RxInfer: A Toolbox for Automated Bayesian inference
"""

# ╔═╡ 96587a66-d294-11ef-2c7a-9fd7bea76582
md"""
## Automating Bayesian Inference by Message Passing

The foregoing message update rules can be worked out in closed-form and put into tables (e.g., see Tables 1 through 6 in [Loeliger (2007)](https://github.com/bmlip/course/blob/main/assets/files/Loeliger-2007-The-factor-graph-approach-to-model-based-signal-processing.pdf) for many standard factors such as essential probability distributions and operations such as additions, fixed-gain multiplications and branching (equality nodes).

In the optional slides below, we have worked out a few more update rules for the [addition node](#sp-for-addition-node) and the [multiplication node](#sp-for-multiplication-node).

If the update rules for all node types in a graph have been tabulated, then inference by message passing comes down to executing a set of table-lookup operations, thus creating a completely **automatable Bayesian inference framework**. 

In our research lab [BIASlab](http://biaslab.org) (FLUX 7.060), we are developing [RxInfer](http://rxinfer.com), which is a (Julia) toolbox for automating Bayesian inference by message passing in a factor graph.

In general, a code package that automates Bayesian inference is called a [Probabilistic Programming](https://en.wikipedia.org/wiki/Probabilistic_programming) Language (PLL). RxInfer is a PLL that automates inference through message passing-based inference in a factor graph. 

"""

# ╔═╡ 89e2757e-a09f-40c6-8dd7-9b4b4d232e17
md"""
![](https://github.com/bmlip/course/blob/main/assets/figures/RxInfer-gif.gif?raw=true)
"""

# ╔═╡ c4b5b124-e52a-41fc-b27e-a58181622e5c
md"""
The figure above (a screen recording from the [RxInfer webpage](http://rxinfer.com)) is an animated GIF illustrating how RxInfer operates. The model is represented as a graph, where each node passes messages to its neighbors. When messages meet on an edge, the belief about the variable associated with that edge is updated.
"""

# ╔═╡ 96589eb0-d294-11ef-239a-2513a805cdcf
md"""
## Code Example: Bayesian Linear Regression by Message Passing

"""

# ╔═╡ 9658c106-d294-11ef-01db-cfcff611ed81
md"""
Assume we want to estimate some function ``f: \mathbb{R}^D \rightarrow \mathbb{R}`` from a given data set ``D = \{(x_1,y_1), \ldots, (x_N,y_N)\}``.

"""

# ╔═╡ 96594d44-d294-11ef-22b8-95165fb08ce4
md"""
#### model specification

We will assume a linear model with white Gaussian noise and a Gaussian prior on the coefficients ``w``:

```math
\begin{align*}
  y_i &= w^T x_i  + \epsilon_i \\
  \epsilon_i &\sim \mathcal{N}(0, \sigma^2) \\ 
  w &\sim \mathcal{N}(0,\Sigma)
\end{align*}
```

or equivalently

```math
\begin{align*}
p(w,\epsilon,D) &= \overbrace{p(w)}^{\text{weight prior}} \prod_{i=1}^N  \overbrace{p(y_i\,|\,x_i,w,\epsilon_i)}^{\text{data-generating}} \overbrace{p(\epsilon_i)}^{\text{noise prior}} \\
  &= \mathcal{N}(w\,|\,0,\Sigma) \prod_{i=1}^N \delta(y_i - w^T x_i - \epsilon_i) \mathcal{N}(\epsilon_i\,|\,0,\sigma^2) 
\end{align*}
```

"""

# ╔═╡ 96597ce0-d294-11ef-3478-25c6bbef601e
md"""
#### Inference (parameter estimation)

We are interested in inferring the posterior ``p(w|D)``. We will execute inference by message passing on the FFG for the model.

"""

# ╔═╡ 965998a8-d294-11ef-1d18-85876e3656c5
md"""
The left figure shows the factor graph for this model for one observation ``(x,y)``. The figure on the right shows the message passing scheme. 

![](https://github.com/bmlip/course/blob/v2/assets/figures/ffg-bayesian-linear-regression.png?raw=true)

"""

# ╔═╡ 9659ab66-d294-11ef-027a-d3f7206050af
md"""
#### Inference Execution (by RxInfer)

Let's solve this problem by message passing-based inference with Julia's FFG toolbox [RxInfer](https://biaslab.github.io/rxinfer-website/).

"""

# ╔═╡ 6d90a958-6f2b-4f18-a121-0d1bab9e4d91
md"""
#### Parameters
"""

# ╔═╡ 1070063a-ef85-4527-ae82-1f01c1a506ff
Σ = 1e5 * Diagonal(I,3) # Covariance matrix of prior on w

# ╔═╡ ba7a2dbd-f068-4249-bc29-77f2d0804676
σ2 = 2.0;                # Noise variance

# ╔═╡ 480165f9-33d9-4db1-bf05-8d99f0d9fb3e
md"""
#### Generating a data set
"""

# ╔═╡ aec4726a-954e-4e76-aae5-2dd6c979b12d
w = [1.0; 2.0; 0.25]

# ╔═╡ 1c9c7994-672c-42a3-8ae7-8ce092ada9f0
N = 30;

# ╔═╡ 99265e22-e8dc-40fe-989f-0d2a6c72faac
z = 10.0*rand(N)

# ╔═╡ e20e9048-1271-41c7-97d3-635f320aa365
x_train = [[1.0; z; z^2] for z in z] # Feature vector x = [1.0; z; z^2]

# ╔═╡ 96ef3cfb-ca18-46d6-bcac-0122c2c85fba
f(x) = (w'*x)[1];

# ╔═╡ 34ebbbe1-2a6b-422b-aeb1-cd2953acddca
y_train = map(f, x_train) + sqrt(σ2)*randn(N) # y[i] = w' * x[i] + ϵ

# ╔═╡ 7764541a-c11e-4e12-bbac-f8906cbc5dc6
scatter(z, y_train, label="data", xlabel=L"z", ylabel=L"f([1.0, z, z^2]) + \epsilon")

# ╔═╡ 965a1df0-d294-11ef-323c-3da765f1104a
md"""
Now build the factor graph in RxInfer, perform sum-product message passing and plot results (mean of posterior).

"""

# ╔═╡ fd338a30-9622-405a-96fa-caca6bd4ccfb
@model function linear_regression(y,x, N, Σ, σ2)

    w ~ MvNormalMeanCovariance(zeros(3),Σ)
    
    for i in 1:N
        y[i] ~ NormalMeanVariance(dot(w , x[i]), σ2)
    end
end

# ╔═╡ c03b1140-adce-467a-b953-50ad1bf3bc34
# Run message passing algorithm 
results = infer(
    model      = linear_regression(N=length(x_train), Σ=Σ, σ2=σ2),
    data       = (y = y_train, x = x_train),
    returnvars = (w = KeepLast(),),
    iterations = 20,
)

# ╔═╡ 83a70a4b-b114-4351-8fa2-dd565ebc9916
convert(MvNormal, results.posteriors[:w])

# ╔═╡ 965a37e8-d294-11ef-340f-0930b229dd32
let
	plt = scatter(z, y_train, label="data", xlabel=L"z", ylabel=L"f([1.0, z, z^2]) + \epsilon")
	z_test = collect(0:0.2:12)
	x_test = [[1.0; z; z^2] for z in z_test]
	for i=1:10
	    w_sample = rand(results.posteriors[:w])
	    f_est(x) = (w_sample'*x)[1]
	    plot!(plt, z_test, map(f_est, x_test), alpha=0.3, label=nothing);
	end
	plt
end

# ╔═╡ 965a6c20-d294-11ef-1c91-4bd237afbd20
md"""
## Final thoughts: Modularity and Abstraction

The great Michael Jordan (no, not [this one](https://youtu.be/cuLprHh_BRg), but [this one](https://people.eecs.berkeley.edu/~jordan/)), wrote:   

> "I basically know of two principles for treating complicated systems in simple ways: the first is the principle of **modularity** and the second is the principle of **abstraction**. I am an apologist for computational probability in machine learning because I believe that probability theory implements these two principles in deep and intriguing ways — namely through factorization and through averaging. Exploiting these two mechanisms as fully as possible seems to me to be the way forward in machine learning." — Michael Jordan, 1997 (quoted in [Fre98](https://mitpress.mit.edu/9780262062022/)).

Factor graphs capture these ideas elegantly—both visually and computationally.

**Visually**, the graph structure displays the modularity of conditional independencies in the model. Each node encapsulates internal complexity, and by closing the box, we can hierarchically move to higher levels of abstraction.

**Computationally**, message-passing inference exploits the distributive law to avoid unnecessary computations.

Although RxInfer is still under active development, my prediction is that within 5–10 years, RxInfer—or a comparable toolbox—will be able to automate Bayesian inference for virtually any interesting probabilistic model you can conceive. In principle, you will then have all the tools needed to implement the four-step Bayesian ML recipe—model specification, parameter learning, model evaluation, and application—for any (Bayesian) information processing problem. 


"""

# ╔═╡ 25492eea-e649-43f9-b71f-ac6d1a80d0ee
md"""
# Exercises
"""

# ╔═╡ a5cd774f-57ad-4cb5-86c0-35987aa6e221
md"""
##### Message Passing in a State Space Model (*)
"""

# ╔═╡ b6de3f00-d3b8-44d8-b72a-48cd5628b607
TwoColumn(md""" Consider the following state-space model:

```math
\begin{align*}
z_k &= A z_{k-1} + w_k \\
x_k &= C z_k + v_k 
\end{align*}
```

where ``k=1,2,\ldots,n`` is the time step counter; ``z_k`` is  an *unobserved* state sequence; ``x_k`` is an *observed* sequence; ``w_k \sim \mathcal{N}(0,\Sigma_w)`` and ``v_k \sim \mathcal{N}(0,\Sigma_v)`` are (unobserved) state and observation noise sequences respectively; ``z_0 \sim \mathcal{N}(0,\Sigma_0)`` is the initial state and ``A``, ``C``, ``\Sigma_v``,``\Sigma_w`` and ``\Sigma_0`` are known parameters. """,
@htl """
<img src="https://github.com/bmlip/course/blob/main/assets/figures/ffg-5SSB0-exam-Kalman-filter.png?raw=true" alt=" " style="display: block; width: 100%; margin: 0 auto;">
""")
	

# ╔═╡ 05375a01-4d1b-44cc-b1c4-a5eb4b6c5c5b
md"""
- (a) Rewrite the state-space equations as a set of conditional probability distributions.                 

```math
\begin{align*}
 p(z_k|z_{k-1},A,\Sigma_w) &= \ldots \\
 p(x_k|z_k,C,\Sigma_v) &= \ldots \\
 p(z_0|\Sigma_0) &= \ldots
\end{align*}
```

- (b) Define ``z^n \triangleq (z_0,z_1,\ldots,z_n)``, ``x^n \triangleq (x_1,\ldots,x_n)`` and ``\theta=\{A,C,\Sigma_w,\Sigma_v\}``. Now write out the generative model ``p(x^n,z^n|\theta)`` as a product of factors.     

- (c) We are interested in estimating ``z_k`` from a given estimate for ``z_{k-1}`` and the current observation ``x_k``, i.e., we are interested in computing ``p(z_k|z_{k-1},x_k,\theta)``. Can ``p(z_k|z_{k-1},x_k,\theta)`` be expressed as a Gaussian distribution? Explain why or why not in one sentence.    

- (d) Copy the graph onto your exam paper and draw the message passing schedule for computing ``p(z_k|z_{k-1},x_k,\theta)`` by drawing arrows in the factor graph. Indicate the order of the messages by assigning numbers to the arrows.    

- (e) Now assume that our belief about parameter ``\Sigma_v`` is instead given by a distribution ``p(\Sigma_v)`` (rather than a known value). Adapt the factor graph drawing of the previous answer to reflect our belief about ``\Sigma_v``.  
"""

# ╔═╡ 206c34b3-1873-460b-911e-f2cd4f8886af
details("Click for solution",
md"""

- (a) Rewrite the state-space equations as a set of conditional probability distributions.                 

```math
\begin{align*}
 p(z_k|z_{k-1},A,\Sigma_w) &= \ldots \\
 p(x_k|z_k,C,\Sigma_v) &= \ldots \\
 p(z_0|\Sigma_0) &= \ldots
\end{align*}
```

This is a linear system with only Gaussian source signals (``w_k`` and ``v_k``), hence the distributions for ``z_k`` and ``x_k`` will also be Gaussian. As a result, we only need to compute the mean and covariance matrix. We begin with the mean for ``p(z_k|z_{k-1},A,\Sigma_w)``:


```math
\begin{align*}
  E[z_k|z_{k-1},A,\Sigma_w] &= E[A z_{k-1} + w_k|z_{k-1},A,\Sigma_w] \\
  &= E[A z_{k-1}|z_{k-1},A] + E[w_k|\Sigma_w] \\
  &= A z_{k-1} + 0
  \end{align*}
```

And now the variance:


```math
\begin{align*}
  V[z_k|z_{k-1},A,\Sigma_w] &= E[(z_k - E[z_k])(z_k-E[z_k])^T \,|\,z_{k-1},A,\Sigma_w ] \\ &= E[(\overbrace{A z_{k-1} + w_k}^{z_k} - \overbrace{A z_{k-1}}^{E[z_k]})(A z_{k-1} + w_k-A z_{k-1})^T|z_{k-1},A,\Sigma_w] \\
  &= E[w_k w_k^T|\Sigma_w] \\
  &= \Sigma_w
  \end{align*}
```

You can execute similar computations for the other distributions, leading to


```math
\begin{align*}
 p(z_k|z_{k-1},A,\Sigma_w) &= \mathcal{N}(z_k|A z_{k-1},\Sigma_w) \\
 p(x_k|z_k,C,\Sigma_v) &= \mathcal{N}(x_k|C z_k,\Sigma_v) \\
  p(z_0|\Sigma_0) &= \mathcal{N}(z_0|0,\Sigma_0)
\end{align*}
```

- (b) Define ``z^n \triangleq (z_0,z_1,\ldots,z_n)``, ``x^n \triangleq (x_1,\ldots,x_n)`` and ``\theta=\{A,C,\Sigma_w,\Sigma_v\}``. Now write out the generative model ``p(x^n,z^n|\theta)`` as a product of factors.     

```math
\begin{align*}
p(x^n,z^n|\theta) &= p(z_0|\Sigma_0) \prod_{k=1}^n p(x_k|z_k,C,\Sigma_v) \,p(z_k|z_{k-1},A,\Sigma_w) \\
  &= \mathcal{N}(z_0|0,\Sigma_0) \prod_{k=1}^n  \mathcal{N}(x_k|C z_k,\Sigma_v) \,\mathcal{N}(z_k|A z_{k-1},\Sigma_w)
\end{align*}
```

- (c) We are interested in estimating ``z_k`` from a given estimate for ``z_{k-1}`` and the current observation ``x_k``, i.e., we are interested in computing ``p(z_k|z_{k-1},x_k,\theta)``. Can ``p(z_k|z_{k-1},x_k,\theta)`` be expressed as a Gaussian distribution? Explain why or why not in one sentence.    

Yes, since the generative model ``p(x^n,z^n|\theta)`` is (one big) Gaussian.


""")

# ╔═╡ a9a9f3a2-67e7-4ff5-bc94-29229656ca40
TODO("FONS, I like to put a two-column block into one foldable solution, so that answers for (d) and (e) are also in the details() function.")

# ╔═╡ 45251c19-6eae-41e7-b0ed-8bd70a67d4e0
TwoColumn(
	md"""
	- (d) Copy the graph onto your exam paper and draw the message passing schedule for computing ``p(z_k|z_{k-1},x_k,\theta)`` by drawing arrows in the factor graph. Indicate the order of the messages by assigning numbers to the arrows.      
	
	Some permutations of this order are also possible. The most important thing here is that you recognize the tree with ``Z_k`` as a root of the tree and pass messages from the terminals (e.g., ``Z_{k-1}``, ``X_k``, etc.) towards the root.
	""", 
	@htl """
	<img src="https://github.com/bmlip/course/blob/main/assets/figures/ffg-5SSB0-exam-Kalman-filter-wMessages-wUncertainSigmaV.png?raw=true" alt=" " style="display: block; width: 100%; margin: 0 auto;">
	""")

# ╔═╡ 7f4c06cb-139e-4e15-a032-b8991183634f
md"""
	- (e) Now assume that our belief about parameter ``\Sigma_v`` is instead given by a distribution ``p(\Sigma_v)`` (rather than a known value). Adapt the factor graph drawing of the previous answer to reflects our belief about ``\Sigma_v``.      
	
	For answer, see drawing for answer (d).
	"""

# ╔═╡ a6e155eb-7376-4e57-8e63-628934e14e78
md"""
##### Messages for the Addition Node (*)

"""

# ╔═╡ 9dc870d7-a5f3-447c-96ee-ad23199bc253
TwoColumn(
md"""
Consider an addition node

```math
f_+(x,y,z) = \delta(z-x-y)
```
- Derive an expression for the outgoing message ``\overrightarrow{\mu}_{Z}(z)`` in terms of the incoming messages ``\overrightarrow{\mu}_{X}(\cdot)`` and ``\overrightarrow{\mu}_{Y}(\cdot)``.   

""", 
	
@htl """

<img src="https://github.com/bmlip/course/blob/main/assets/figures/ffg-addition-node.png?raw=true" alt=" " style="display: block; width: 100%; margin: 0 auto;">

""")

# ╔═╡ e8a35c28-6d6d-4066-8251-f091f28622a9
details("Click for solution", 
md"""

We use the sum-product rule to compute


```math
\begin{align*}
  \overrightarrow{\mu}_{Z}(z) &= \iint  \overrightarrow{\mu}_{X}(x) \overrightarrow{\mu}_{Y}(y) \,\delta(z-x-y) \,\mathrm{d}x \mathrm{d}y \\
   &=  \int  \overrightarrow{\mu}_{X}(x) \overrightarrow{\mu}_{Y}(z-x) \,\mathrm{d}x \,, 
  \end{align*}
```

i.e., ``\overrightarrow{\mu}_{Z}`` is the convolution of the messages ``\overrightarrow{\mu}_{X}`` and ``\overrightarrow{\mu}_{Y}``.

		""")

# ╔═╡ 965a8a1a-d294-11ef-1d2f-65abf76665e8
md"""
# OPTIONAL SLIDES

"""

# ╔═╡ 965aa14c-d294-11ef-226f-65d587fefa64
md"""
## $(HTML("<span id='sp-for-multiplication-node'>Sum-Product Messages for Multiplication Nodes</span>"))

Next, let us consider a **multiplication** by a fixed (invertible matrix) gain ``f_A(x,y) = \delta(y-Ax)``

![](https://github.com/bmlip/course/blob/v2/assets/figures/ffg-gain-node.png?raw=true)

"""

# ╔═╡ 965ab77c-d294-11ef-2510-95b1a998589f
md"""
```math
\begin{align*}
\overrightarrow{\mu}_{Y}(y) &= \int  \overrightarrow{\mu}_{X}(x) \,\delta(y-Ax) \,\mathrm{d}x \\
&= \int  \overrightarrow{\mu}_{X}(x) \,|A|^{-1}\delta(x-A^{-1}y) \,\mathrm{d}x \\
&= |A|^{-1}\overrightarrow{\mu}_{X}(A^{-1}y) \,.
\end{align*}
```

"""

# ╔═╡ 965af708-d294-11ef-112c-f5470031dbbe
md"""
For a Gaussian message input message ``\overrightarrow{\mu}_{X}(x) = \mathcal{N}(x|\overrightarrow{m}_{X},\overrightarrow{V}_{X})``, the output message is also Gaussian with 

```math
\begin{align*}
\overrightarrow{m}_{Y} = A\overrightarrow{m}_{X} \,,\,\text{and}\,\,
\overrightarrow{V}_{Y} = A\overrightarrow{V}_{X}A^T
\end{align*}
```

since 

```math
\begin{align*}
\overrightarrow{\mu}_{Y}(y) &= |A|^{-1}\overrightarrow{\mu}_{X}(A^{-1}y) \\
  &\propto \exp \left( -\frac{1}{2} \left( A^{-1}y - \overrightarrow{m}_{X}\right)^T \overrightarrow{V}_{X}^{-1} \left(  A^{-1}y - \overrightarrow{m}_{X}\right)\right) \\
   &= \exp \big( -\frac{1}{2} \left( y - A\overrightarrow{m}_{X}\right)^T \underbrace{A^{-T}\overrightarrow{V}_{X}^{-1} A^{-1}}_{(A \overrightarrow{V}_{X} A^T)^{-1}} \left( y - A\overrightarrow{m}_{X}\right)\big) \\
  &\propto  \mathcal{N}(y| A\overrightarrow{m}_{X},A\overrightarrow{V}_{X}A^T) \,.
\end{align*}
```

"""

# ╔═╡ 965b11a4-d294-11ef-1d04-dbdf39ce91a3
md"""
**Exercise**: Prove that, for the same factor ``\delta(y-Ax)`` and Gaussian messages, the (backward) sum-product message ``\overleftarrow{\mu}_{X}`` is given by 

```math
\begin{align*}
\overleftarrow{\xi}_{X} &= A^T\overleftarrow{\xi}_{Y} \\
\overleftarrow{W}_{X} &= A^T\overleftarrow{W}_{Y}A
\end{align*}
```

where ``\overleftarrow{\xi}_X \triangleq \overleftarrow{W}_X \overleftarrow{m}_X`` and ``\overleftarrow{W}_{X} \triangleq \overleftarrow{V}_{X}^{-1}`` (and similarly for ``Y``).

"""

# ╔═╡ 965b25ac-d294-11ef-0b9a-9d5a50a76069
md"""
## $(HTML("<span id='sp-for-addition-node'>Code example: Gaussian forward and backward messages for the Addition node</span>"))

Let's calculate the Gaussian forward and backward messages for the addition node in RxInfer.  ![](https://github.com/bmlip/course/blob/v2/assets/figures/ffg-addition-node.png?raw=true)

"""

# ╔═╡ bfbf3d09-23f5-4f54-96f6-bfe536cfc228
md"Forward message on ``Z``:"

# ╔═╡ e7e4b6d0-bdf0-4a93-9a73-7971e6e33065
@call_rule typeof(+)(:out, Marginalisation) (m_in1 = NormalMeanVariance(1.0, 1.0), m_in2 = NormalMeanVariance(2.0, 1.0))

# ╔═╡ 2f5415e5-70b1-47ea-9790-7ac953bca538
md"Backward message on ``X``:"

# ╔═╡ 1b76ab6c-ffa2-40eb-a6c6-55d7097a5108
@call_rule typeof(+)(:in1, Marginalisation) (m_out = NormalMeanVariance(3.0, 1.0), m_in2 = NormalMeanVariance(2.0, 1.0))

# ╔═╡ 965b886e-d294-11ef-1b10-0319896874cf
md"""
## Code Example: forward and backward messages for the Matrix Multiplication node

In the same way we can also investigate the forward and backward messages for the matrix multiplication ("gain") node  ![](https://github.com/bmlip/course/blob/v2/assets/figures/ffg-gain-node.png?raw=true)

"""

# ╔═╡ 0efe10d8-1d0e-4a8f-8005-25ee261322b8
md"Forward message on ``Y``:"

# ╔═╡ 1be3121d-be18-46a1-9af9-f108a2257c22
@call_rule typeof(*)(:out, Marginalisation) (m_A = PointMass(4.0), m_in = NormalMeanVariance(1.0, 1.0))

# ╔═╡ e5658c95-6cd0-426f-b819-31f9f2c7eaf4
md"Backward message on ``X``:"

# ╔═╡ 94ca674e-1a01-424c-8657-6510be7097c3
@call_rule typeof(*)(:in, Marginalisation) (m_out = NormalMeanVariance(2.0, 1.0), m_A = PointMass(4.0))

# ╔═╡ 965c18f8-d294-11ef-2456-b945a46241f4
md"""
## Example: Sum-Product Algorithm to infer a posterior

Consider a generative model 

```math
p(x,y_1,y_2) = p(x)\,p(y_1|x)\,p(y_2|x) .
```

This model expresses the assumption that ``Y_1`` and ``Y_2`` are independent measurements of ``X``.

![](https://github.com/bmlip/course/blob/v2/assets/figures/ffg-observations.png?raw=true)

"""

# ╔═╡ 965c2a4e-d294-11ef-1aab-73725568c64e
md"""
Assume that we are interested in the posterior for ``X`` after observing ``Y_1= \hat y_1`` and ``Y_2= \hat y_2``. The posterior for ``X`` can be inferred by applying the sum-product algorithm to the following graph:

![](https://github.com/bmlip/course/blob/v2/assets/figures/ffg-observations-2.png?raw=true)

"""

# ╔═╡ 965c39a8-d294-11ef-1d83-bde85e3ca790
md"""
!!! note
	We usually draw terminal nodes for observed variables in the graph by smaller solid-black squares. This is just to help the visualization of the graph, since the computational rules are no different than for other nodes. 

"""

# ╔═╡ 965c5f28-d294-11ef-324e-4df3e38b5045
md"""
## Code for Sum-Product Algorithm to infer  a posterior

We'll use RxInfer to build the above graph, and perform sum-product message passing to infer the posterior ``p(x|y_1,y_2)``. We assume ``p(y_1|x)`` and ``p(y_2|x)`` to be Gaussian likelihoods with known variances:

```math
\begin{align*}
    p(y_1\,|\,x) &= \mathcal{N}(y_1\,|\,x, v_{y1}) \\
    p(y_2\,|\,x) &= \mathcal{N}(y_2\,|\,x, v_{y2})
\end{align*}
```

Under this model, the posterior is given by:

```math
\begin{align*}
    p(x\,|\,y_1,y_2) &\propto \overbrace{p(y_1\,|\,x)\,p(y_2\,|\,x)}^{\text{likelihood}}\,\overbrace{p(x)}^{\text{prior}} \\
    &=\mathcal{N}(x\,|\,\hat{y}_1, v_{y1})\, \mathcal{N}(x\,|\,\hat{y}_2, v_{y2}) \, \mathcal{N}(x\,|\,m_x, v_x) 
\end{align*}
```

so we can validate the answer by solving the Gaussian multiplication manually.

"""

# ╔═╡ d27f7af6-e094-44fa-8ba4-4ad2fa38f8bc
y1_hat = 1.0; y2_hat = 2.0;

# ╔═╡ 90d62ba0-ca97-43f6-8f5a-0c1086a13f3d
md"""
Construct the factor graph

"""

# ╔═╡ 053e9dde-c088-4f15-9ca6-98b8185a8a11
@model function my_model(y1,y2)

    # `x` is the hidden states
    x ~ NormalMeanVariance(0.0, 4.0)

    # `y1` and `y2` are "clamped" observations
    y1 ~ NormalMeanVariance(x, 1.0)
    y2 ~ NormalMeanVariance(x, 2.0)
    
    return x
end

# ╔═╡ 07b09ac1-7fa7-4b62-b130-97315adb6fa7
result = infer(model=my_model(), data=(y1=y1_hat, y2 = y2_hat,))

# ╔═╡ defb2149-294b-47a8-99ed-1b3746b275f1
Text("Sum-product message passing result: p(x|y1,y2) = \n\t𝒩($(mean(result.posteriors[:x])),$(var(result.posteriors[:x])))")

# ╔═╡ c95bf9a4-2e7b-4b3a-a161-56f3fd16ad0f
# TODO: could also write this as:
var"p(x|y1,y2)" = convert(Normal, result.posteriors[:x])

# ╔═╡ b3656d6c-4717-4fcd-90c6-ae4f4aa5e1be


# ╔═╡ b15f28ce-c8c1-439b-aeca-74a58d2557e2
md"""
We calculate mean and variance of p(x|y1,y2) manually by multiplying 3 Gaussians (see lesson 4 for details)
"""

# ╔═╡ 86e67c05-068d-4de4-80f3-1a20cc8a43ea
v = 1 / (1/4 + 1/1 + 1/2)

# ╔═╡ fffa27d5-eb68-4dd3-9995-4a53fba6c1e4
m = v * (0/4 + y1_hat/1.0 + y2_hat/2.0)

# ╔═╡ 578ec319-337d-4396-bb75-eaf99d95a38d
Text("Manual result: p(x|y1,y2) = \n\t𝒩($(m), $(v))")

# ╔═╡ 89da2fc0-a7c8-4a9d-82d9-622a311d010d
md"""
# Appendix
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
RxInfer = "86711068-29c9-4ff7-b620-ae75d7495b3d"

[compat]
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
Plots = "~1.40.13"
PlutoTeachingTools = "~0.3.1"
PlutoUI = "~0.7.62"
RxInfer = "~4.4.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "aacc534113e47e6438eb875663f1139a4ac160cd"

[[deps.ADTypes]]
git-tree-sha1 = "7927b9af540ee964cc5d1b73293f1eb0b761a3a1"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.16.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "9606d7832795cbef89e06a550475be300364a8aa"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.19.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "120e392af69350960b1d3b89d41dcc1d66543858"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.11.2"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BayesBase]]
deps = ["Distributions", "DomainSets", "LinearAlgebra", "Random", "SpecialFunctions", "StaticArrays", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "TinyHugeNumbers"]
git-tree-sha1 = "232c38ab317e6e84596414fb2e1c29786b85806f"
uuid = "b4ee3484-f114-42fe-b91c-797d54a0c67e"
version = "1.5.7"
weakdeps = ["FastCholesky"]

    [deps.BayesBase.extensions]
    FastCholeskyExt = "FastCholesky"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BitSetTuples]]
deps = ["TupleTools"]
git-tree-sha1 = "aa19428fb6ad21db22f8568f068de4f443d3bacc"
uuid = "0f2f92aa-23a3-4d05-b791-88071d064721"
version = "1.1.5"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "291532989f81db780e435452ccb2a5f902ff665f"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.7.0"

    [deps.BlockArrays.extensions]
    BlockArraysAdaptExt = "Adapt"
    BlockArraysBandedMatricesExt = "BandedMatrices"

    [deps.BlockArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "5a97e67919535d6841172016c9530fd69494e5ec"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.6"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "5ac098a7c8660e217ffac31dc2af0964a8c3182a"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "2.0.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "a656525c8b46aa6a1c76891552ed5381bb32ae7b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.30.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Combinatorics]]
git-tree-sha1 = "8010b6bb3388abe68d95743dcbea77650bb2eddf"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.3"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "0037835448781bb46feb39866934e243886d756a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "a86af9c4c4f33e16a2b2ff43c2113b2f390081fa"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.5"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "f620da805b82bec64ab4d5f881c7592c82dbc08a"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.3"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3e6d038b77f22791b8e3472b7c633acea1ecac06"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.120"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.DomainIntegrals]]
deps = ["CompositeTypes", "DomainSets", "FastGaussQuadrature", "GaussQuadrature", "HCubature", "IntervalSets", "LinearAlgebra", "QuadGK", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "934bf806ef2948114243f25e84a3ddf775d0f1a6"
uuid = "cc6bae93-f070-4015-88fd-838f9505a86c"
version = "0.5.2"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "c249d86e97a7e8398ce2068dce4c078a1c3464de"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.7.16"

    [deps.DomainSets.extensions]
    DomainSetsMakieExt = "Makie"
    DomainSetsRandomExt = "Random"

    [deps.DomainSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EnumX]]
git-tree-sha1 = "bddad79635af6aec424f53ed8aad5d7555dc6f00"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.5"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.ExponentialFamily]]
deps = ["BayesBase", "BlockArrays", "Distributions", "DomainSets", "FastCholesky", "FillArrays", "ForwardDiff", "HCubature", "HypergeometricFunctions", "IntervalSets", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "PositiveFactorizations", "Random", "SparseArrays", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "TinyHugeNumbers"]
git-tree-sha1 = "00188d3ea03cfe63d6b82e9e5b81972d56f8403b"
uuid = "62312e5e-252a-4322-ace9-a5f4bf9b357b"
version = "2.0.7"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "83dc665d0312b41367b7263e8a4d172eac1897f4"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.4"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3a948313e7a41eb1db7a1e733e6335f17b4ab3c4"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "7.1.1+0"

[[deps.FastCholesky]]
deps = ["LinearAlgebra", "PositiveFactorizations"]
git-tree-sha1 = "1c0a81e006e40e9fcbd5f6f6cb42ac2700f86889"
uuid = "2d5283b6-8564-42b6-bb00-83ed8e915756"
version = "1.4.3"
weakdeps = ["StaticArraysCore"]

    [deps.FastCholesky.extensions]
    StaticArraysCoreExt = "StaticArraysCore"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "fd923962364b645f3719855c88f7074413a6ad92"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.0.2"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "b66970a70db13f45b7e57fbda1736e1cf72174ea"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "f089ab1f834470c525562030c8cfde4025d5e915"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.27.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedArguments]]
deps = ["TupleTools"]
git-tree-sha1 = "befa1ad59c77643dec6fc20d71fd6f5c3afcdadd"
uuid = "4130a065-6d82-41fe-881e-7a5c65156f7d"
version = "0.1.1"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "1828eb7275491981fa5f1752a5e126e8f26f8741"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.17"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "27299071cc29e409488ada41ec7643e0ab19091f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.17+0"

[[deps.GaussQuadrature]]
deps = ["SpecialFunctions"]
git-tree-sha1 = "eb6f1f48aa994f3018cbd029a17863c6535a266d"
uuid = "d54b0c1a-921d-58e0-8e36-89d8069c0969"
version = "0.5.8"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "35fbd0cefb04a516104b8e183ce0df11b70a3f1a"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.84.3+0"

[[deps.GraphPPL]]
deps = ["BitSetTuples", "DataStructures", "Dictionaries", "MacroTools", "MetaGraphsNext", "NamedTupleTools", "Static", "StaticArrays", "TupleTools", "Unrolled"]
git-tree-sha1 = "efc643a7065bdba366fc4e50dbc20661194b7806"
uuid = "b3f8163a-e979-4e85-b43e-1f63d8c8b42c"
version = "4.6.2"

    [deps.GraphPPL.extensions]
    GraphPPLDistributionsExt = "Distributions"
    GraphPPLGraphVizExt = "GraphViz"
    GraphPPLPlottingExt = ["Cairo", "GraphPlot"]

    [deps.GraphPPL.weakdeps]
    Cairo = "159f3aea-2a34-519c-b102-8c37f9878175"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    GraphPlot = "a2cc645c-3eea-5389-862e-a155d0052231"
    GraphViz = "f526b714-d49f-11e8-06ff-31ed36ee7ee0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "c5abfa0ae0aaee162a3fbb053c13ecda39be545b"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HCubature]]
deps = ["Combinatorics", "DataStructures", "LinearAlgebra", "QuadGK", "StaticArrays"]
git-tree-sha1 = "19ef9f0cb324eed957b7fe7257ac84e8ed8a48ec"
uuid = "19dc6840-f33b-545b-b366-655c7e3ffd49"
version = "1.7.0"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "ed5e9c58612c4e081aecdb6e1a479e18462e041e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "8e070b599339d622e9a081d17230d74a5c473293"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.17"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IntervalSets]]
git-tree-sha1 = "5fbb102dcb8b1a858111ae81d56682376130517d"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.11"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues", "TranscodingStreams"]
git-tree-sha1 = "d97791feefda45729613fafeccc4fbef3f539151"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.15"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "e09121f4c523d8d8d9226acbed9cb66df515fcf2"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.4"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "4f34eaabe49ecb3fb0d58d6015e32fd31a733199"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.8"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "SparseArrays"]
git-tree-sha1 = "76627adb8c542c6b73f68d4bfd0aa71c9893a079"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "2.6.2"

    [deps.LazyArrays.extensions]
    LazyArraysBandedMatricesExt = "BandedMatrices"
    LazyArraysBlockArraysExt = "BlockArrays"
    LazyArraysBlockBandedMatricesExt = "BlockBandedMatrices"
    LazyArraysStaticArraysExt = "StaticArrays"

    [deps.LazyArrays.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "4adee99b7262ad2a1a4bbbc59d993d24e55ea96f"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.4.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "e5afce7eaf5b5ca0d444bcb4dc4fd78c54cbbac0"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.172"

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.LoopVectorization.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "73b98709ad811a6f81d84e105f4f695c229385ba"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.4.3"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MatrixCorrectionTools]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "73f93b21eae5714c282396bfae9d9f13d6ad04b6"
uuid = "41f81499-25de-46de-b591-c3cfc21e9eaf"
version = "1.2.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MetaGraphsNext]]
deps = ["Graphs", "JLD2", "SimpleTraits"]
git-tree-sha1 = "1e3b196ecbbf221d4d3696ea9de4288bea4c39f9"
uuid = "fa8bd995-216d-47f1-8a91-f3b68fbeb377"
version = "0.7.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "25a6638571a902ecfb1ae2a18fc1575f86b1d4df"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.10.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NamedTupleTools]]
git-tree-sha1 = "90914795fc59df44120fe3fff6742bb0d7adb1d0"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.14.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+4"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "87510f7292a2b21aeff97912b0898f9553cc5c2c"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["Compat", "EnumX", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "61942645c38dd2b5b78e2082c9b51ab315315d10"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.13.2"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "275a9a6d85dc86c24d03d1837a0010226a96f540"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.3+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "3db9167c618b290a05d4345ca70de6d95304a32a"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.17"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoLinks", "PlutoUI"]
git-tree-sha1 = "8252b5de1f81dc103eb0293523ddf917695adea1"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.3.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ec9e63bd098c50e4ad28e7cb95ca7a4860603298"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.68"

[[deps.PolyaGammaHybridSamplers]]
deps = ["Distributions", "Random", "SpecialFunctions", "StatsFuns"]
git-tree-sha1 = "9f6139650ff57f9d8528cd809ebc604c7e9738b1"
uuid = "c636ee4f-4591-4d8c-9fae-2dea21daa433"
version = "1.2.6"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "13c5103482a8ed1536a54c08d0e742ae3dca2d42"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.4"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "eb38d376097f47316fe089fc62cb7c6d85383a52"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "e1d5e16d0f65762396f9ca4644a5f4ddab8d452b"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.ReactiveMP]]
deps = ["BayesBase", "DataStructures", "DiffResults", "Distributions", "DomainIntegrals", "DomainSets", "ExponentialFamily", "FastCholesky", "FastGaussQuadrature", "FixedArguments", "ForwardDiff", "HCubature", "LazyArrays", "LinearAlgebra", "LoopVectorization", "MacroTools", "MatrixCorrectionTools", "Optim", "PolyaGammaHybridSamplers", "PositiveFactorizations", "Random", "Rocket", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "TinyHugeNumbers", "Tullio", "TupleTools", "Unrolled"]
git-tree-sha1 = "feff187996d9163f0e277673c17c0f458f5f6dbe"
uuid = "a194aa59-28ba-4574-a09c-4a745416d6e3"
version = "5.4.7"

    [deps.ReactiveMP.extensions]
    ReactiveMPOptimisersExt = "Optimisers"
    ReactiveMPProjectionExt = "ExponentialFamilyProjection"
    ReactiveMPRequiresExt = "Requires"

    [deps.ReactiveMP.weakdeps]
    ExponentialFamilyProjection = "17f509fa-9a96-44ba-99b2-1c5f01f0931b"
    Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "20ccb7e2501e9da93fe8450d01aeabf16a5f0c82"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.8.1"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Rocket]]
deps = ["DataStructures", "Sockets", "Unrolled"]
git-tree-sha1 = "af6e944256dc654a534082f08729afc1189933e4"
uuid = "df971d30-c9d6-4b37-b8ff-e965b2cb3a40"
version = "1.8.2"

[[deps.RxInfer]]
deps = ["BayesBase", "DataStructures", "Dates", "Distributions", "DomainSets", "ExponentialFamily", "FastCholesky", "GraphPPL", "HTTP", "JSON", "LinearAlgebra", "Logging", "MacroTools", "Optim", "Preferences", "PrettyTables", "ProgressMeter", "Random", "ReactiveMP", "Reexport", "Rocket", "Static", "Statistics", "TupleTools", "UUIDs"]
git-tree-sha1 = "c820266d2e70f4d7bac1254186b2f9cefda3bb1e"
uuid = "86711068-29c9-4ff7-b620-ae75d7495b3d"
version = "4.4.3"

    [deps.RxInfer.extensions]
    ProjectionExt = "ExponentialFamilyProjection"

    [deps.RxInfer.weakdeps]
    ExponentialFamilyProjection = "17f509fa-9a96-44ba-99b2-1c5f01f0931b"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "456f610ca2fbd1c14f5fcf31c6bfadc55e7d66e0"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.43"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "7f44eef6b1d284465fafc66baf4d9bdcc239a15b"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.4.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "f737d444cb0ad07e61b3c1bef8eb91203c321eff"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.2.0"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "cbea8a6bd7bed51b1619658dec70035e07b8502f"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.14"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "b81c5035922cc89c2d9523afc6c54be512411466"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.5"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "d969183d3d244b6c33796b5ed01ab97328f2db85"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.5"

[[deps.TinyHugeNumbers]]
git-tree-sha1 = "83c6abf376718345a85c071b249ef6692a8936d4"
uuid = "783c9a47-75a3-44ac-a16b-f1ab7b3acf04"
version = "1.0.3"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "0fc001395447da85495b7fef1dfae9789fdd6e31"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.11"

[[deps.Tullio]]
deps = ["DiffRules", "LinearAlgebra", "Requires"]
git-tree-sha1 = "972698b132b9df8791ae74aa547268e977b55f68"
uuid = "bc48ee85-29a4-5162-ae0b-a64e1601d4bc"
version = "0.3.8"

    [deps.Tullio.extensions]
    TullioCUDAExt = "CUDA"
    TullioChainRulesCoreExt = "ChainRulesCore"
    TullioFillArraysExt = "FillArrays"
    TullioTrackerExt = "Tracker"

    [deps.Tullio.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.TupleTools]]
git-tree-sha1 = "41e43b9dc950775eac654b9f845c839cd2f1821e"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.6.0"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d2282232f8a4d71f79e85dc4dd45e5b12a6297fb"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.23.1"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "af305cc62419f9bd61b6644d19170a4d258c7967"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.7.0"

[[deps.Unrolled]]
deps = ["MacroTools"]
git-tree-sha1 = "6cc9d682755680e0f0be87c56392b7651efc2c7b"
uuid = "9602ed7d-8fef-5bc8-8597-8f21381861e8"
version = "0.1.5"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "4ab62a49f1d8d9548a1c8d1a75e5f55cf196f64e"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.71"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "9caba99d38404b285db8801d5c45ef4f4f425a6d"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.1+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "c5bf2dad6a03dfef57ea0a170a1fe493601603f2"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.5+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4bba74fa59ab0755167ad24f98800fe5d727175b"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.12.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
"""

# ╔═╡ Cell order:
# ╟─96547560-d294-11ef-0fa7-6b6489f7baba
# ╟─af24aa27-b0a1-4c9b-aee0-0e5143d2f47e
# ╟─9654ea3e-d294-11ef-335c-657af1ceaf19
# ╟─96552348-d294-11ef-16d8-b53563054687
# ╟─9655959e-d294-11ef-0ca6-5f20aa579e91
# ╟─05db1eab-5b63-4ab9-8b4a-ab2cde554295
# ╟─9655b2c2-d294-11ef-057f-9b3984064411
# ╟─9655c1ae-d294-11ef-061a-991947cee620
# ╟─9655d360-d294-11ef-0f06-ab58e2ad0e5f
# ╟─9655e06c-d294-11ef-0393-9355d6e20afb
# ╟─9655ed6e-d294-11ef-370f-937b590036f3
# ╟─9655fb88-d294-11ef-1ceb-91585012d142
# ╟─965606f2-d294-11ef-305b-870427879e50
# ╟─2d5cd42e-26ea-4ee4-979f-3a4e2c6271b6
# ╟─96561594-d294-11ef-1590-198382927808
# ╟─9656cf72-d294-11ef-03aa-b715dd686c09
# ╟─27b9e811-4542-4fe8-86a5-f8ba09074761
# ╟─9656d850-d294-11ef-21a1-474b07ea7729
# ╟─9658329c-d294-11ef-0d03-45e6872c4985
# ╟─f0181b53-a604-489f-a89e-db6fc58571dd
# ╟─ea4a720f-a644-46a0-ad35-b215780e0928
# ╟─00c69a22-feb5-4d1e-9ab5-a136435d7d22
# ╟─9656e606-d294-11ef-1daa-312623552a5b
# ╟─9656ee62-d294-11ef-38f4-7bc8031df7ee
# ╟─9656fae2-d294-11ef-10d8-ff921d5956bd
# ╟─b33b2aef-e672-490c-bdf4-a5f655fa4695
# ╟─cb3df230-6c7e-41b9-ba13-3c5f8a7fbb62
# ╟─96570d3e-d294-11ef-0178-c34dda717495
# ╟─9657b088-d294-11ef-3017-e95c4c69b62b
# ╟─0afe3cdc-15ed-4d9a-848a-d1977d051866
# ╟─96571c34-d294-11ef-11ef-29beeb1f96c2
# ╟─253d4703-03d6-4961-8c3b-b70d2cbc0710
# ╟─a7b1f559-3c34-491e-83e7-ba95c8c22c80
# ╟─70736e62-2b6c-4b3a-ab59-7e51522d620b
# ╟─96575dd4-d294-11ef-31d6-b39b4c4bdea1
# ╟─5cc2016e-0383-448c-bd33-5b3a687b7436
# ╟─f65f5d0e-2583-4b88-b9f2-5fee15257c05
# ╟─91f81188-727c-4754-9a07-e754eef8bbe0
# ╟─0633afea-5e92-4bad-8402-d159c534af81
# ╟─f11564db-aafc-4df9-b494-4e5ced9bfcfe
# ╟─9651f976-b834-4b81-8810-649f0290969d
# ╟─96587a66-d294-11ef-2c7a-9fd7bea76582
# ╟─89e2757e-a09f-40c6-8dd7-9b4b4d232e17
# ╟─c4b5b124-e52a-41fc-b27e-a58181622e5c
# ╟─96589eb0-d294-11ef-239a-2513a805cdcf
# ╟─9658c106-d294-11ef-01db-cfcff611ed81
# ╟─96594d44-d294-11ef-22b8-95165fb08ce4
# ╟─96597ce0-d294-11ef-3478-25c6bbef601e
# ╟─965998a8-d294-11ef-1d18-85876e3656c5
# ╟─9659ab66-d294-11ef-027a-d3f7206050af
# ╠═965a08f4-d294-11ef-0604-1586ff37c0d4
# ╟─6d90a958-6f2b-4f18-a121-0d1bab9e4d91
# ╠═1070063a-ef85-4527-ae82-1f01c1a506ff
# ╠═ba7a2dbd-f068-4249-bc29-77f2d0804676
# ╟─480165f9-33d9-4db1-bf05-8d99f0d9fb3e
# ╠═aec4726a-954e-4e76-aae5-2dd6c979b12d
# ╠═1c9c7994-672c-42a3-8ae7-8ce092ada9f0
# ╠═99265e22-e8dc-40fe-989f-0d2a6c72faac
# ╠═e20e9048-1271-41c7-97d3-635f320aa365
# ╠═96ef3cfb-ca18-46d6-bcac-0122c2c85fba
# ╠═34ebbbe1-2a6b-422b-aeb1-cd2953acddca
# ╠═7764541a-c11e-4e12-bbac-f8906cbc5dc6
# ╟─965a1df0-d294-11ef-323c-3da765f1104a
# ╠═2cb7d369-e7fd-4d66-8321-66a9197a26bd
# ╠═fd338a30-9622-405a-96fa-caca6bd4ccfb
# ╠═c03b1140-adce-467a-b953-50ad1bf3bc34
# ╠═83a70a4b-b114-4351-8fa2-dd565ebc9916
# ╟─965a37e8-d294-11ef-340f-0930b229dd32
# ╟─965a6c20-d294-11ef-1c91-4bd237afbd20
# ╟─25492eea-e649-43f9-b71f-ac6d1a80d0ee
# ╟─a5cd774f-57ad-4cb5-86c0-35987aa6e221
# ╟─b6de3f00-d3b8-44d8-b72a-48cd5628b607
# ╟─05375a01-4d1b-44cc-b1c4-a5eb4b6c5c5b
# ╟─206c34b3-1873-460b-911e-f2cd4f8886af
# ╟─a9a9f3a2-67e7-4ff5-bc94-29229656ca40
# ╟─45251c19-6eae-41e7-b0ed-8bd70a67d4e0
# ╟─7f4c06cb-139e-4e15-a032-b8991183634f
# ╟─a6e155eb-7376-4e57-8e63-628934e14e78
# ╟─9dc870d7-a5f3-447c-96ee-ad23199bc253
# ╟─e8a35c28-6d6d-4066-8251-f091f28622a9
# ╟─965a8a1a-d294-11ef-1d2f-65abf76665e8
# ╟─965aa14c-d294-11ef-226f-65d587fefa64
# ╟─965ab77c-d294-11ef-2510-95b1a998589f
# ╟─965af708-d294-11ef-112c-f5470031dbbe
# ╟─965b11a4-d294-11ef-1d04-dbdf39ce91a3
# ╟─965b25ac-d294-11ef-0b9a-9d5a50a76069
# ╟─bfbf3d09-23f5-4f54-96f6-bfe536cfc228
# ╠═e7e4b6d0-bdf0-4a93-9a73-7971e6e33065
# ╟─2f5415e5-70b1-47ea-9790-7ac953bca538
# ╠═1b76ab6c-ffa2-40eb-a6c6-55d7097a5108
# ╟─965b886e-d294-11ef-1b10-0319896874cf
# ╟─0efe10d8-1d0e-4a8f-8005-25ee261322b8
# ╠═1be3121d-be18-46a1-9af9-f108a2257c22
# ╟─e5658c95-6cd0-426f-b819-31f9f2c7eaf4
# ╠═94ca674e-1a01-424c-8657-6510be7097c3
# ╟─965c18f8-d294-11ef-2456-b945a46241f4
# ╟─965c2a4e-d294-11ef-1aab-73725568c64e
# ╟─965c39a8-d294-11ef-1d83-bde85e3ca790
# ╟─965c5f28-d294-11ef-324e-4df3e38b5045
# ╠═d27f7af6-e094-44fa-8ba4-4ad2fa38f8bc
# ╟─90d62ba0-ca97-43f6-8f5a-0c1086a13f3d
# ╠═053e9dde-c088-4f15-9ca6-98b8185a8a11
# ╠═07b09ac1-7fa7-4b62-b130-97315adb6fa7
# ╟─defb2149-294b-47a8-99ed-1b3746b275f1
# ╠═c95bf9a4-2e7b-4b3a-a161-56f3fd16ad0f
# ╟─b3656d6c-4717-4fcd-90c6-ae4f4aa5e1be
# ╟─b15f28ce-c8c1-439b-aeca-74a58d2557e2
# ╠═86e67c05-068d-4de4-80f3-1a20cc8a43ea
# ╠═fffa27d5-eb68-4dd3-9995-4a53fba6c1e4
# ╟─578ec319-337d-4396-bb75-eaf99d95a38d
# ╟─89da2fc0-a7c8-4a9d-82d9-622a311d010d
# ╠═5a8dcadb-f0c2-4fb0-b8cd-db8cf49cc292
# ╠═981b08cc-7fb4-4880-8e8a-0b60a5dd72a2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
