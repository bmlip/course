### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 91e44708-6e1b-11f0-08a1-6da644d7510d
md"""
# ORAL EXAM EXAMPLE

### Bayesian Machine Learning and Information Processing (5SSD0)

"""

# ╔═╡ 91e466b6-6e1b-11f0-3e37-e398e68b554a
md"""
In this short notebook, we provide some examples of the type of questions that you can expect at the oral exam. In general, oral exams do not lend themselves well to proofing theorems or other exact mathematical manipulations. Instead, the focus is more on testing if you understand the conceptual ideas in this class. You should know what Bayesian machine learning (BML) is about, and talk about *how* you would solve the typical tasks that are assciated with BML.

In particular, for the models that we discussed in the class, you should be able to talk about the four stages of solving problems by probabilistic modeling: (**1- Model specification**, **2- Parameter estimation** (i.e., learning from an observed data set using Bayesian inference) **3- Model evaluation** (how good is this (trained) model?), and **4- Apply model**, e.g., for prediction or classification of new data. For some models, we did not discuss all these stages, e.g., we did not dicuss Model Evaluation for all models. I don't expect you to read beyond the class notes, so if it's not treated in the class, then I don't expect you to know about it.

The materials for the exam are unchanged, so it includes the notebooks (lessons 1-12) + probabilistic programming notebooks. Some notebooks contain in the first cell a link to extra "mandatory" materials that are also included in the tested materials (of course in the same spirit: try to understand). The Exercise notebooks are less important than for a written exam version. I advise you to read through them once and skip exercises with lots of mathematics.  

The style of the examination is conversational. We like to engage in a conversation with you about what you learned in the class. Finally, don't get too stressed out about this exam style. If you read and understand the notebooks reasonably well, then you can pose some obvious exam questions to yourself.

Next we present some example questions.

"""

# ╔═╡ 91e48332-6e1b-11f0-1baf-69d678c83af3
md"""
### Question 1: Regression

  * What is regression?
  * Which of the following two models do you recognize as a regression model? 

    (a) For ``x_n \in \mathbb{R}^M``, ``y_n \in \mathbb{R}``, ``\phi(a) = 1/(1+\exp(-a))``

```math
    y_n = w^T \phi(x_n) + \epsilon_n; \qquad \epsilon_n \sim \mathcal{N}(0,\sigma^2)
    
```

(b) For ``x_n \in \mathbb{R}^M``, ``y_n \in \{0,1\}``, ``\phi(a) = 1/(1+\exp(-a))`` 

```math
    p(y_n|x_n) = \phi(w^T x_n + \epsilon_n); \qquad \epsilon_n \sim \mathcal{N}(0,\sigma^2)
    
```

  * Let's train the parameters ``w`` for this model by Bayesian inference. How would you go about this task?
  * Let's set a prior for the weights ``w``? Any suggestions?
  * How do you compute the posterior from the prior?
  * Is the posterior a Gaussian distribution?

"""

# ╔═╡ 91e4c82c-6e1b-11f0-32c2-c77f164247ef
md"""
### Question 2: Variational Inference

  * What is Variational Bayesian inference (VB)?
  * How is VB related to Bayes rule?
  * Consider a generative model ``p(x,z|m)`` with ``x`` observed and ``z`` hidden variables. We observe a data set ``D = \{x_1, x_2, \ldots, x_N\}`` and define a free energy (FE) functional as (shares screen)

```math
 F[q] = \sum_z q(z) \log \frac{q(z)}{p(D,z|m)}
```

  * How would you use this FE functional to find the posterior ``p(z|D)``?
  * Can you also use FE energy minimization to estimate Bayesian model evidence ``p(D|m)``?
  * What is the mean-field assumption? How does that help with FE minimization?

"""

# ╔═╡ 91e4edb6-6e1b-11f0-3368-27593de5443f
open("../../styles/aipstyle.html") do f
    display("text/html", read(f,String))
end

# ╔═╡ Cell order:
# ╟─91e44708-6e1b-11f0-08a1-6da644d7510d
# ╟─91e466b6-6e1b-11f0-3e37-e398e68b554a
# ╟─91e48332-6e1b-11f0-1baf-69d678c83af3
# ╟─91e4c82c-6e1b-11f0-32c2-c77f164247ef
# ╠═91e4edb6-6e1b-11f0-3368-27593de5443f
