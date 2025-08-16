### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 8be7c9ea-6e1b-11f0-2eb2-69a23c699fe5
md"""
# Intelligent Agents and Active Inference

  * **[1]** (##) I asked you to watch a video segment (https://youtu.be/L0pVHbEg4Yw) where Karl Friston talks about two main approaches to goal-directed acting by agents: (1) choosing actions that  maximize (the expectation of) a value function ``V(s)`` of the state (``s``) of the environment; or (2) choosing actions that minimize a functional (``F[q(s)]``) of *beliefs* (``q(s)``) over environmental states (``s``). Discuss the advantage of the latter appraoch.
  * **[2]** (#) The *good regulator theorem* states that a "successful and efficient" controller of the world must contain a model of the world. But it's hard to imagine how just learning a model of the world leads to goal-directed behavior, like learning how to read or drive a car. Which other ingredient do we need to get learning agents to behave as goal-directed agents?
  * **[3]** (##) The figure below reflects the state of a factor graph realization of an active inference agent after having pushed action ``a_t`` onto the environment and having received observation ``x_t``. In this graph, the variables ``x_\bullet``, ``u_\bullet`` and ``s_\bullet`` correspond to observations, and unobserved control and internal states respectively. Copy the figure onto your sheet and draw a message passing schedule to infer a posterior belief (i.e. after observing ``x_t``) over the next control state ``u_{t+1}``.

<img src="./i/fig-active-inference-model-specification.png" style="width:500px;">

  * **[4]** (##) The Free Energy Principle (FEP) is a theory about biological self-organization, in particular about how brains develop through interactions with their environment. Which of the following statements is not consistent with FEP (and explain your answer):       (a) We act to fullfil our predictions about future sensory inputs.       (b) Perception is inference about the environmental causes of our sensations.        (c) Our actions aim to reduce the complexity of our model of the environment.

"""

# ╔═╡ Cell order:
# ╟─8be7c9ea-6e1b-11f0-2eb2-69a23c699fe5
