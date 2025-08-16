### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ da0d3f1c-6e1b-11f0-35a1-0706ebea9dcd
md"""
# Intelligent Agents and Active Inference

  * **[1]** (##) I asked you to watch a video segment (https://youtu.be/L0pVHbEg4Yw) where Karl Friston talks about two main approaches to goal-directed acting by agents: (1) choosing actions that  maximize (the expectation of) a value function ``V(s)`` of the state (``s``) of the environment; or (2) choosing actions that minimize a functional (``F[q(s)]``) of *beliefs* (``q(s)``) over environmental states (``s``). Discuss the advantage of the latter appraoch. 

    > We'll discuss two advantages here. Either one would suffice for full credit and there are likely multiple alternative answers that would be adequate as well. (1) One advantage is that the value function ``V`` needs to be uniquely chosen for each problem. Brains cannot afford to come up with a new value function for each problem as thousands of new problems are encountered each day. In contrast, ``F[q(s)]`` holds the free-energy functional (a given cost functional) for posterior beliefs that technically are defined by a generative model ``p`` and Bayes rule. In other words, in the latter approach, there is one value (cost) function for *all* problems. (2) A second advantage of ``F[q(s)]`` is that inference for actions can take into account the uncertainty about our state-of-knowledge of the environment. This may lead to actions that are information seeking rather than goal-driven if our belief are very uncertain. For instance, if I want to cross a street, my first actions will be to seek information (look for cars and how fast they go), and only after enough information has been collected, the goal-driven action (decision) cross-vs-stay will be executed. When minimization of ``F[q(s)]`` drives actions, both information-seeking and goal-driven actions can be accomodated in the same framework. This is very difficult when the value function is a direct function of the state of the world (``V(s)``, because there is no accomodation to represent our uncertainties about the state of the world.
  * **[2]** (#) The *good regulator theorem* states that a "successful and efficient" controller of the world must contain a model of the world. But it's hard to imagine how just learning a model of the world leads to goal-directed behavior, like learning how to read or drive a car. Which other ingredient do we need to get learning agents to behave as goal-directed agents? 

    > In the Free Energy Principle framework, goals (targets) are encoded in a generative model of the environment as prior distributions on future observations. Actions are inferred through free energy minimization in this extended model. As a result, the inferred actions aim to generate future observations that are maximally consistent with the goal priors. This kind of behavior can be interpreted as goal-directed behavior.
  * **[3]** (##) The figure below reflects the state of a factor graph realization of an active inference agent after having pushed action ``a_t`` onto the environment and having received observation ``x_t``. In this graph, the variables ``x_\bullet``, ``u_\bullet`` and ``s_\bullet`` correspond to observations, and unobserved control and internal states respectively. Copy the figure onto your sheet and draw a message passing schedule to infer a posterior belief (i.e. after observing ``x_t``) over the next control state ``u_{t+1}``.

<img src="./i/fig-active-inference-model-specification.png" style="width:500px;">

> Imagine picking up the tree at the ``u_{t+1}`` edge (call this edge the root of the tree). Then pass messages from the leaves of the tree towards the root, see Figure below. Note that the posterior belief over next control (action) ``u_{t+1}``


incorporates information from the recent past (blue messages), from prior information about what worked in the past (green arrow), and from expectations about future observations (red messages).

<img src="./i/fig-solution-active-inference-model-specification.png" style="width:500px;">

  * **[4]** (##) The Free Energy Principle (FEP) is a theory about biological self-organization, in particular about how brains develop through interactions with their environment. Which of the following statements is not consistent with FEP (and explain your answer):       (a) We act to fullfil our predictions about future sensory inputs.       (b) Perception is inference about the environmental causes of our sensations.        (c) Our actions aim to reduce the complexity of our model of the environment.       

    > Statement (c) is not consistent with the FEP formulation of biological self-organization. The Complexity-Accuracy decomposition of the Free Energy reveals that the "data" (observations) is exclusively part of the accuracy term (not in the complexity term). Observations are controlled by actions and hence actions aim to maximize accuracy rather than minimize model complexity.

"""

# ╔═╡ Cell order:
# ╟─da0d3f1c-6e1b-11f0-35a1-0706ebea9dcd
