### A Pluto.jl notebook ###
# v0.20.19

#> [frontmatter]
#> image = "https://imgur.com/FlRMz2f.png"
#> language = "en-US"
#> description = "Introduction to Active Inference and application to the design of synthetic intelligent agents"
#> 
#>     [[frontmatter.author]]
#>     name = "BMLIP"
#>     url = "https://github.com/bmlip"

using Markdown
using InteractiveUtils

# ╔═╡ 0b5b816b-2dd2-4fe8-8f84-4eb2d58b5d59
using RxInfer

# ╔═╡ 97a0384a-0596-4714-a3fc-bf422aed4474
using BmlipTeachingTools

# ╔═╡ 278382c0-d294-11ef-022f-0d78e9e2d04c
title("Intelligent Agents and Active Inference")

# ╔═╡ 9fbae8bf-2132-4a9a-ab0b-ef99e1b954a4
PlutoUI.TableOfContents()

# ╔═╡ 27839788-d294-11ef-30a2-8ff6357aa68b
md"""
## Preliminaries

##### Goal 

  * Introduction to Active Inference and application to the design of synthetic intelligent agents

##### Materials        

  * Mandatory

      * These lecture notes
     
  * Optional

     * Noumenal labs (2025), [WTF is the FEP? A short explainer on the free energy principle](https://www.noumenal.ai/post/wtf-is-the-fep-a-short-explainer-on-the-free-energy-principle)
        *  A concise, accessible introduction to the Free Energy Principle, aimed at demystifying it for a broader audience—matching the tone and intent suggested by the playful but clear title.   

     * De Vries et al. (2025), [Expected Free Energy-based Planning as Variational Inference](https://arxiv.org/pdf/2504.14898)
        * On minimizing expected free energy by variational free energy minimization.

     * Friston et al. (2023), [Path integrals, particular kinds, and strange things](https://doi.org/10.1016/j.plrev.2023.08.016)
        *  The most recent formal developments of the Free Energy Principle (FEP). This paper frames FEP as a principle of least action over trajectories. 

     * Bert de Vries, Tim Scarfe and Keith Duggar (2023), Podcast on [Active Inference](https://youtu.be/2wnJ6E6rQsU?si=I4_k40j42_8E4igP). Machine Learning Street Talk podcast
        * Quite extensive discussion on many aspect regarding the Free Energy Principle and Active Inference, in particular relating to its implementation.

     * Friston et al. (2022), [Designing Ecosystems of Intelligence from First Principles](https://arxiv.org/abs/2212.01354)
       * Friston's vision on the future of AI.

     * Bert de Vries (2021), Presentation on [Beyond deep learning: natural AI systems](https://youtu.be/QYbcm6G_wsk?si=G9mkjmnDrQH9qk5k) (video)
        * 30-minute introduction to active inference from an engineering point of view.
     * Raviv (2018), [The Genius Neuroscientist Who Might Hold the Key to True AI](https://github.com/bmlip/course/blob/main/assets/files/WIRED-Friston.pdf).
        * Interesting article on Karl Friston, who is a leading theoretical neuroscientist working on a theory that relates life and intelligent behavior to physics (and Free Energy minimization). (**highly recommended**)
     * Karl Friston (2011), [What Is Optimal about Motor Control?](https://doi.org/10.1016/j.neuron.2011.10.018), Neuron 72-3, p488-498
        * This work critiques classical optimal control theory for being an insufficient model of biological motor control, and instead advocates active inference as a more suitable framework.
 

 

  

"""

# ╔═╡ 2783a99e-d294-11ef-3163-bb455746bf52
md"""
## Agents

In the previous lessons, we assumed that a data set was given. 

In this lesson, we consider *agents*. An agent is a system that *interacts* with its environment through both sensors and actuators.

Crucially, by acting on the environment, the agent is able to affect the data that it will sense in the future.

  * As an example, by changing the direction where I look, I can affect the (visual) data that will be sensed by my retina.

With this definition of an agent, (biological) organisms are agents, and so are robots, self-driving cars, etc.

In an engineering context, we are particularly interested in agents that behave with a *purpose*, that is, with a specific goal in mind, such as driving a car or trading in financial markets.

In this lesson, we will describe how **goal-directed behavior** by biological (and synthetic) agents can also be interpreted as the minimization of a free energy functional. 

"""

# ╔═╡ aed436fd-6773-4932-a5d8-d01cf99c10ec
challenge_statement("The Door-Key MiniGrid Problem", color="red",header_level=1)

# ╔═╡ 983873d0-e1bc-4e1b-9b6c-3df0a17d83f6
TwoColumn(
	md"""##### Problem
In this example, we consider [the Door-Key MiniGrid problem](https://minigrid.farama.org/environments/minigrid/DoorKeyEnv/), which is part of the challenging MiniGrid environment family. In these environments, an agent is positioned in a gridworld and has to solve a particular task. The red triangle indicates the agent’s location and viewing direction, while the green square marks the target location. """, 
	md"""
![Minigrid environment](https://github.com/bmlip/course/blob/main/assets/ai_agent/minigrid_environment.png?raw=true)
""")


# ╔═╡ 2783b312-d294-11ef-2ebb-e5ede7a86583
md"""
At each timestep, the agent observes the portion of the environment contained within the shaded rectangle in front of its viewing direction. The agent’s task is to find the key, use it to open the door (yellow square in the third column), and then navigate to the target square.

We assume that the agent has complete knowledge of the environmental process dynamics (e.g., how to walk or which action corresponds to picking up a key). However, the locations of the key and door are randomized and therefore unknown.

The challenge is to design an agent that autonomously navigates to the target square. The agent should be defined as a probabilistic model, with its control signals obtained by casting action selection as a Bayesian inference problem.

"""

# ╔═╡ 939e74b0-8ceb-4214-bbc0-407c8f0b2f26
md"""
##### Solution 
At the [end of this lesson](#Challenge-Revisited:-The-Door-Key-MiniGrid-Problem).
"""

# ╔═╡ e3d5786b-49e0-40f7-9056-13e26e09a4cf
md"""
# The Free Energy Principle
"""

# ╔═╡ 2f38a3c9-67d4-4c45-ad0b-e757f62d6a5e
md"""
## Motivation

The human brain is the most complex control system we know. It processes inputs from about ``10`` million sensory neurons through roughly ``10^{11}`` (``=100`` billion) neurons, and drives action via about half a million motor neurons. None of these neurons “know” anything about Fourier transforms, dynamic programming, or backpropagation, they simply minimize variational free energy (VFE).

Remarkably, this autonomous process is sufficient to create a control system that outperforms anything we have ever engineered with our hand-crafted theories of control and signal processing. As engineers, this motivates our work on active inference agents: if nature can produce such superior systems solely through autonomous VFE minimization, perhaps this is also the right path forward in engineering. To achieve truly high-performant adaptive control in volatile environments, it may be necessary to let go of hand-crafted algorithms and build systems that function solely by VFE minimization.

This lecture explores that idea.
"""

# ╔═╡ 4ed2ef1f-4d55-4f9b-b8f5-7f77b5fc62ca
@htl """

<img src="https://github.com/bmlip/course/blob/main/assets/figures/robot-from-disciplines-to-AIF.png?raw=true" alt=" " style="display: block; width: 80%; margin: 0 auto;">

"""

# ╔═╡ 2783c686-d294-11ef-3942-c75d2b559fb3
md"""
## What Drives Intelligent Behavior?

We begin with an example that requires "intelligent" decision-making. Assume that you are an owl and that you're hungry. What are you going to do?

Have a look at [Prof. Karl Friston](https://www.wired.com/story/karl-friston-free-energy-principle-artificial-intelligence/)'s answer in this  [video segment on the cost function for intelligent behavior](https://youtu.be/L0pVHbEg4Yw). (**Do watch the video!**)

![image Friston presentation at CCN-2016](https://github.com/bmlip/course/blob/main/assets/figures/Friston-2016-presentation.png?raw=true)

In his answer, Friston emphasizes that the first step is to search for food, for instance, a mouse. You cannot eat the mouse unless you know where it is, so the first imperative is to reduce your uncertainty about the location of the mouse. In other words, purposeful behavior begins with [epistemic](https://www.merriam-webster.com/dictionary/epistemic) behavior: searching to resolve uncertainty.

This stands in contrast to more traditional approaches to intelligent behavior, such as [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning), where the objective is to maximize a value function of future states, e.g., ``V(s)``, where ``s`` might encode how hungry the agent is. However, this paradigm falls short in scenarios where the optimal next action is to gather information, because uncertainty is not an attribute of states themselves, but of *beliefs* over states, which are expressed as probability distributions.

Therefore, Friston argues that intelligent behavior requires us to optimize a functional ``F[q(s|u)]``, where ``q(s|u)`` is a probability distribution over (future) states ``s`` for a given action sequence ``u``, and ``F`` evaluates the quality of this belief.

Later in his lectures and papers, Friston expands on this belief-based objective ``F`` and formalizes it as a variational free energy functional—laying the foundation for the **Free Energy Principle**. This principle offers a unifying framework that connects biological (or “intelligent”) decision-making and behavior directly to Bayesian inference.

"""

# ╔═╡ 126c3221-34b8-4a8f-b5b5-c14ff4c6a2a1
keyconcept("","Friston’s key insight is that intelligent behavior necessarily involves uncertainty-reducing behavior, which should be framed as the optimization of a functional of beliefs (i.e., probabilities) over future states.")

# ╔═╡ 29592915-cadf-4674-958b-5743a8f73a8b
md"""

## The Free Energy Principle

The Free Energy Principle (FEP) is neither a model nor a theory. Rather, it is a principle, that is, a **methodological framework** for describing the information-processing dynamics that *must* unfold in living systems to keep them within viable (i.e., livable) states over extended periods of time.
  -  Think of the processes continuously occurring in our bodies to maintain an internal temperature between approximately ``36^{\circ}\text{C}`` and ``37^{\circ}\text{C}``, regardless of the surrounding ambient temperature.

The literature on the FEP is widely regarded as difficult to access. It was first formally derived as a specific case of the [Least Action Principle](#The-FEP-is-a-Least-Action-Principle-for-"Things") by Friston in his monograph [Friston (2019), A Free Energy Principle for a Particular Physics (2019)](https://arxiv.org/abs/1906.10184), and more recently presented in a more accessible form in [Friston et al. (2023), Path integrals, particular kinds, and strange things](https://doi.org/10.1016/j.plrev.2023.08.016). For a concise and approachable introduction, the explainer by [Noumenal Labs (2025), WTF is the FEP?](https://www.noumenal.ai/post/wtf-is-the-fep-a-short-explainer-on-the-free-energy-principle) is currently the most accessible resource I am aware of.

In this lecture, we present only a simplified account. According to the FEP, the brain is a generative model for its sensory inputs, such as visual and auditory signals, and **continuously minimizes variational free energy** (VFE) in that model to stay aligned with these observations. Crucially, VFE minimization is the *only* ongoing process, and it underlies perception, learning, attention, emotions, consciousness, intelligent decision-making, etc. 

To illustrate the idea that perception arises from a (variational) inference process—driven by top-down predictions from the brain and corrected by bottom-up sensory inputs—consider the following figure: "The Gardener" by Giuseppe Arcimboldo (ca. 1590).

![](https://github.com/bmlip/course/blob/v2/assets/figures/the-gardener.png?raw=true)

On the left, you’ll likely perceive a bowl of vegetables. However, when the same image is turned upside down, most people first see a gardener’s face.

This perceptual flip arises because the brain’s generative model assigns a much higher probability to being in an environment with upright human faces than with inverted bowls of vegetables. While the sensory input is consistent with both interpretations, the brain’s prior beliefs drive our perception toward seeing upright faces (and upright bowls of vegetables).

In short, the FEP characterizes “intelligent” behavior as the outcome of a VFE minimization process. Next, we derive the dynamics of an *Active Inference* agent—an agent whose behavior is entirely governed by VFE minimization. We will demonstrate that minimizing VFE within a generative model constitutes a sufficient mechanism for producing basic intelligent behavior.
"""

# ╔═╡ 5b8405f7-8878-43dd-8f84-bca17450924f
keyconcept("","The **Free Energy Principle** (FEP) can be seen as a specific instance of the Least Action Principle applied to biological systems, which are technically systems that act to preserve their functional and structural integrity.")

# ╔═╡ 9708215c-72c9-408f-bd10-68ae02e17243
md"""
# The Expected Free Energy Theorem
"""

# ╔═╡ f9b241fd-d853-433e-9996-41d8a60ed9e8
md"""
## Setup of Prior Beliefs

Let's make the above notions more concrete. We consider an agent that interacts with its environment. At the current time ``t``, the agent holds a generative model to predict its future observations, 

```math
\begin{align}
p(y,x,u,\theta) \,, \tag{P1}
\end{align}
```

where ``y`` denotes future observations, ``x`` refers to internal (hidden) future states, ``u`` represents the agent's future actions, and ``\theta`` are model parameters.

Since model (P1) is designed to predict how the future is expected to unfold, we refer to (P1) as the **predictive model**. A typical example is a rollout to the future of a state-space model,

```math
p(y,x,u,\theta) = p(x_t) p(\theta)\underbrace{\prod_{k=t+1}^T  p(y_k|x_k,\theta) p(x_k|x_{k-1},u_k) p(u_k)}_{\text{rollout to the future}}\,.
```

In addition to the predictive model, we assume that the agent holds beliefs ``\hat{p}(x)`` about the *desired* future states. For example, the owl in our earlier example holds the belief that it will not be hungry in the future. We refer to ``\hat{p}(x)`` as the **goal prior**.

Finally, we assume that the agent also maintains **epistemic** (= information-seeking) prior beliefs, denoted by ``\tilde{p}(u)``, ``\tilde{p}(x)``, and ``\tilde{p}(y,x)``, which will be further specified below.

The predictive model, together with the goal and epistemic priors, constitutes the agent’s complete set of prior beliefs about the future.

"""


# ╔═╡ 97136f81-3468-439a-8a22-5aae96725937
md"""

## The Expected Free Energy Theorem

We now state the [Expected Free Energy theorem](https://arxiv.org/pdf/2504.14898#page=8). Let the variational free energy functional ``F[q]`` be defined as
```math
\begin{align}
F[q] = \mathbb{E}_{q(y,x,u,\theta)} \bigg[ \log \frac{q(y,x,u,\theta)}{\underbrace{p(y,x,u,\theta)}_{\text{predictive}} \underbrace{\hat{p}(x)}_{\text{goal}}  \underbrace{\tilde{p}(u) \tilde{p}(x) \tilde{p}(y,x)}_{\text{epistemics}}} \bigg] \,. \tag{F1}
\end{align}
```

Let the agent’s epistemic priors be defined as

```math
\begin{align}
\tilde{p}(u) &= \exp\left( H[q(x|u)]\right) \tag{E1}\\ 
\tilde{p}(x) &= \exp\left( -H[q(y|x)]\right) \tag{E2} \\  
\tilde{p}(y,x) &= \exp\left( D[q(\theta|y,x) , q(\theta|x)]\right) \tag{E3}
\end{align}
```
where ``H[q] = \mathbb{E}_q\left[ -\log q\right]`` is the entropy functional, and ``D[q,p] = \mathbb{E}_q\left[ \log q - \log p\right]`` is the Kullback–Leibler divergence.

Then, the variational free energy ``F[q]`` decomposes as

```math
\begin{align}
F[q] = \underbrace{\mathbb{E}_{q(u)}\left[ G(u)\right]}_{\substack{ \text{expected policy} \\ \text{costs}} } + \underbrace{ \mathbb{E}_{q(y,x,u,\theta)}\left[ \log \frac{q(y,x,u,\theta)}{p(y,x,u,\theta)}\right]}_{\text{complexity}} \tag{F2}\,,
\end{align}
```
where the function ``G(u)``, known as the **Expected Free Energy** (EFE) cost function, is given by 
```math
\begin{align}
G(u) = \underbrace{\underbrace{\mathbb{E}_{q}\bigg[ \log \frac{q(x|u)}{\hat{p}(x)}\bigg]}_{\text{risk}}}_{\text{scores goal-driven behavior}} + \underbrace{\underbrace{\mathbb{E}_{q}\bigg[ \log \frac{1}{q(y|x)}\bigg]}_{\text{ambiguity}} - \underbrace{\mathbb{E}_{q}\bigg[ \log \frac{q(\theta|y,x)}{q(\theta|x)}\bigg]}_{\text{novelty}}}_{\text{scores information-seeking behavior}} \,. \tag{G1}
\end{align}
```


"""

# ╔═╡ 4e990b76-a2fa-49e6-8392-11f98d769ca8
details("Click for proof of the EFE Theorem",
    md"""

    For the following proof, see also Appendix A in [De Vries et.al., Expected Free Energy-based Planning as Variational Inference (2025)](https://arxiv.org/pdf/2504.14898#page=15).
    		
    ```math
    \begin{flalign}
        F[q] &= E_{q(y x  u \theta)}\bigg[ \log \frac{q(y x  u \theta)}{p(y x  u \theta)  \hat{p}(x) \tilde{p}(u) \tilde{p}(x)  \tilde{p}(yx)} \bigg] \\
        &= E_{q(u)}\bigg[ \log \frac{q(u)}{p(u)} 
        + \underbrace{E_{q(yx\theta | u)}\big[ \log \frac{q(y x \theta | u)}{p(yx \theta|u)  \hat{p}(x) \tilde{p}(u) \tilde{p}(x)  \tilde{p}(yx)}\big]}_{B(u)}  
         \bigg] \; &&\text{(C1)}\\
         &= E_{q(u)}\bigg[ \log \frac{q(u)}{p(u)} 
        + \underbrace{G(u) +E_{q(yx\theta | u)} \big[\log \frac{q(yx\theta|u)}{p(yx\theta|u)}\big]}_{=B(u) \text{ if conditions (E1), (E2) and (E3) hold}}  
         \bigg] &&\text{(C2)} \\
        &= E_{q(u)}\big[ G(u)\big]+ E_{q(yx u\theta)}\bigg[\log \frac{q(yx u \theta)}{p(yx u \theta) }\bigg]\,,   
    \end{flalign}
    ```		
    if the conditions in Eqs. ``(\mathrm{E}1)``, ``(\mathrm{E}2)``, and ``(\mathrm{E}3)`` hold.
    	
    In the above derivation, we still need to prove the equivalence of ``B(u)`` in
    Eqs. ``(\mathrm{C}1)`` and ``(\mathrm{C}2)``, which we address next. 
    In the following, all expectations are with respect to ``q(y,x,\theta|u)`` unless otherwise indicated. 

    ```math
    \begin{flalign}
    B(&u) = E\bigg[ \log \frac{ \overbrace{q(yx\theta|u)}^{\text{posterior}} }{ \underbrace{p(yx\theta|u)}_{\text{predictive}} \underbrace{\hat{p}(x)}_{\text{goals}} \underbrace{\tilde{p}(u) \tilde{p}(x) \tilde{p}(yx)}_{\text{epistemic priors}}} \bigg]  \; &&\text{(C3)} \\
    &= \underbrace{ E\bigg[\log\bigg( \underbrace{\frac{q(x|u)}{\hat{p}(x)}}_{\text{risk}}\cdot \underbrace{\frac{1}{q(y|x  )}}_{\text{ambiguity}} \cdot \underbrace{\frac{ q(\theta|x)}{ q(\theta|yx )}}_{-\text{novelty}} \bigg) \bigg] }_{G(u) = \text{Expected Free Energy}} +   \\
    &\quad + E\bigg[ \log\bigg( \underbrace{\frac{\hat{p}(x) q(y|x ) q(\theta| yx)}{q(x|u) q(\theta|x)}}_{\text{inverse factors from }G(u)} \cdot \underbrace{\frac{q(yx\theta|u)}{p(yx\theta|u) \hat{p}(x) \tilde{p}(u) \tilde{p}(x) \tilde{p}(yx) }}_{\text{leftover factors from (C3)}} \bigg)\bigg] \notag \\
    &= G(u) + \underbrace{E\bigg[ \log \frac{q(yx\theta|u)}{p(yx\theta|u)}\bigg]}_{=C(u)} + \underbrace{E\bigg[ \log  \frac{q(y|x ) q(\theta|yx)}{q(x|u) q(\theta|x) \tilde{p}(u) \tilde{p}(x) \tilde{p}(yx)} \bigg]}_{\text{choose epistemic priors to let this vanish}} \\
    &= G(u) + C(u) +  \\
    &\quad + E\bigg[\log \frac{1}{q(x|u) \tilde{p}(u)} \bigg] + E\bigg[ \log  \frac{q(y|x)}{\tilde{p}(x)} \bigg] + E\bigg[ \log  \frac{q(\theta|yx)}{q(\theta|x) \tilde{p}(yx) } \bigg] \notag \\
    &= G(u) + C(u) +  \\
    &\qquad + \sum_{y\theta} q(y\theta|x) \bigg( \underbrace{\underbrace{-\sum_x q(x|u) \log q(x|u)}_{= H[q(x|u)]} - \sum_x q(x|u) \log \tilde{p}(u)}_{=0 \text{ if }\tilde{p}(u) = \exp(H[q(x|u)])}\bigg) \\
    &\qquad + \sum_{x} q(x|u) \bigg( \underbrace{\underbrace{\sum_{y} q(y|x) \log q(y|x)}_{= -H[q(y|x)]} - \sum_{y} q(y|x) \log \tilde{p}(x)}_{=0 \text{ if }\tilde{p}(x) = \exp(-H[q(y|x)])} \bigg)   \notag \\
    &\qquad + \sum_{yx} q(yx|u) \bigg( \underbrace{\underbrace{\sum_\theta q(\theta|yx) \log \frac{q(\theta|yx)}{q(\theta|x)}}_{D[q(\theta|yx),q(\theta|x)]} - \sum_\theta q(\theta|yx) \log \tilde{p}(yx)}_{=0 \text{ if } \tilde{p}(yx) = \exp(D[q(\theta|yx),q(\theta|x)])} \bigg) \notag \\
    &= G(u) + E_{q(yx\theta|u)}\bigg[ \log \frac{q(yx\theta|u)}{p(yx\theta|u)}\bigg] \,,
    \end{flalign}
    ```
    if Eqs. (E1), (E2), and (E3) hold.

    """)

# ╔═╡ aec84a6d-33fc-4541-80c0-091998f8c4d1
begin
    ambiguity_as_expected_entropy = details("Click to show derivation of ambiguity as an expected entropy",
md""" Starting from Eq.(G1),
```math										
\begin{align}
  \mathbb{E}_{q(y,x|u)}\bigg[ \log \frac{1}{q(y|x)}\bigg] &= \mathbb{E}_{q(x|u)}\bigg[ \mathbb{E}_{q(y|x)} \big[\log \frac{1}{q(y|x)}\big] \bigg] \\ 
  &= \mathbb{E}_{q(x|u)}\left[H[q(y|x)] \right]
\end{align}		
```	
""")

novelty_as_mutual_information = details("Click to show derivation of novelty in terms of mutual information",
md""" Starting from Eq.(G1), 
```math
\begin{align}
    \mathbb{E}_{q(y,x,\theta|u)}\bigg[ \log \frac{q(\theta|y,x)}{q(\theta|x)}\bigg] &= \mathbb{E}_{q(y,\theta|x) q(x|u)}\bigg[ \log \frac{q(\theta|y,x)}{q(\theta|x)}\bigg] \\  
    &= \mathbb{E}_{q(x|u)}\bigg[ \mathbb{E}_{q(y,\theta|x)} \big[ \log \frac{q(\theta|y,x)}{q(\theta|x)} \big] \bigg] \\ 
    &= \mathbb{E}_{q(x|u)}\bigg[ \underbrace{\mathbb{E}_{q(y,\theta|x)} \big[ \log \frac{q(\theta,y|x)}{q(\theta|x) q(y|x)} \big]}_{I[\theta,y\,|x]} \bigg] \\  
    &= \mathbb{E}_{q(x|u)}\big[ I[\theta,y\,|x] \big]
        \end{align}
```									
""")
end;

# ╔═╡ aaa07dc5-9105-4f70-b924-6e51e5c36600
md"""
## Interpretation of Expected Free Energy ``G(u)``

``G(u)`` is a cost function defined over a sequence of future actions ``u = (u_{t+1},u_{t+2}, \ldots, u_{T})``, commonly referred to as a **policy**. ``G(u)`` decomposes into three distinct components:

###### risk
  - The risk term is the KL divergence between ``q(x|u)``, the *predicted* future states under policy ``u``, and ``\hat{p}(x)``, the *desired* future states (the goal prior). As a result, ``G(u)`` penalizes policies that lead to expectations which diverge from the agent’s preferences — that is, from what the agent wants to happen.

###### ambiguity
  - Ambiguity can be expressed as ``\mathbb{E}_{q(x|u)}\left[H[q(y|x)] \right]``, which quantifies the expected entropy of future observations ``y``, under policy ``u``. It measures how ambiguous or noisy the relationship is between hidden states ``x`` and observations ``y``. Policies with low ambiguity are preferable because they lead to observations that are more informative about the hidden state, thus facilitating more accurate inference and better decision-making.
  - $(ambiguity_as_expected_entropy)


###### novelty
  - The novelty term can be worked out to ``\mathbb{E}_{q(x|u)}\big[ I[\theta,y\,|x] \big]``, where ``I[\theta,y\,|x]`` is the [mutual information](https://en.wikipedia.org/wiki/Mutual_information) between parameters ``\theta`` and observations ``y``, given states ``x``. Novelty complements the ambiguity term. While ambiguity scores information-seeking behavior aimed at reducing uncertainty about hidden states ``x``, the novelty term extends this idea to parameters ``\theta`` of the generative model. It encourages policies that are expected to lead to observations that reduce uncertainty about ``\theta``, i.e., learning about the structure or dynamics of the environment itself.
  - $(novelty_as_mutual_information)

Clearly, policies with lower Expected Free Energy are preferred. Such policies strike a balance between goal-directed behavior—by minimizing risk—and information-seeking behavior—by minimizing ambiguity (to infer hidden states) and maximizing novelty (to learn about model parameters). This unified objective naturally promotes both exploitation and exploration.

"""

# ╔═╡ bed6a9bd-9bf8-4d7b-8ece-08c77fddb6d7
md"""
# Active Inference
"""

# ╔═╡ ef54a162-d0ba-47ef-af75-88c92276ed66
md"""
## Optimal Planning by Variational Inference

Assume that our agent is continually engaged in minimizing its variational free energy ``F[q]``, defined in Eq. (F2). This process tracks the following optimal posterior beliefs over policies,

```math
\begin{align}
q^*(u) &\triangleq \arg\min_q F[q]  \\ 
&= \sigma\left( - G(u) -C(u) -P(u) \right) \,, \tag{Q*}
\end{align}
```
where
- ``\sigma(\cdot)`` denotes the softmax function,
- ``G(u)`` is the expected free energy, defined in Eq. (G1), scoring both goal-directed and epistemic value of each policy,
- ``C(u) = \mathbb{E}_{q(y,x,\theta|u)}\Big[ \log \frac{q(y,x,\theta|u)}{p(y,x,\theta|u)}\Big]`` is a complexity term, capturing divergence between the variational posterior and prior beliefs for a given policy ``u``,     
- ``P(u) = -\log p(u)`` reflects prior preferences over policies from the generative model.

"""

# ╔═╡ 94391132-dee6-4b22-9900-ba394f4ad66b
details(md"""Click for proof of ``q^*(u)``""",
    md"""
    Starting from Eq. (F2), 
    ```math
    \begin{align}
    F[q] &=\mathbb{E}_{q(u)}\left[ G(u)\right] + \mathbb{E}_{q(y,x,u,\theta)}\left[ \log \frac{q(y,x,u,\theta)}{p(y,x,u,\theta)}\right] \tag{F2} \\  
    &=\mathbb{E}_{q(u)}\bigg[G(u) + \underbrace{\mathbb{E}_{q(y,x,\theta|u)}\Big[ \log \frac{q(y,x,\theta|u)}{p(y,x,\theta|u)}\Big]}_{C(u)}	+ \log \frac{q(u)}{p(u)}  \bigg]	\\
    &=\mathbb{E}_{q(u)}\bigg[ \log \frac{1}{\exp(-G(u))} + \log \frac{1}{\exp(-C(u))} + \log \frac{q(u)}{\exp(-P(u))}  \Big]	\bigg]	\\
    &= 	\mathbb{E}_{q(u)}\bigg[ \log \frac{q(u)}{\exp(-G(u) - C(u) -P(u) )}\bigg]	
    \end{align}
    ```
    which is (proportional to) a Kullback-Leibler divergence that is minimized for 
    ```math
    \begin{align}
    q^*(u) = \sigma\left(-G(u) - C(u) -P(u) \right)	\,.
    \end{align}
    ```	
    """)

# ╔═╡ a8c88dff-b10c-4c25-8dbe-8f04ee04cffa
md"""
## An Active Inference Agent!

Eq. (Q*) marks a central result: an agent that minimizes the variational free energy ``F[q]``, as defined in Eq.(F2), naturally selects policies that are goal-directed, epistemically valuable, and computationally parsimonious.
- Goal-directed policies **minimize risk** by steering predicted future states toward preferred or desired outcomes.
- Epistemically valuable policies reduce uncertainty by favoring informative observations (**low ambiguity**) and supporting model learning (**high novelty**).
- Computationally parsimonious policies **minimize complexity**, ensuring that posterior beliefs remain close to prior expectations. This limits the extent of belief updating, thereby *conserving computational resources* and reducing inference overhead.

The process of minimizing ``F[q]`` is called an **Active Inference** (AIF) process, and an agent that realizes this process is referred to as an **active inference agent**. The “active” aspect highlights that an AIF agent does not passively consume a fixed data set, but instead actively selects its own data set through purposeful interaction with the environment.

From an engineering perspective, if one accepts that effective decision-making systems should exhibit goal-directed behavior, epistemic exploration, and computational efficiency, then an AIF agent can be viewed as an "intelligent" controller. Given a well-defined set of predictive, goal-oriented, and epistemic prior beliefs, the agent’s behavior follows directly from the minimization of variational free energy. In this sense, the agent acts rationally—or Bayes-optimally—with respect to its design objectives and internal model.

"""

# ╔═╡ 63609dc2-2413-4822-8401-2d2ba22adfce
keyconcept("","The process underlying naturally intelligent behavior is termed **Active Inference** (AIF). It can be described as the minimization of a variational free energy under a generative model that incorporates a rollout into the future.")

# ╔═╡ 5b66f8e5-4f01-4448-82e3-388bc8ea31de
md"""
## Interpretation of the Epistemic Priors

In the formulation introduced in Eq. (E1), the epistemic prior
``\tilde{p}(u) = \exp\big(H[q(x | u)]\big)`` biases the agent toward selecting policies ``u`` that maximize the entropy of the predicted future states ``x``.

This reflects an information-seeking preference: high entropy over future states implies that the agent is actively maintaining flexibility and postponing premature commitment. Rather than treating uncertainty as something to avoid, this formulation encourages the agent to seek out policies that enable adaptation as new observations arrive.

Additionally, the epistemic prior ``\tilde{p}(x) = \exp(−H[q(y|x)])``
in (E2), favors policies that reduce uncertainty about future states by
selecting observations that are informative about them. Together, ``\tilde{p}(u)`` and ``\tilde{p}(x)`` induce a **bias toward ambiguity-minimizing behavior**.

Similarly, the epistemic priors ``\tilde{p}(u)`` and ``\tilde{p}(y,x)`` from (E1) and (E3), jointly shape a **preference for policies that maximize novelty**, i.e., that are expected to be informative about the parameters of the generative model.

"""

# ╔═╡ 65e34124-fda2-4a60-9722-ece2f68a39d9
md"""
## Realization by Reactive Message Passing

An AIF agent can be efficiently realized by an autonomous reactive message passing process in a Forney-style Factor Graph (FFG) representation of (a rollout to the future of) the generative model, augmented with goal and epistemic priors.
"""

# ╔═╡ 6e14d8e3-9dc2-44fb-9fa5-c48f10cb1076
Resource("https://github.com/bmlip/course/blob/v4/assets/figures/AIF-generative-model-as-FFG.png?raw=true", :alt => "FFG for an AIF agent", :style => "background: white; border-radius: 1em;")

# ╔═╡ bece6e3d-57b2-4bf4-8bf8-d5a11ad8983f
md"""
In the above figure, the agent's generative (predictive) model
```math
\prod_{k=1}^T p(y_k|x_k) p(x_k|x_{k-1},u_k)\,,
```
is represented by the white nodes in the factor graph. The initial and desired final states are constrained by initial and goal priors ``\hat{p}(x_0|x^+)`` and ``\hat{p}(x_T|x^+)``, which are typically generated by a higher-level state ``x^+`` and shown here as orange and blue nodes, respectively.

At time ``k = 0``, the agent is tasked to infer a future action sequence (a "policy") ``u_{1:T}`` such that the posterior ``q(x_T|y_{1:T})`` matches the goal prior ``\hat{p}(x_T|x^+)`` as closely as possible. Inference proceeds entirely via reactive message passing in the factor graph, with no external control.

The figure shows the state of the system at time ``t``, after having executed actions ``u_{1:t}`` and having observed ``y_{1:t}``. The future rollout for steps ``t+1`` to ``T`` terminates the predictive model (white) with both epistemic priors (green and red nodes) and the goal prior (blue node). As new actions are selected and new observations are sensed, the epistemic priors are replaced by posteriors (small black boxes), enabling an ongoing free energy minimization process.

"""

# ╔═╡ 64474167-bf52-456c-9099-def288bd17bf
challenge_solution("The Door-Key MiniGrid Problem", header_level=1,color="green")

# ╔═╡ 2784f45e-d294-11ef-0439-1903016c1f14
md"""

We now return to the Door-Key MiniGrid problem from the [beginning of this lesson](#Challenge:-The-Door-Key-MiniGrid-Problem). Below, we present RxInfer pseudocode for the augmented generative model of the MiniGrid agent. The key insight from this code fragment is that there is no explicit “algorithmic” planning code: in an AIF agent, all processes reduce to VFE minimization (of [Eq. (F1)](#The-Expected-Free-Energy-Theorem)), which is carried out automatically by the RxInfer inference engine.

*NB: A complete implementation of this solution is available [here](https://github.com/biaslab/EFEasVFE). Please note that the full code is a research prototype and not yet production-ready.*
"""



# ╔═╡ 1f468b26-b4fe-4f61-af41-0a15fdc44365
@model function minigrid_aif(prior_state, prior_key, prior_door, target_location, horizon, transition_params, observation_params)
	initial_state ~ prior_state
	key_location ~ prior_key
	door_location ~ prior_door

	prev_state = initial_state
	for t in 1:horizon
		action[t] ~ ExplorationPrior()
		state[t] ~ AmbiguityPrior()
		state[t] ~ DiscreteTransition(prev_state, transition_params, key_location,       door_location, action)
		observation[t] ~ DiscreteTransition(state[t], observation_params, 			     key_location, door_location)
	end
	state[end] ~ Goal(target_location)
end

# ╔═╡ 8ac328de-b7ba-457f-add6-9af506832282
md"""
The animation below is a recording of representative agent behavior during the VFE minimization process. Initially, the agent rotates to search for the key—an uncertainty-reducing strategy **driven by the ambiguity term** of the Expected Free Energy. Once the key’s location is revealed, the agent retrieves it and proceeds to the goal location. This constitutes goal-directed behavior that **minimizes the risk term** in the EFE. Because the environmental dynamics are assumed to be known, no novelty-driven exploration occurs in this example.
"""

# ╔═╡ 651a4081-68f1-4237-9503-d4e28969a836
Resource("https://github.com/bmlip/course/raw/refs/heads/main/assets/figures/minigrid%20loop.mp4", :autoplay => true, :loop=>true)

# ╔═╡ f4509603-36be-4d24-8933-eb7a705eb933
md"""
# Discussion
"""

# ╔═╡ 8d7058c4-0e13-4d05-b131-32b1f118129f
md"""
The Free Energy Principle and active inference are deep and fast-moving areas of research. They bring fresh ideas to intelligent reasoning, control, and AI, with exciting applications in robotics, adaptive systems, cognitive modeling, and more. There’s a lot to unpack, but in this lecture, we’ve only had time to scratch the surface. To wrap up, we’ll end with a few closing thoughts.
"""

# ╔═╡ 1c53d48b-6950-4921-bf03-292b5ed8980e
md"""
## Comparison Decision-theoretic vs Active Inference Agents

The idea of framing decision-making and planning as the minimization of expected cost over future states has become foundational across many disciplines, including machine learning (e.g., reinforcement learning), control theory (e.g., model-predictive and optimal control), and economics (e.g., utility theory and operations research). In what follows, we will refer to such systems collectively as *decision-theoretic* (DT) agents.

AIF agents fundamentally differ from DT agents in that variational free energy (VFE) minimization is the sole underlying process. As a result, policies are evaluated based on a function of beliefs about states, rather than directly on the states themselves. The FEP formally captures this distinction.

From an engineering perspective, what is gained by moving from DT to AIF agents? While our treatment here is necessarily brief and not intended as a comprehensive academic assessment, several key advantages already stand out:

- **A principled grounding in fundamental physics**
  - If we aim to understand how brains—human or animal—give rise to intelligent behavior, we must start from the premise that they operate entirely within the laws of physics. The FEP is consistent with this physical grounding.

- **Balanced goal-directed and information-seeking behavior**
  - The epistemic components that emerge naturally from the EFE functional often need to be added through ad hoc mechanisms in decision-theoretic frameworks that do not explicitly score beliefs about future states.

- **No need for task-specific reward (or value) functions**
  - In DT agents, a recurring question is: where do the reward functions come from? These functions are typically hand-crafted. In an AIF agent, preferences are encoded as prior distributions over desired outcomes. These priors can be parameterized and updated through hyper-priors and Bayesian learning at higher levels of the generative model, allowing agents to adapt their preferences on the fly, rather than relying on externally specified reward functions.

- **A universal cost function for all problems**
  - A related limitation of the DT systems is that the value function must be specified anew for each problem. Brains, however, cannot afford to construct a different value function for every task, given that thousands of novel problems are encountered daily. In contrast, AIF agents employ the same free-energy functional ``F`` that measures the quality of the beliefs, for all problems. The structure of ``F`` (complexity minus accuracy) does not change and applies universally.

- **AIF agents are explainable and trustworthy by nature**
  - Explainability and trustworthiness are critical concerns in AI, for instance, in medical AI applications. An AIF agent’s reasoning process is Bayes-optimal, and therefore logically consistent and inherently *trustworthy*. Crucially, domain-specific knowledge and inference are cleanly separated: all domain-specific assumptions reside in the model. As a result, the agent’s behavior can be *explained* as the logical (Bayesian) consequence of its generative model.

- **Robustness by realization as a reactive message passing process!**
  - In contrast to decision-theoretic (DT) agents, an active inference (AIF) agent can be fully realized as a reactive variational message passing (RMP) process, since variational free energy (VFE) minimization is the only ongoing process. RMP is an event-driven, fully distributed process—both in time and space—that exhibits robustness to fluctuating computational resources. It “lands softly” when resources such as power, data, or time become limited. As a result, an AIF agent continues to function during power dips, handles missing or noisy observations gracefully, and can be interrupted at any time during decision-making without catastrophic failure, making it naturally suited for real-world, resource-constrained environments.

- **Easy to code** 
  - Since VFE minimization can be automated by a toolbox, the engineer’s main task is to specify the generative model and priors, typically achievable within a single page of code. However, because a professional-level automated VFE minimization toolbox is not (yet) available, this “easy-to-code” feature has not (yet) been realized.

- **Other advantages**
  - Additional advantages include the potential for scalability, particularly in real-time applications. Realizing this potential will require further research into efficient, real-time message passing capabilities that are difficult to match in frameworks that cannot be implemented as reactive message passing processes.

While the advantages listed above hold great promise for the future of synthetic AIF agents in solving complex engineering problems, it’s important to acknowledge current limitations. The vast majority of engineers and scientists have been trained within DT frameworks, and the **tooling and methodologies for DT agents are far more mature**. For many practical problems, several of the above-mentioned advantages of AIF agents have yet to be conclusively demonstrated in real-world applications.


"""

# ╔═╡ e3b4d8cc-d028-4e00-b4f3-369f1601510d
keyconcept("","In principle, active inference offers a unified way to address several long-standing challenges faced by current approaches to agentic AI. In practice, however, the available toolboxes for active inference remain less mature than those for more conventional AI paradigms, such as deep learning, large language models, and reinforcement learning.")

# ╔═╡ d823599e-a87f-4586-999f-fbbd99d0db65
md"""
## The FEP: A New Frontier for Understanding Intelligent Behavior


The FEP is often misunderstood as a scientific theory that counterexamples can falsify. In reality, the **FEP is a principle**, a general modeling framework for describing the dynamics of systems that exhibit ("life-like") attractor dynamics, repeatedly returning to states that preserve their functional organization and structural integrity. According to the FEP, such systems can be interpreted as performing variational Bayesian inference in a generative model, where the goal priors correspond to the system’s attractors.

In this lecture, we’ve seen how the FEP can be used to describe the dynamics of rational AI agents, but its reach goes far beyond AI and control. As a unifying framework for understanding adaptive self-organization, the FEP touches neuroscience, biology, cognition, and even physics. 

For example, the Expected Free Energy used to evaluate policies is not just an arbitrary cost function—it follows naturally from common assumptions in fundamental physics. EFE-based policy scoring also makes sense from a philosophical standpoint: if minimizing variational free energy is the only process driving the system, then it is a logical consequence to rank policies by how much VFE we expect them to minimize in the future. 

Looking ahead to the future of artificial intelligence, adaptive robotics, and agentic AI, the Free Energy Principle stands out as a framework with the potential to transform not only how we build intelligent systems, but how we fundamentally understand their nature, purpose, and place within the broader landscape of self-organizing life.

"""

# ╔═╡ df4950bc-4532-44e3-b8c7-18dd62fdfb09
md"""
# Summary
"""

# ╔═╡ c51232ba-9908-476a-949f-d5d0949d7960
keyconceptsummary()

# ╔═╡ aed24f2d-105e-4583-b9d6-24d1886002d8
exercises(header_level=1)

# ╔═╡ acefdbf6-1beb-4ce5-9106-0fc7be63dabe
md"""
There are no more exercises. If you understand the Free Energy Principle and active inference, you now hold a lens for seeing into the mechanics of life, consciousness, and intelligent behavior. The next insights are yours to discover.
"""

# ╔═╡ 6d697856-cc58-4d6a-afd3-c0c6bfbc0d88
md"""
# Optional Slides
"""

# ╔═╡ 8627b90a-79d8-43bf-8d53-1043f6e816ab
md"""

## The FEP is a Least Action Principle for "Things"

Almost all of physics can be described as processes that minimize energy differences. This is formalized in the celebrated [Principle of Least Action](https://en.wikipedia.org/wiki/Action_principles) (PLA).

For example, Newton’s second law ``F = ma`` [can be derived from minimizing the "action"](https://vixra.org/pdf/2501.0036v1.pdf)—the time integral of the difference between kinetic and potential energy for a particle. Many branches of physics that describe motion can be formulated through similar derivations.

"""

# ╔═╡ 58c51f5e-69a0-4802-892b-0a52462c4e55
Resource("https://github.com/bmlip/course/blob/v4/assets/figures/FEP.png?raw=true", :style => "background: white; border-radius: 1em;")

# ╔═╡ 14ef2403-cba3-4e76-91c7-4655656a632d
md"""

The Free Energy Principle (FEP) can be viewed as a kind of PLA as well—one that governs the necessary “motion” of beliefs (ie, necessary information processing) in systems that maintain their structural and functional integrity over time. All living systems fall under its scope.

"""

# ╔═╡ 54cc2500-f06f-4f39-95f3-8d9d09c37234
md"""
## Active Inference and The Scientific Inquiry Loop

In the [Machine Learning Overview lecture](https://bmlip.github.io/course/lectures/Machine%20Learning%20Overview.html), we introduced a picture illustrating the [Scientific Inquiry Loop](https://bmlip.github.io/course/lectures/Machine%20Learning%20Overview.html#Machine-Learning-and-the-Scientific-Inquiry-Loop). 

Active inference completes this “scientific loop” as a fully variational inference process. Under the FEP, all processes, including state updating, learning, and trial design (in living systems: perception, learning, and control/behavior, respectively) are driven by VFE minimization. Bayesian probability theory, together with the FEP, provides all the equations needed to run the process of scientific inquiry.

The figure below illustrates this process. We do not depict the epistemic priors as an external input, since they can be computed internally by the agent itself.
"""

# ╔═╡ b900b7d4-3a49-4651-a94b-74fdbbf094d9
Resource("https://github.com/bmlip/course/blob/v4/assets/figures/AIF-agent-loop.png?raw=true", :style => "background: white; border-radius: 1em;")

# ╔═╡ cd2dafbe-5131-4ee0-8eb7-33bd04f3133f
md"""
If an agent has no goal priors, then active inference reduces to an automated (Bayes-optimal) scientist: the agent’s generative model will converge to a veridical (“true”) description of the environment. With goal priors, however, an active inference agent becomes a (Bayes-optimal) engineer: its model converges to beliefs that generate purposeful behavior. For example, the goal prior “I will not get hit by a car” leads to the inference of actions that allow safe crossing of the road. In the same way, carefully structured goal priors enable the brain to develop solutions for tasks such as object recognition, locomotion, and speech generation.

In short, **AIF is an automated (Bayes-optimal) engineering design loop**.

The challenge is computational efficiency: the human brain runs on about 20 [W], with the neocortex using just 4 [W], roughly the power of a bicycle light, yet performs tasks that would consume millions of times more power on current silicon hardware.

"""

# ╔═╡ be0dc5c0-6340-4d47-85ae-d70e06df1676
md"""
# Code
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BmlipTeachingTools = "656a7065-6f73-6c65-7465-6e646e617262"
RxInfer = "86711068-29c9-4ff7-b620-ae75d7495b3d"

[compat]
BmlipTeachingTools = "~1.3.1"
RxInfer = "~4.6.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "86b15ea5fe65fe39dea8ac5342d4e576fcb3219b"

[[deps.ADTypes]]
git-tree-sha1 = "27cecae79e5cc9935255f90c53bb831cc3c870d7"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.18.0"

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
git-tree-sha1 = "7e35fca2bdfba44d797c53dfe63a51fabf39bfc0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.4.0"
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
git-tree-sha1 = "d2cd034553ee6ca084edaaf8ed6c9d50fd01555d"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.21.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
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
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "355ab2d61069927d4247cd69ad0e1f140b31e30d"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.12.0"
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
git-tree-sha1 = "5b723bf6b1081cab4d263e425be097224e0f434f"
uuid = "b4ee3484-f114-42fe-b91c-797d54a0c67e"
version = "1.5.8"
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

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "79e651aa489a7879107d66e3d1948e9aa1b4055e"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.7.2"

    [deps.BlockArrays.extensions]
    BlockArraysAdaptExt = "Adapt"
    BlockArraysBandedMatricesExt = "BandedMatrices"

    [deps.BlockArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"

[[deps.BmlipTeachingTools]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "PlutoTeachingTools", "PlutoUI", "Reexport"]
git-tree-sha1 = "806eadb642467b05f9d930f0d127f1e6fa5130f0"
uuid = "656a7065-6f73-6c65-7465-6e646e617262"
version = "1.3.1"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "51f4c10ee01bda57371e977931de39ee0f0cdb3e"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.0"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "cee8104904c53d39eb94fd06cbe60cb5acde7177"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.0.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

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
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

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

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

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
git-tree-sha1 = "529bebbc74b36a4cfea09dd2aecb1288cd713a6d"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.9"

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
git-tree-sha1 = "3bc002af51045ca3b47d2e1787d6ce02e68b943a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.122"

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

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.ExponentialFamily]]
deps = ["BayesBase", "BlockArrays", "Distributions", "DomainSets", "FastCholesky", "FillArrays", "ForwardDiff", "HCubature", "HypergeometricFunctions", "IntervalSets", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "PositiveFactorizations", "Random", "SparseArrays", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "TinyHugeNumbers"]
git-tree-sha1 = "8351e116e111c97ad57718851da00ae5a5f92e0c"
uuid = "62312e5e-252a-4322-ace9-a5f4bf9b357b"
version = "2.1.1"

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
git-tree-sha1 = "0044e9f5e49a57e88205e8f30ab73928b05fe5b6"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.1.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "d60eb76f37d7e5a40cc2e7c36974d864b82dc802"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.1"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "173e4d8f14230a7523ae11b9a3fa9edb3e0efd78"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.14.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "9340ca07ca27093ff68418b7558ca37b05f8aeb1"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.29.0"

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

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "ba6ce081425d0afb2bedd00d9884464f764a9225"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.2.2"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GaussQuadrature]]
deps = ["SpecialFunctions"]
git-tree-sha1 = "eb6f1f48aa994f3018cbd029a17863c6535a266d"
uuid = "d54b0c1a-921d-58e0-8e36-89d8069c0969"
version = "0.5.8"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.GraphPPL]]
deps = ["BitSetTuples", "DataStructures", "Dictionaries", "MacroTools", "MetaGraphsNext", "NamedTupleTools", "Static", "StaticArrays", "TupleTools", "Unrolled"]
git-tree-sha1 = "db4aece54ddddaa9e8d2880eb7cfc6f29bd1a650"
uuid = "b3f8163a-e979-4e85-b43e-1f63d8c8b42c"
version = "4.6.5"

    [deps.GraphPPL.extensions]
    GraphPPLDistributionsExt = "Distributions"
    GraphPPLGraphVizExt = "GraphViz"
    GraphPPLPlottingExt = ["Cairo", "GraphPlot"]

    [deps.GraphPPL.weakdeps]
    Cairo = "159f3aea-2a34-519c-b102-8c37f9878175"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    GraphPlot = "a2cc645c-3eea-5389-862e-a155d0052231"
    GraphViz = "f526b714-d49f-11e8-06ff-31ed36ee7ee0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "7a98c6502f4632dbe9fb1973a4244eaa3324e84d"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.1"

[[deps.HCubature]]
deps = ["Combinatorics", "DataStructures", "LinearAlgebra", "QuadGK", "StaticArrays"]
git-tree-sha1 = "19ef9f0cb324eed957b7fe7257ac84e8ed8a48ec"
uuid = "19dc6840-f33b-545b-b366-655c7e3ffd49"
version = "1.7.0"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e6fe50ae7f23d171f44e311c2960294aaa0beb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.19"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

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

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JLD2]]
deps = ["ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "da2e9b4d1abbebdcca0aa68afa0aa272102baad7"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.2"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

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
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

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

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "SparseArrays"]
git-tree-sha1 = "79ee64f6ba0a5a49930f51c86f60d7526b5e12c8"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "2.8.0"

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
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.11.1+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "4adee99b7262ad2a1a4bbbc59d993d24e55ea96f"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.4.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

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
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
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
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3cce3511ca2c6f87b19c34ffc623417ed2798cbd"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.10+0"

[[deps.MetaGraphsNext]]
deps = ["Graphs", "JLD2", "SimpleTraits"]
git-tree-sha1 = "c3f7e597f1cf5fe04e68e7907af47f055cad211c"
uuid = "fa8bd995-216d-47f1-8a91-f3b68fbeb377"
version = "0.7.4"

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
version = "2025.5.20"

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
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
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

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "d922b4d80d1e12c658da7785e754f4796cc1d60d"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.36"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

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

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "dacc8be63916b078b592806acd13bb5e5137d7e9"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3faff84e6f97a7f18e0dd24373daa229fd358db5"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.73"

[[deps.PolyaGammaHybridSamplers]]
deps = ["Distributions", "Random", "SpecialFunctions", "StatsFuns"]
git-tree-sha1 = "9f6139650ff57f9d8528cd809ebc604c7e9738b1"
uuid = "c636ee4f-4591-4d8c-9fae-2dea21daa433"
version = "1.2.6"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.ReactiveMP]]
deps = ["BayesBase", "DataStructures", "DiffResults", "Distributions", "DomainIntegrals", "DomainSets", "ExponentialFamily", "FastCholesky", "FastGaussQuadrature", "FixedArguments", "ForwardDiff", "HCubature", "LazyArrays", "LinearAlgebra", "MacroTools", "MatrixCorrectionTools", "Optim", "PolyaGammaHybridSamplers", "PositiveFactorizations", "Random", "Rocket", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "TinyHugeNumbers", "Tullio", "TupleTools", "Unrolled"]
git-tree-sha1 = "dcd2f85ba9f2f7be1b1b31708db889d078b161f6"
uuid = "a194aa59-28ba-4574-a09c-4a745416d6e3"
version = "5.6.2"

    [deps.ReactiveMP.extensions]
    ReactiveMPOptimisersExt = "Optimisers"
    ReactiveMPProjectionExt = "ExponentialFamilyProjection"
    ReactiveMPRequiresExt = "Requires"

    [deps.ReactiveMP.weakdeps]
    ExponentialFamilyProjection = "17f509fa-9a96-44ba-99b2-1c5f01f0931b"
    Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "4395a4cad612f95c1d08352f8c53811d6af3060b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Rocket]]
deps = ["DataStructures", "Sockets", "Unrolled"]
git-tree-sha1 = "fe7373bf6b935c4431002fd91fa581d5eb835d09"
uuid = "df971d30-c9d6-4b37-b8ff-e965b2cb3a40"
version = "1.8.3"

[[deps.RxInfer]]
deps = ["BayesBase", "DataStructures", "Dates", "Distributions", "DomainSets", "ExponentialFamily", "FastCholesky", "GraphPPL", "HTTP", "JSON", "LinearAlgebra", "Logging", "MacroTools", "Optim", "Preferences", "ProgressMeter", "Random", "ReactiveMP", "Reexport", "Rocket", "Static", "Statistics", "TupleTools", "UUIDs"]
git-tree-sha1 = "bcaf218d6b5329dc5e5be4299cbb5818c2c7bb19"
uuid = "86711068-29c9-4ff7-b620-ae75d7495b3d"
version = "4.6.2"

    [deps.RxInfer.extensions]
    PrettyTablesExt = "PrettyTables"
    ProjectionExt = "ExponentialFamilyProjection"

    [deps.RxInfer.weakdeps]
    ExponentialFamilyProjection = "17f509fa-9a96-44ba-99b2-1c5f01f0931b"
    PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLPublic]]
git-tree-sha1 = "ed647f161e8b3f2973f24979ec074e8d084f1bee"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

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

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "49440414711eddc7227724ae6e570c7d5559a086"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.3.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"

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
git-tree-sha1 = "2c962245732371acd51700dbb268af311bddd719"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.6"

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

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TinyHugeNumbers]]
git-tree-sha1 = "83c6abf376718345a85c071b249ef6692a8936d4"
uuid = "783c9a47-75a3-44ac-a16b-f1ab7b3acf04"
version = "1.0.3"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

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

[[deps.Unrolled]]
deps = ["MacroTools"]
git-tree-sha1 = "6cc9d682755680e0f0be87c56392b7651efc2c7b"
uuid = "9602ed7d-8fef-5bc8-8597-8f21381861e8"
version = "0.1.5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.5.0+2"
"""

# ╔═╡ Cell order:
# ╟─278382c0-d294-11ef-022f-0d78e9e2d04c
# ╟─9fbae8bf-2132-4a9a-ab0b-ef99e1b954a4
# ╟─27839788-d294-11ef-30a2-8ff6357aa68b
# ╟─2783a99e-d294-11ef-3163-bb455746bf52
# ╟─aed436fd-6773-4932-a5d8-d01cf99c10ec
# ╟─983873d0-e1bc-4e1b-9b6c-3df0a17d83f6
# ╟─2783b312-d294-11ef-2ebb-e5ede7a86583
# ╟─939e74b0-8ceb-4214-bbc0-407c8f0b2f26
# ╟─e3d5786b-49e0-40f7-9056-13e26e09a4cf
# ╟─2f38a3c9-67d4-4c45-ad0b-e757f62d6a5e
# ╟─4ed2ef1f-4d55-4f9b-b8f5-7f77b5fc62ca
# ╟─2783c686-d294-11ef-3942-c75d2b559fb3
# ╟─126c3221-34b8-4a8f-b5b5-c14ff4c6a2a1
# ╟─29592915-cadf-4674-958b-5743a8f73a8b
# ╟─5b8405f7-8878-43dd-8f84-bca17450924f
# ╟─9708215c-72c9-408f-bd10-68ae02e17243
# ╟─f9b241fd-d853-433e-9996-41d8a60ed9e8
# ╟─97136f81-3468-439a-8a22-5aae96725937
# ╟─4e990b76-a2fa-49e6-8392-11f98d769ca8
# ╟─aaa07dc5-9105-4f70-b924-6e51e5c36600
# ╟─aec84a6d-33fc-4541-80c0-091998f8c4d1
# ╟─bed6a9bd-9bf8-4d7b-8ece-08c77fddb6d7
# ╟─ef54a162-d0ba-47ef-af75-88c92276ed66
# ╟─94391132-dee6-4b22-9900-ba394f4ad66b
# ╟─a8c88dff-b10c-4c25-8dbe-8f04ee04cffa
# ╟─63609dc2-2413-4822-8401-2d2ba22adfce
# ╟─5b66f8e5-4f01-4448-82e3-388bc8ea31de
# ╟─65e34124-fda2-4a60-9722-ece2f68a39d9
# ╟─6e14d8e3-9dc2-44fb-9fa5-c48f10cb1076
# ╟─bece6e3d-57b2-4bf4-8bf8-d5a11ad8983f
# ╟─64474167-bf52-456c-9099-def288bd17bf
# ╟─2784f45e-d294-11ef-0439-1903016c1f14
# ╠═0b5b816b-2dd2-4fe8-8f84-4eb2d58b5d59
# ╠═1f468b26-b4fe-4f61-af41-0a15fdc44365
# ╟─8ac328de-b7ba-457f-add6-9af506832282
# ╟─651a4081-68f1-4237-9503-d4e28969a836
# ╟─f4509603-36be-4d24-8933-eb7a705eb933
# ╟─8d7058c4-0e13-4d05-b131-32b1f118129f
# ╟─1c53d48b-6950-4921-bf03-292b5ed8980e
# ╟─e3b4d8cc-d028-4e00-b4f3-369f1601510d
# ╟─d823599e-a87f-4586-999f-fbbd99d0db65
# ╟─df4950bc-4532-44e3-b8c7-18dd62fdfb09
# ╟─c51232ba-9908-476a-949f-d5d0949d7960
# ╟─aed24f2d-105e-4583-b9d6-24d1886002d8
# ╟─acefdbf6-1beb-4ce5-9106-0fc7be63dabe
# ╟─6d697856-cc58-4d6a-afd3-c0c6bfbc0d88
# ╟─8627b90a-79d8-43bf-8d53-1043f6e816ab
# ╟─58c51f5e-69a0-4802-892b-0a52462c4e55
# ╟─14ef2403-cba3-4e76-91c7-4655656a632d
# ╟─54cc2500-f06f-4f39-95f3-8d9d09c37234
# ╟─b900b7d4-3a49-4651-a94b-74fdbbf094d9
# ╟─cd2dafbe-5131-4ee0-8eb7-33bd04f3133f
# ╟─be0dc5c0-6340-4d47-85ae-d70e06df1676
# ╠═97a0384a-0596-4714-a3fc-bf422aed4474
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
