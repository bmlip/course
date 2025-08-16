### A Pluto.jl notebook ###
# v0.19.47

using Markdown
using InteractiveUtils

# ╔═╡ 89207b78-6e1b-11f0-37b9-718db3a990e3
md"""
# Dynamic Models

  * **[1]** (##) Given the Markov property

\begin{equation*} p(x*n|x*{n-1},x*{n-2},\ldots,x*1) = p(x*n|x*{n-1}) \tag{A1} \end{equation*} proof that, for any ``n``, \begin{align*} p(x*n,x*{n-1},&\ldots,x*{k+1},x*{k-1},\ldots,x*1|x*k) = \
&p(x*n,x*{n-1},\ldots,x*{k+1}|x*k) \cdot p(x*{k-1},x*{k-2},\ldots,x*1|x*k) \tag{A2}\,. \end{align*} In other words, proof that, if the Markov property A1 holds, then, given the "present" (``x_k``), the "future" ``(x_n,x_{n-1},\ldots,x_{k+1})`` is *independent* of the "past" ``(x_{k-1},x_{k-2},\ldots,x_1)``.

  * **[2]** (#)      (a) What's the difference between a hidden Markov model and a linear Dynamical system?    

    (b) For the same number of state variables, which of these two models has a larger memory capacity, and why?
  * **[3]** (#)

(a) What is the 1st-order Markov assumption?       (b) Derive the joint probability distribution ``p(x_{1:T},z_{0:T})`` (where ``x_t`` and ``z_t`` are observed and latent variables respectively) for the state-space model with transition and observation models ``p(z_t|z_{t-1})`` and ``p(x_t|z_t)``.       (c) What is a Hidden Markov Model (HMM)?        (d) What is a Linear Dynamical System (LDS)?       (e) What is a Kalman Filter?       (f) How does the Kalman Filter relate to the LDS?        (g) Explain the popularity of Kalman filtering and HMMs?        (h) How relates a HMM to a GMM? 

"""

# ╔═╡ Cell order:
# ╟─89207b78-6e1b-11f0-37b9-718db3a990e3
