### A Pluto.jl notebook ###
# v0.20.28

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 10bdf484-848d-44a1-af9b-64c0c11395f7
begin
	using BmlipTeachingTools
	using LaTeXStrings
	using Random
	using Printf
	using LinearAlgebra
	using SpecialFunctions
	using Distributions
	using RxInfer
	using Plots; default(label="", linewidth=4, margin=10Plots.pt)
end

# ╔═╡ 35de8df9-ab15-4ea7-a099-6f6523635267
using HypertextLiteral

# ╔═╡ a3eccf21-dab7-48ae-8091-e3f4569a2a04
using Memoization

# ╔═╡ 71fa9b4b-ca28-4517-b6d4-0135cce1b800
title("Probabilistic Programming 1: Bayesian inference in conjugate models")

# ╔═╡ 779f3ea6-36dd-11f0-0323-613cd06c6f3c
md"
#### Learning objectives:
  - Specify models in a probabilistic programming language.
  - Specify inference procedures in a probabilistic programming language.
  - Perform message passing as Bayesian infernece on factor graphs.

#### Materials:
  - Mandatory
    - This notebook
    - Lecture notes on factor graphs
    - Lecture notes on continuous data
    - Lecture notes on discrete data
  - Optional
    - Chapters 2 and 3 of [Model-Based Machine Learning](http://www.mbmlbook.com/LearningSkills.html).
    - [Differences between Julia and Matlab / Python](https://docs.julialang.org/en/v1/manual/noteworthy-differences/index.html).
"

# ╔═╡ feef03fb-f9c5-4a76-8727-2d690c4dac1f
TableOfContents()

# ╔═╡ 34a1be54-4641-455d-ab3e-8d4d08aaf966
md"""
In this session, we will play with probability distributions and automated Bayesian inference software to solve a few basic problems.
"""

# ╔═╡ dfa63b82-3593-45b1-b13f-c47274ce724c
NotebookCard("https://bmlip.github.io/course/minis/Distributions%20in%20Julia.html")

# ╔═╡ de99bbcb-ce5b-44f1-85ef-ad26a9238509
challenge_statement(" Problem: A Job Interview"; header_level=1)

# ╔═╡ 4dfba22d-1d61-4325-90a0-52afed1787ee
md"Suppose you have graduated and applied for a job at a tech company. The company wants a talented and skilled employee, but measuring a person's skill is tricky; even a highly-skilled person makes mistakes and - vice versa - people with few skills can get lucky. They decide to approach this objectively and construct a statistical model of responses. 

In this session, we will look at estimating parameters in various distributions under the guise of assessing skills based on different types of interview questions. We will practice message passing on factor graphs using a probabilistic programming language developed at the TU/e: [RxInfer.jl](https://rxinfer.com/).
"

# ╔═╡ f05721c1-e085-4c7d-92c8-292aa17c8040
md"""
## Part 1: _Right or wrong_

To start, the company wants to test the applicants' programming skills and created a set of bug detection questions. We have outcome variables $X_i$. An answer is either right or wrong, which can be modelled with a Bernoulli likelihood function. The company assumes you have a skill level, denoted $\theta \in [0,1]$, and the higher the skill, the more likely you are to get the question right. Since the company doesn't know anything about you, they chose an uninformative prior distribution: the ``\text{Beta}(1,1)``. We can write the generative model for the test results as follows:

```math 
\begin{align}
p(X, \theta \, | \, \alpha, \beta) &= p(\theta \, | \, \alpha, \beta) \prod_{i=1}^{N} p(X_i \mid \theta) \\ &= \text{Beta}(\theta \, | \, \alpha, \beta) \prod_{i=1}^{N} \text{Bernoulli}(X_i \mid \theta) \, , 
\end{align}
```

We are now going to construct this probabilistic model in RxInfer.
"""

# ╔═╡ 35d8e213-09ef-4419-a5b8-ac9a1b4e3c1f
@model function beta_bernoulli(X,N,α,β)
    "Beta-Bernoulli model with multiple observations"
    
    # Prior distribution
    θ ~ Beta(α, β)
        
    # Loop over data
    for i in 1:N
        
        # Likelihood of i-th data points
        X[i] ~ Bernoulli(θ)
        
    end
end

# ╔═╡ 5cf8f6fe-31cc-4226-89dd-6e8fe6527442
md"""
!!! info "The ~ operator"
	We can define random variables using a tilde symbol `~`, which should be read as:
	
	> _[random variable]_ is distributed according to _[probability distribution function]_.
	
	For example, $\theta \sim \text{Beta}(1,1)$ should be read as 
	
	>  $\theta$ is distributed according to a Beta($\theta$ | $a$=1, $b$=1) probability distribution.
"""

# ╔═╡ 6dae4878-8765-4bae-b8df-399d43cdf53c
md"""
### Explore the prior

In this model, the parameter ``\theta`` (the probability of a correct answer) has a prior: the ``\text{Beta}`` distribution. Below, you can play with the parameters of this distribution.
"""

# ╔═╡ 0cb98ad6-12b5-4644-810a-eaf0a8436084
@htl """
prior <strong>shape</strong> parameter α = &nbsp; $(@bind α_explore Slider(0.02:0.02:20, show_value=true, default=1)) ,  <br>
prior <strong>rate</strong> parameter β = &nbsp; &nbsp; &nbsp; $(@bind β_explore Slider(0.02:0.02:20, show_value=true, default=1))
"""

# ╔═╡ ffc0d2c1-cbb6-496b-91c1-39dc505a8df1
plot(
	range(.001, step=0.01, stop=0.999), 
	x -> pdf(Beta(α_explore,β_explore), x);
	color="red", label="", xlabel="θ", ylabel="p(θ)", size=(600,300),
	ylim=(-.1,4)
)

# ╔═╡ c6d8d802-ee4b-4eac-84cb-a1769d2ab175
Foldable("Exercise",
			md"""
			What would a prior distribution look like where you believe that applicants will get many questions right? Provide your answer by changing α and β.
			"""
		)

# ╔═╡ 928f748d-60eb-45bb-bc8d-0a99b3f11b01
md"""
Show exercise feedback: $(@bind show_solution1 CheckBox())
"""

# ╔═╡ 0735f44e-2b14-41f8-a7fd-ab18b20a6d87

let
	yes = cdf(Beta(α_explore,β_explore), 0.5) < 0.3

	if show_solution1
		if yes
			correct()
		else
			hint("For which values of α,β is the probability for θ > 0.5 high?")
		end
	end
end

# ╔═╡ 90210954-a4b0-4e98-b00e-667479bfe60d
md"""
🙋 Want to play more with making prior distributions? Check this out:
"""

# ╔═╡ 5163221b-9572-4cfe-950a-ac435ba57c19
NotebookCard("https://bmlip.github.io/course/minis/prior%20playground.html")

# ╔═╡ 2cd6a20d-80bb-4aaf-8017-c657ed1315bc
md"""
### Explore the posterior
Let's come up with some score outcomes. Click on the checkboxes to mark answers as **correct** or **incorrect**.
"""

# ╔═╡ 388981ac-ee33-46b1-8162-1f51e479b5d7
begin
Xspec = @htl """
		<div style="font-family: monospace; font-weight: bold">
		X = [
			$(@bind X1 CheckBox(true)) ,
			$(@bind X2 CheckBox(true)) ,
			$(@bind X3 CheckBox(false)) ,
			$(@bind X4 CheckBox(false)) ,
			$(@bind X5 CheckBox(false)) ,
			$(@bind X6 CheckBox(false)) ,
			$(@bind X7 CheckBox(false)) ,
			$(@bind X8 CheckBox(false))
		]
	"""
end

# ╔═╡ 613e50b4-535e-4420-8ecd-c1ddc89c262e
X = Int[X1, X2, X3, X4, X5, X6, X7, X8]

# ╔═╡ bc347329-d4f1-4999-89ce-fd50e4d890e1
md"""Having defined the model, we can now call an inference procedure which will automatically compute the posterior distribution for the random variable:"""

# ╔═╡ 307acd4b-72a7-4aa5-add8-8269708adaa9
md"""
Under the hood, RxInfer is performing message passing. Each variable definition actually creates a factor node and each node will send a message. The collision of messages will automatically update the marginal distributions. 

#### Results!
Now we have a posterior for ``\theta``! It is shown in the graph below, together with the prior. Play with the prior parameters, and the data, to see how the posterior updates.
"""

# ╔═╡ b34f2fce-3944-4c9c-bd8b-f48aca983979
Xspec

# ╔═╡ 6671664b-1efc-437e-8690-bd8d4c9a7277
begin
	priorparamspecBB = @htl """
		prior shape parameter α = &nbsp; $(@bind α Slider(0.01:0.01:20, show_value=true)) ,  <br>
		prior rate parameter β = &nbsp; &nbsp; &nbsp; $(@bind β Slider(0.01:0.01:20, show_value=true))
	"""
end

# ╔═╡ 5411235f-e823-4db4-b4c4-1532d4bc8927
resultsBB = infer(
	model = beta_bernoulli(N=length(X), α=α, β=β),
	data  = (X = X,),
)

# ╔═╡ c72aabb2-779f-421c-b828-277c3b4d1233
begin
	# Range of values to plot pdf for
	θ_range = range(.001, step=0.01, stop=0.999)
	
	# Prior
	plot( θ_range, x -> pdf(Beta(α,β), x), color="red", label="Prior", xlabel="θ", ylabel="p(θ)")
	
	# Posterior
	plot!(θ_range, x -> pdf(resultsBB.posteriors[:θ], x), color="blue", linestyle=:dash, label="Posterior", size=(800,300))
end

# ╔═╡ 4fd197db-cac8-4a39-a9eb-9399f945a4e1
Foldable("Exercise",
			md"""
				Can you make the prior and posterior distribution look similar?
			"""
		)

# ╔═╡ a8da905e-16df-4fc0-a4ef-ff10cbd60a85
md"""
Show feedback: $(@bind show_solution2 CheckBox())
"""

# ╔═╡ aa1528c5-1c46-4c45-a8e8-8a899e8124c8
md"""
## Part 2: _Scoring questions 0-1-2_

Suppose you are not tested on a right-or-wrong question, but on a score question. For instance, you have to complete a piece of code for which you get a score. If all of it was wrong you get a score of $0$, if some of it was correct you get a score of $1$ and if all of it was correct you get a score $2$. That means we have a likelihood with three outcomes: $X_1 = \{ 0,1,2\}$. 

Suppose we once again ask two questions, $X_1$ and $X_2$. The order in which we ask these questions does not matter, so that means we choose Categorical distributions for these likelihood functions: $X_1, X_2 \sim \text{Categorical}(\theta)$. 

The parameter $\theta$ is no longer a single number, indicating the probability of a correct answer, but a vector of three numbers: $\theta = (\theta_1, \theta_2, \theta_3)$. Each $\theta_k$ indicates the probability of getting the $k$-th outcome. In other words, $\theta_1$ indicates the probability of getting $0$ points, $\theta_2$ of getting $1$ point and $\theta_3$ of getting $2$ points. A highly-skilled applicant mights have a parameter vector of $(0.05, 0.1, 0.85)$, for example.
"""

# ╔═╡ 4a0d19ef-1756-4d14-a787-0b31763577f5
md"""
### Dirichlet prior
The prior distribution conjugate to the Categorical distribution is the **Dirichlet** distribution. 

Visualizing a Dirichlet distribution is a bit tricky. In the special case of $3$ parameters, we can plot the probabilities on a [simplex](https://en.wikipedia.org/wiki/Simplex). As a reminder, a simplex in 3-dimensions is the triangle between the coordinates $[0,0,1]$, $[0,1,0]$ and $[1,0,0]$:

![Visualisation of the simplex in 3D space](https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/2D-simplex.svg/150px-2D-simplex.svg.png)

Every point on that triangle is 3D vector that sums to 1. Since the triangle is a 2-dimensional subspace, we can map the 3D simplex to a 2D triangular surface and plot the Dirichlet's probability density over it.
"""

# ╔═╡ 0a536620-fc4a-41c6-8f34-7015b400c910
md"""
Let's look at the generative model:

```math
p(X_1, X_2, \theta) = p(X_1 \mid \theta) p(X_2 \mid \theta) p(\theta) \, .
```

It's the same as before. The only difference is the parameterization of the distributions:

```math
\begin{align} p(X_1 \mid \theta) =&\ \text{Categorical}(X_1 \mid \theta) \\ p(X_2 \mid \theta) =&\ \text{Categorical}(X_2 \mid \theta) \\ p(\theta) =&\ \text{Dirichlet}(\theta \mid \alpha) \, , \end{align}
```

where $\alpha$ are the concentration parameters of the Dirichlet. This model can be written directly in RxInfer:
"""

# ╔═╡ 4973ed66-7e7f-402b-b636-0a7f8771e743
@model function dirichlet_categorical(Y, N, α)
    
    # Prior distribution
    θ ~ Dirichlet(α)
    
    # Likelihood
    for i in 1:N

        Y[i] ~ Categorical(θ)
        
    end
end

# ╔═╡ ff49a7f8-7f83-4fe0-a470-0f2c265fc619
md"""
Suppose you got a score of $1$ on the first question, a score of $2$ on the second question and a score of $2$ on the third question. In a **one-hot encoding**, this is represented as:

```julia
Y = [[0, 1, 0],
	 [0, 0, 1],
	 [0, 0, 1]]
```
"""

# ╔═╡ 617339f1-a22a-40ff-aa17-6258cdb8ef6c
md"""
You can fill in the answer scores here:
"""

# ╔═╡ e73893ad-d6d4-496d-ae39-20650ee175ab
begin
	score_text_bond = @bindname score_text TextField(default="1, 2, 2")
end

# ╔═╡ 861a82c4-87ea-4a01-b511-9847786f347f
Y = let
	scores = parse.(Int, collect(replace(score_text, r"[^\d]+" => "")))
	filter!(in(0:2), scores)
	scores = isempty(scores) ? [1] : scores
	map(scores) do i
		xs = zeros(Float64, 3)
		xs[i + 1] = 1
		xs
	end
end

# ╔═╡ e2aeb352-6a50-45a0-89f9-c68e287245ca
md"""
### Prior distribution
You can set the prior concentration parameters yourself:

"""

# ╔═╡ 8442878d-068c-4cee-b51a-782190703f58
md"""
The yellow spot is the area of high probability, with the color gradient to dark blue indicating decreasing probability. 
"""

# ╔═╡ 41e0eb82-bf76-49e0-8742-fcb30e675905
md"""
### Inference
Now we infer the posterior distribution:
"""

# ╔═╡ 68a37b6f-6bad-4752-85c4-ad929fc59e16
score_text_bond

# ╔═╡ cea0cb2a-8e11-443f-8e2f-c49b09d9fdf5
@bindname var"show prior" CheckBox(default=false)

# ╔═╡ ab3fb0b2-242f-46eb-93e9-6f70f18df707
Foldable("Exercise",
md"""
Which corner of the triangle corresponds to score ``0``?

!!! hint
	Try changing the score, and see how to posterior updates.
"""
)

# ╔═╡ 13a5594d-b675-43c5-ad7c-1b9e0e0a9572
Foldable("Exercise",
md"""
Can you find an **uninformative** prior, i.e. a prior where the likelihood equals the posterior?
"""
)

# ╔═╡ ac1321cb-764a-48d3-8ac4-9cedfe34d370
md"""
## Part 3: _Continuous-valued score_

Suppose the company wants to know how fast applicants respond to questions. The interviewer has a stopwatch to measure the **response time** per question.
"""

# ╔═╡ 9c1da54d-edad-499a-91ae-88483b5d5a72
Z = [ 52.3
      74.4
      50.9
      39.5 ];  # seconds

# ╔═╡ 174aedea-93e2-4e21-aac1-21a4c27cede9
md"""
Each applicant is assumed to have some underlying response speed ``\theta``. Each measurement ``X_i`` is a noisy observation of that response speed, where the noise is assumed to be symmetric, i.e., the applicant might a bit as faster as often as they are a bit slower than usual. The Gaussian, or Normal, distribution is a symmetric continuous-valued distribution and will characterize the assumption well. The likelihood is therefore:

```math
p(X \mid \theta) = \mathcal{N}(X \mid \theta, \sigma^2) \, ,
```

where ``\sigma`` is the standard deviation. Think of this as the accuracy with which the interviewer measures your score; sometimes the interviews stops the watch immediately after your answer and sometimes they are distracted and react late. We've given you a slider below with which you can try out different values.
"""

# ╔═╡ abe4105c-c42b-4d97-bc15-ef9741c23fcf
begin
	likvarspec = @htl """
	likelihood variance σ2 = &nbsp; $(@bind σ2 Slider(0.01:0.01:2, default=0.5, show_value=true))
	"""
end

# ╔═╡ 73025b50-feac-4723-8a65-024689232112
md"""
To see the effect of σ², here you see a simulation of 10 measurements, assuming ``\theta = 70``:
"""

# ╔═╡ 0af6ca08-cece-460b-9166-5b8168fb9b73
let
	θ = 70 # seconds
	rand(Normal(θ, sqrt(σ2)), 10)
end

# ╔═╡ caf8ede0-a410-4b4a-81dd-4342b8a2385a
md"""
### Gaussian Prior
The conjugate prior to the mean in a Gaussian likelihood is another Gaussian distribution: 

```math
p(\theta) = \mathcal{N}(\theta \mid m_0, v_0)
```

with $m_0, v_0$ as prior mean and variance. 

The company designed the questions such that they think it may take the average participant 60 seconds to respond, $\pm$ 20 seconds. That translates to the following values for the prior parameters:
"""

# ╔═╡ 75abee86-a43c-4a53-8edb-cf943f4e570c
begin
	priorparamspecNN = @htl """
		prior mean m₀ = &nbsp; &nbsp;  $(@bind m0 Slider(40:1.0:80, default=60, show_value=true))   <br>
		prior variance v₀ =   $(@bind v0 Slider(round.((0.1:.1:30).^2; sigdigits=2), default=20^2, show_value=true))
	"""
end

# ╔═╡ 71b7e9cb-a6ce-4982-b00c-b98dcd08888f
md"""
But feel free to change these to what you think is reasonable.
"""

# ╔═╡ 7cc095a9-fbf3-4402-bba6-f25fdb1347b1
exercise_statement("Real scores"; prefix="Model ", header_level=3)

# ╔═╡ 27991fcd-6754-42fe-8e47-d52187065afb
md"""
Can you specify a model in RxInfer code for the continuous-valued score case, described above? 
"""

# ╔═╡ 86f213ce-b2b2-48ce-bd94-ccd863e3e2d7
hint(md"""
- Hint: for a Gaussian distribution, use `NormalMeanVariance(μ,σ²)`.
- Hint: the parameters are called `m0`, `v0`, `σ2`.	 
""")

# ╔═╡ 12d5f6d1-f214-448e-9c2e-da691b997d60
### YOUR CODE HERE

# ╔═╡ 67b0f499-9492-48fb-9540-b6600abffc3a
hide_solution(md"""
```julia
@model function normal_normal(Z,m0,v0,σ2,N)
    
	# Prior distribution
    θ ~ NormalMeanVariance(m0,v0)
    
    # Likelihood
    for i in 1:N
        Z[i] ~ NormalMeanVariance(θ,σ2)
    end 
end
```
""")

# ╔═╡ 017e4cec-544f-46aa-b612-1b1ba83bb792
exercise_statement("Reaction time", prefix="Inference ", header_level=3)

# ╔═╡ 66b2711a-f4b9-43fe-8f0f-112d1365bbbe
md"""
We will also need a `infer` procedure. Can you define and execute one below? Please use the format:

```julia
results = infer(
	...
)
```
"""

# ╔═╡ 0bb66b2a-5d41-4f6b-b245-0360274b9296
### YOUR CODE HERE

# ╔═╡ df362c96-d688-42c1-acfc-c270c33d5db7
hide_solution(md"""
```julia			  
results = infer(
	model = normal_normal(m0=m0,v0=v0,σ2=σ2,N=length(Z)),
	data = (Z = Z,),			  
)
```
""")

# ╔═╡ 5fa37c24-c90e-4e21-921e-03422943f225
md"""
To visualize the prior and posterior distribution, uncomment the line below.
"""

# ╔═╡ 6d7f9ad2-8dc4-450d-afb6-c55deaeed564
# visualize_results(results)

# ╔═╡ 5c315bdb-7375-459e-a9e2-0d180271ed78
likvarspec

# ╔═╡ ff70f05f-7259-4d56-9a33-19ebd3483c63
priorparamspecNN

# ╔═╡ 7e6ac0f6-710e-4388-acc2-b2b255a961ca
Foldable("Bonus Exercise",
			md"""
			 Your posterior is probably narrower than your prior distribution. Thought exercise: can you make the posterior be wider than the prior distribution?
			"""
		)

# ╔═╡ 02156dca-2fcb-4654-80a0-3666a7807ea3
hide_solution(md"""
This is not possible, because every observation will strictly decrease the variance of the distribution. This can be seen in the formula for the posterior variance of the Gaussian distribution:
			  
$\begin{aligned}
  	v_{1} = \frac{1}{1/v_0 + 1/σ^2}
\end{aligned}$
			  
$(NotebookCard("https://bmlip.github.io/course/lectures/The%20Gaussian%20Distribution.html#Inference"))
""")

# ╔═╡ 5ef49dcd-16ff-4183-a28b-061572d98b98
md"""
## Appendix
"""

# ╔═╡ d8218c14-e005-4e34-a468-44ebb790ef11
function visualize_results(results)

	# Range of values to plot pdf for
	μ_range = range(50.0, step=0.05, stop=65.0)
	
	# Prior
	plot( μ_range, x -> pdf(Normal(m0, sqrt(v0)), x), color="red", label="Prior", xlabel="θ", ylabel="p(θ)")
	
	# Posterior
	plot!(μ_range, x -> pdf(results.posteriors[:θ], x), color="blue", linestyle=:dash, label="Posterior", size=(800,300))
	
end

# ╔═╡ 7bb464b2-0cd2-484e-aacd-f01ff92fc30d
const learn_more_kl = NotebookCard("https://bmlip.github.io/course/minis/KL%20Divergence.html");

# ╔═╡ 9d9542b3-cadf-4ada-95a9-d1873061df8d
if show_solution2
	let div = kldivergence(Beta(α,β), resultsBB.posteriors[:θ])
		if div < 0.1
			correct()
		else
			str = @sprintf("%.2f", div)
			@htl("""
				 <p>That's not quite it. Looking at the KL divergence, the prior and posterior are too dissimilar.</p>
			<div style="display: grid; place-items: center">
			<div style=" display: inline-block; width: max-content; padding: 2em; border: 4px solid #88888833; border-radius: 1em; font-family: system-ui; background: var(--white); font-variant-numeric: tabular-nums;">
			<strong>Kullback-Leibler Divergence: $(str)</strong>
			</div>
			</div>
				<p>You can learn more about <em>KL divergence</em> here:</p>
				$(learn_more_kl)
			""") |> keep_working
		end
	end
end

# ╔═╡ 948a442d-64c3-4a8b-ad9f-b98f5db82438
header(text, x) = md"""
	#### $(text)
	$x
	"""

# ╔═╡ f1395ab5-637c-4894-999c-4f45489cfccb
dirichlet_slider() = Slider([(0.1:.1:5)..., (5:5:100)...]; show_value=true, default=3)

# ╔═╡ ec2fb935-0e29-4188-b507-b7009452a99e
begin
	α₀_bond = 
	PlutoUI.ExperimentalLayout.vbox([
		@bindname(α₀_1, dirichlet_slider())
		@bindname(α₀_2, dirichlet_slider())
		@bindname(α₀_3, dirichlet_slider())
	])
end

# ╔═╡ 4df18a8f-7483-45db-9988-8579dc4b9103
α₀ = [α₀_1, α₀_2, α₀_3]

# ╔═╡ 2f0932a8-7f5e-41f8-aea6-882a147aadf3
resultsDC = infer(
    model = dirichlet_categorical(α=α₀, N=length(Y)),
    data = (Y = Y,),
)

# ╔═╡ c600e2cc-1094-4177-a533-26cba3ff4eaa
md"""
#### Prior parameters
$α₀_bond
"""

# ╔═╡ 02e2ddee-afb9-43a5-98b1-420e1e6169d8
md"""
### Triangle mesh plot

Fons wrote a little JS triangle mesh plotting function to make this notebook load faster. It uses [regl](https://github.com/regl-project/regl) (WebGL) and [AbstractPlutoDingetjes.jl](https://plutojl.org/en/docs/abstractplutodingetjes/#published_to_js).

The triangle mesh is generated with [Triangulate.jl](https://github.com/JuliaGeometry/Triangulate.jl).
"""

# ╔═╡ c2dfe525-508e-4af7-9224-7f56051c651d
import AbstractPlutoDingetjes

# ╔═╡ 7db33856-3f31-481b-9622-d673ae16c5e7
# This function is memoized, so that AbstractPlutoDingethes.Display.published_to_js can cache the result. This way, Pluto only needs to send changes to the vertex values.
@memoize function triangle_data_to_js(points::Matrix, triangles::Matrix)
	collect.(eachrow(points)),
	collect.(eachcol(triangles .- 1))
end

# ╔═╡ acf9c5bd-e9a6-44c2-9baf-0d1a0b41c708
function mesh_heatmap_superfast(points::Matrix, triangles::Matrix, vals::Vector{<:Real}; clim=(0.0,maximum(vals)))

	to_share = triangle_data_to_js(points, triangles)
	
@htl("""
<div>
<script id="asdf">
const {default: regl_lib} = await import("https://esm.sh/regl@2")
const dpr = window.devicePixelRatio ?? 1;

const canvas = this ?? document.createElement("canvas")
	canvas.width = 350 * dpr
	canvas.height = 350 * dpr
	canvas.style.width = "350px"
	canvas.style.height = "350px"
const regl = regl_lib({
	canvas: canvas,
	attributes: {
		preserveDrawingBuffer: true
	}
})

	   
const drawHeatmap = regl({
	frag: `
		precision mediump float;
		varying float v_value;
		uniform float u_minVal;
		uniform float u_maxVal;

		vec3 viridis(float t) {
			const vec3 c0 = vec3(0.2777, 0.0056, 0.3316);
			const vec3 c1 = vec3(0.1050, 0.5150, 0.6180);
			const vec3 c2 = vec3(0.3410, 0.7773, 0.4102);
			const vec3 c3 = vec3(0.9940, 0.9065, 0.1440);
			t = clamp(t, 0.0, 1.0);
			if (t < 0.33333) {
				return mix(c0, c1, t * 3.0);
			} else if (t < 0.66666) {
				return mix(c1, c2, (t - 0.33333) * 3.0);
			} else {
				return mix(c2, c3, (t - 0.66666) * 3.0);
			}
		}
		
		void main() {
			float normalized = (v_value - u_minVal) / (u_maxVal - u_minVal);
			vec3 color = viridis(normalized);
			gl_FragColor = vec4(color, 1.0);
		}
	`,

	vert: `
		precision mediump float;
		attribute vec2 position;
		attribute float value;
		varying float v_value;
		uniform vec2 resolution;
		
		void main() {
			vec2 pos = position*2.0 - 1.0;
			// pos.y *= 1.0;
			gl_Position = vec4(pos, 0, 1);
			v_value = value;
		}
	`,

	attributes: {
		position: regl.prop('positions'),
		value: regl.prop('values')
	},

	elements: regl.prop('elements'),

	uniforms: {
		resolution: [canvas.width, canvas.height],
		u_minVal: regl.prop('minVal'),
		u_maxVal: regl.prop('maxVal')
	}
});

function plotHeatmap(pts, triangleList, val) {
	const [minVal, maxVal] = $(clim)

	// Prepare data for regl
	// let positions = pts.map(x => Array.from(x));
	const positions = new Array(pts[0].length).fill(0).map((x,i) => [
		pts[0][i],
		pts[1][i]
		])

	
	const values = Array.from(val);
	const elements = triangleList;

	

	// Clear and draw
	regl.clear({
		color: [0.9, 0.9, 0.9, 1.0],
		depth: 1
	});

	drawHeatmap({
		positions: positions,
		values: values,
		elements: elements,
		minVal: minVal,
		maxVal: maxVal
	});
}


plotHeatmap(
	$(identity(to_share[1])), 
	$(identity(to_share[2])), 
	$(identity(vals))
 )

	return canvas
</script>
	</div>
""")
end

# ╔═╡ 6f170031-948a-4a02-8912-913e531ac1a1
md"""
### Triangle mesh generation

The code below generates a triangular mesh for a given maximum triangle area.

We ran this code in advance, and the output is stored directly in the notebook file as Julia code. This makes the notebook load faster.
"""

# ╔═╡ 57aa818c-eb7d-44c4-a94c-a61b59a98192
tris = Int32[362 342 734 46 114 35 43 31 21 18 19 18 18 16 25 14 19 109 20 18 17 25 12 19 11 11 11 42 26 42 33 31 35 67 25 51 46 43 12 34 32 39 62 56 36 39 26 36 44 24 39 40 51 37 40 43 110 37 37 24 31 38 10 54 45 41 56 97 63 63 41 58 12 40 30 45 53 149 53 53 54 54 55 78 36 55 41 58 49 10 73 34 116 29 61 150 95 110 59 65 88 64 119 90 46 52 68 72 74 80 194 129 94 76 114 76 73 78 153 103 75 84 73 77 79 52 77 79 67 84 91 193 139 162 85 155 84 82 74 87 93 86 91 171 88 88 90 89 145 102 87 146 101 86 91 85 102 60 73 30 60 101 113 98 78 65 151 101 100 98 102 93 101 101 119 143 286 124 269 143 123 145 120 63 17 110 113 111 60 118 126 75 111 112 110 113 115 48 49 120 124 119 121 120 134 119 125 128 118 118 123 123 128 115 111 128 124 130 130 148 131 269 271 122 122 128 131 134 132 164 159 155 237 86 325 139 138 104 147 264 141 144 146 138 141 144 107 145 107 91 143 147 130 121 107 100 151 79 98 7 151 160 158 156 156 155 160 155 210 159 167 156 164 210 159 164 153 162 155 163 151 162 159 166 183 81 85 155 175 172 159 167 395 278 178 184 172 191 81 177 179 330 180 175 198 177 209 178 183 177 198 179 179 173 180 184 174 190 173 181 182 184 190 137 230 187 182 187 173 187 192 342 187 204 190 205 170 190 216 274 171 137 196 191 300 191 202 195 196 204 213 178 239 199 215 226 311 203 192 197 291 189 215 194 324 135 179 199 231 219 219 210 199 159 136 229 213 206 233 198 212 226 230 216 217 188 217 204 224 329 278 217 386 224 292 208 352 224 328 228 301 200 305 215 331 228 227 226 227 246 233 211 135 249 212 229 247 187 173 230 246 229 236 241 185 237 234 239 229 242 249 437 226 245 238 236 249 250 245 225 245 244 231 251 487 231 232 243 250 245 232 214 214 214 501 501 498 255 249 256 256 255 258 239 229 484 240 236 259 323 271 274 267 284 268 266 279 141 266 129 268 267 141 267 264 269 271 262 273 129 129 271 272 281 262 203 285 301 758 201 290 304 280 291 279 278 192 312 281 299 104 267 321 265 308 284 286 341 270 286 313 341 300 275 298 290 280 292 303 276 222 738 306 729 293 295 296 279 302 305 731 298 311 275 289 289 302 303 305 302 327 306 304 303 358 345 292 277 290 299 730 313 310 192 289 309 311 308 315 308 282 314 285 284 317 315 951 313 265 275 299 308 337 270 733 287 337 321 324 288 317 326 137 205 342 137 324 349 328 306 327 330 330 241 330 334 333 485 374 334 333 334 240 335 329 356 293 337 337 973 322 283 321 337 322 197 340 300 358 346 363 338 353 375 723 352 738 345 351 305 349 355 349 354 346 327 297 356 377 355 723 350 356 365 871 372 358 365 358 335 363 448 372 487 361 543 381 365 363 359 374 371 363 386 590 370 541 372 384 372 370 365 361 372 370 382 387 371 473 382 384 383 375 362 874 381 372 412 462 404 386 700 385 423 377 369 383 541 388 396 386 528 492 387 451 388 463 394 411 382 441 393 392 391 397 397 392 394 402 400 422 403 398 400 398 409 406 432 424 630 368 514 412 403 393 403 412 380 665 395 415 511 407 409 412 435 413 406 390 559 427 410 415 431 411 522 407 410 419 419 522 667 526 419 431 416 534 537 505 511 424 525 404 424 669 425 430 517 428 619 380 427 512 512 431 430 432 426 434 418 433 439 409 434 440 433 390 449 438 379 433 439 443 444 452 463 440 399 461 441 453 379 392 445 467 471 497 483 451 444 462 455 455 387 374 452 460 454 448 455 464 446 504 454 469 495 432 461 456 443 398 459 433 379 392 464 450 481 451 446 446 452 360 452 372 452 469 478 459 473 332 447 454 456 470 456 476 259 476 496 476 474 447 441 507 466 447 481 493 464 446 441 474 483 485 486 488 490 487 497 332 500 253 502 477 488 491 480 476 477 447 494 456 494 492 474 479 456 502 496 333 240 254 252 488 499 477 487 475 502 496 488 255 252 513 503 523 432 418 478 507 419 513 540 512 510 590 427 505 534 411 512 515 514 517 220 512 513 506 220 536 220 520 515 520 512 429 380 523 508 523 531 526 417 526 389 525 401 652 524 530 524 541 531 543 660 669 539 535 600 576 535 553 510 532 520 518 558 532 587 532 539 546 539 691 375 384 545 542 545 544 541 368 541 553 546 555 531 683 568 553 540 551 556 550 559 580 552 563 550 556 557 580 558 559 551 421 563 547 559 766 592 563 585 566 567 568 568 567 571 572 574 568 569 765 568 763 552 574 567 573 587 573 605 576 591 533 590 599 578 553 591 547 601 576 580 562 581 588 581 586 562 592 583 580 588 585 597 590 767 577 576 606 578 594 420 594 561 420 420 592 587 603 588 607 601 600 575 582 601 599 608 366 600 597 610 608 702 606 601 606 614 405 598 605 600 602 615 602 607 622 614 701 613 609 701 639 616 637 606 628 621 578 618 628 635 619 620 701 625 627 613 629 624 626 624 629 625 654 635 613 632 629 649 401 401 637 655 629 624 630 643 666 673 636 637 638 620 637 638 613 626 612 646 640 646 661 643 641 663 642 644 647 707 678 644 642 642 632 704 677 662 645 367 652 630 665 704 630 632 640 677 652 527 665 417 672 668 425 705 367 662 685 651 664 663 706 663 666 665 679 666 660 405 668 529 531 660 686 711 679 721 668 658 673 631 658 405 711 677 671 527 678 706 686 685 634 671 681 679 663 657 706 709 708 691 675 690 678 715 670 1102 684 871 689 718 799 678 691 692 693 694 541 689 542 726 712 548 696 697 800 772 870 695 700 700 384 366 622 638 702 680 616 653 631 674 727 707 707 657 679 681 688 670 712 771 713 714 671 772 706 685 709 714 6 670 791 904 668 692 799 931 756 742 721 712 353 756 740 696 732 6 726 771 711 727 670 931 756 950 753 954 732 954 734 957 730 732 294 733 347 753 738 6 735 723 738 723 742 743 744 725 895 752 739 898 743 741 742 742 743 806 780 725 796 751 802 749 748 728 819 688 756 753 735 744 729 722 750 793 756 749 757 752 735 294 733 734 753 758 750 806 755 750 766 764 766 566 766 766 702 768 637 578 792 786 720 796 770 712 771 773 777 773 774 779 696 772 720 771 777 773 748 793 780 779 781 905 719 800 869 799 868 786 770 798 788 786 786 785 787 788 788 788 784 786 784 798 803 807 812 852 784 792 769 795 793 797 791 794 794 803 790 905 788 800 801 800 786 801 805 797 805 806 809 881 747 807 837 818 761 832 770 720 842 811 818 850 803 832 813 817 809 814 762 809 755 722 842 815 810 832 819 816 846 816 816 827 833 823 888 822 722 850 822 826 821 845 849 827 722 823 832 828 818 834 821 833 834 860 835 836 821 837 858 909 1137 1108 848 912 848 858 817 809 851 851 919 891 845 846 834 845 857 837 848 845 851 829 852 907 844 879 852 847 854 1181 853 861 854 867 909 859 841 860 839 859 1168 1210 867 1112 857 1169 863 865 874 862 860 782 783 872 870 336 336 723 740 723 801 869 700 336 875 770 880 876 919 876 879 880 843 877 891 881 881 810 4 941 888 934 911 889 899 899 844 891 906 918 893 1205 1205 927 886 891 894 880 893 888 893 893 927 894 881 896 897 896 897 792 885 904 790 897 894 900 894 903 883 901 779 901 903 798 778 903 909 910 845 914 858 908 840 908 922 911 911 892 906 936 915 884 916 914 913 839 908 824 912 1049 936 923 887 926 1144 1110 882 1037 910 923 917 1047 915 921 919 1047 926 886 1044 729 1000 901 974 741 997 741 1206 930 901 932 995 724 741 939 936 1135 918 936 913 936 1161 1114 945 966 1153 938 941 948 1003 1081 933 964 994 955 939 953 934 941 930 997 975 311 960 950 308 965 954 950 733 953 953 953 730 950 999 955 960 969 952 958 957 963 969 964 963 964 951 958 965 962 948 960 978 956 949 956 969 968 1007 973 282 962 964 969 981 960 956 949 991 4 973 338 997 984 963 976 981 1050 956 977 976 977 320 979 287 943 976 964 977 1050 988 942 984 986 973 967 979 984 985 988 1007 1041 1015 1064 1064 1007 991 995 994 998 947 996 945 1004 999 1002 929 955 996 1010 1021 1022 1022 929 1000 1004 1012 995 998 1000 1001 1028 1016 972 993 1009 992 1015 1051 1034 1036 1022 1030 1013 1011 1003 1013 942 1014 1015 1015 1076 1055 1018 1024 1019 927 894 1019 1021 1032 981 1011 1001 1026 1024 1024 1027 1000 1027 1006 1010 1026 1018 1038 1066 1030 1039 1059 1023 1026 1028 1036 1066 1032 884 1046 927 1027 1059 1025 1031 942 1038 1057 1033 882 1074 1041 1078 1070 1036 1042 1029 1061 1046 1045 1038 1048 921 1049 927 1047 925 921 1044 921 924 1049 984 1041 1052 1015 1063 1064 992 1053 1016 1054 1057 1057 882 1057 1086 1060 1098 1059 1077 1074 1041 1040 1063 1062 1040 1040 1052 992 1066 1060 1057 1066 1068 1088 1070 1053 1071 1063 1070 1069 1077 1104 1080 1067 1078 1075 1082 1069 1006 1040 1079 1100 1074 1079 1092 1077 1080 1072 1071 1081 1080 1082 1078 1083 1085 1083 1085 1088 1101 1141 1080 1085 1087 1084 1089 1095 1045 1097 1097 1091 1134 1169 1150 1094 1095 1108 1105 1110 1098 1094 1059 1098 1130 1092 1079 1101 1091 1096 1115 1102 1097 1095 1130 1100 1096 1179 1102 1102 1090 1095 1115 1123 1115 1110 1129 1117 1118 1125 1126 1116 1129 1110 1110 1116 1121 1121 1127 1113 1118 1120 1121 1119 1129 1093 1119 1149 1119 1126 1136 1148 1126 1128 1116 1123 1138 1158 1138 1128 1115 1104 1130 1091 915 1122 1133 1132 1121 1133 1131 1137 1135 1135 1122 1135 1151 938 1156 1127 1145 1139 1124 1146 1139 1140 1153 1153 938 1147 863 1170 1107 1140 1142 1146 1145 1142 1150 1155 1133 1135 1149 913 938 1184 1166 1143 940 1209 1162 1155 1127 1156 1148 1141 1196 1163 1158 1162 1157 1160 1147 1162 1177 1160 1182 1163 1182 1168 1166 1165 1124 1209 1164 1154 1167 1147 1169 864 1209 856 1157 1202 1173 1200 1208 1183 1174 1177 1196 1176 1194 1160 1168 1163 1183 1183 1152 1181 1183 856 1175 1184 1191 1180 1184 3 1193 855 1197 1194 1195 1172 1191 1171 1181 1182 1195 1196 1195 1172 3 1187 1189 1191 1193 1182 1172 1198 1179 1203 1177 1195 1181 1180 890 926 1207 931 1207 929 863 1174 1154 863 861 863; 377 300 733 68 75 20 38 17 31 14 11 16 15 1 20 18 18 18 25 19 31 23 27 22 21 25 23 50 28 54 55 21 38 68 28 46 55 21 31 26 64 33 52 36 35 38 34 34 39 50 35 38 45 38 37 40 47 39 44 42 43 43 54 37 33 44 33 52 41 49 63 117 50 50 94 51 64 150 34 52 42 57 33 52 56 56 57 57 110 58 30 9 113 98 117 149 41 63 64 64 90 65 120 93 69 68 67 73 102 91 190 131 73 73 94 69 72 79 162 101 72 76 76 97 78 78 150 66 79 79 88 171 138 153 84 83 83 166 76 85 74 87 80 172 87 86 88 92 142 93 90 91 91 193 92 93 90 63 94 95 95 108 111 97 97 98 149 71 99 29 101 102 102 103 103 141 319 125 147 142 120 108 108 110 12 60 60 114 94 126 115 103 113 47 116 116 47 117 117 96 111 96 107 107 128 123 133 125 124 112 127 121 121 126 124 127 127 128 131 147 273 106 273 134 133 134 105 133 134 168 152 164 185 138 326 70 139 261 148 140 140 142 142 143 143 140 108 71 145 146 146 140 106 148 144 149 99 150 150 153 162 156 213 152 161 161 154 82 136 156 165 154 152 156 158 161 7 161 66 66 163 66 165 164 177 164 166 166 178 171 168 168 408 202 175 177 87 171 167 172 176 333 181 177 165 175 206 179 182 170 175 178 213 182 186 170 179 184 238 182 183 183 181 138 207 188 187 200 187 185 197 197 189 189 196 194 184 194 188 281 193 193 190 184 324 205 203 196 180 216 158 198 242 206 200 244 318 197 203 196 278 204 188 205 70 211 207 211 247 218 217 209 209 210 210 235 212 211 234 213 213 225 235 215 216 216 208 217 227 330 304 219 368 201 294 215 349 228 293 218 224 237 329 169 333 224 225 228 228 249 238 231 231 246 231 258 232 238 207 234 235 233 260 244 237 236 238 238 239 236 257 457 169 237 242 242 255 243 169 244 243 169 246 214 490 135 248 260 240 250 251 251 254 252 499 502 488 253 254 253 258 256 256 258 257 485 243 256 260 141 262 271 263 314 141 264 309 323 263 130 267 265 268 268 266 266 270 281 272 266 273 261 273 274 271 202 286 201 293 303 279 219 278 292 296 280 280 313 263 296 270 284 283 323 312 285 285 340 262 339 282 104 289 315 290 307 291 291 201 290 345 735 292 757 347 298 275 290 297 350 732 295 299 296 300 323 224 301 223 223 303 304 278 328 356 350 306 304 298 307 954 312 309 309 310 310 295 315 312 316 962 313 314 312 315 275 950 316 317 317 318 318 319 319 952 338 321 339 323 337 323 70 205 195 325 326 326 297 327 328 328 223 329 331 225 241 334 484 371 333 332 331 484 332 335 362 327 320 288 320 338 339 337 338 340 342 341 342 363 351 358 287 354 694 346 327 739 355 347 349 348 345 350 351 347 352 349 353 376 354 354 355 355 345 336 373 362 343 357 361 364 457 364 332 360 528 369 358 359 374 452 345 365 378 589 362 694 357 383 370 369 371 371 221 372 526 389 374 472 423 375 377 697 370 376 378 378 380 433 380 528 375 386 424 383 381 384 693 373 423 373 386 447 392 388 387 445 392 419 389 459 392 387 397 398 400 394 393 434 404 621 382 397 397 391 398 408 402 380 628 528 520 400 396 400 404 404 412 667 398 412 422 405 408 408 434 410 409 435 562 405 413 419 506 508 523 428 415 415 416 416 401 524 413 432 419 577 540 523 589 423 417 424 407 658 417 413 220 407 422 416 428 505 534 413 426 431 430 402 517 435 399 434 433 435 434 461 436 440 444 442 442 435 440 455 464 444 442 398 460 454 449 450 464 447 467 240 475 387 454 449 465 451 451 360 446 470 444 454 221 465 480 5 438 457 441 399 459 472 470 461 461 462 462 463 463 465 482 465 465 452 374 361 467 468 468 456 437 470 470 471 467 473 469 473 473 494 484 456 507 474 491 471 495 478 480 480 480 495 482 482 482 489 501 259 484 485 494 486 486 487 486 255 496 485 489 477 492 493 491 492 490 460 493 494 494 495 495 474 483 497 497 503 504 498 498 490 500 501 501 502 499 503 503 411 504 521 458 506 507 469 508 508 537 429 536 578 522 508 535 430 513 514 515 515 518 516 514 517 517 518 519 510 519 519 521 511 522 522 523 505 529 528 525 525 526 526 630 663 530 529 531 718 524 529 417 417 533 534 602 533 512 579 532 540 536 536 551 518 588 539 538 540 540 692 384 381 368 545 542 529 543 545 545 539 550 550 659 712 557 533 550 550 553 551 556 579 566 572 555 555 556 556 557 557 558 564 561 556 563 560 594 559 583 563 565 566 558 561 765 563 565 554 568 766 569 570 765 573 572 572 597 574 598 577 590 577 577 601 618 556 581 585 584 579 581 585 584 599 580 580 561 585 584 586 585 588 587 511 768 590 581 591 590 565 595 561 594 594 596 596 596 607 597 615 605 604 584 602 604 604 598 607 597 604 603 604 598 578 606 605 622 427 608 601 609 582 614 610 603 625 615 366 638 615 615 626 625 617 767 636 619 619 620 635 673 621 427 622 616 629 622 624 633 616 625 623 626 649 628 624 623 627 654 628 667 620 644 632 629 633 627 634 635 635 636 636 637 618 617 636 639 632 367 644 647 367 649 632 649 643 643 652 708 662 640 644 649 612 653 678 661 649 652 651 627 653 649 654 648 655 651 661 666 666 669 658 659 405 405 661 656 678 661 663 657 664 664 665 658 676 527 674 705 527 659 669 658 715 714 634 708 672 672 667 673 674 674 682 706 683 676 677 677 684 670 704 676 680 680 681 707 657 727 721 683 678 684 690 6 685 1106 687 723 696 692 775 650 689 676 692 693 544 691 694 776 710 696 695 694 699 774 873 699 699 698 700 701 701 702 617 704 366 704 705 705 711 706 682 706 708 708 725 725 671 714 711 711 712 710 675 727 776 710 715 715 895 778 718 718 695 902 750 745 682 721 356 761 739 775 731 749 725 710 727 713 726 901 747 282 946 957 298 951 276 954 732 276 293 734 738 760 347 736 760 739 346 872 735 740 745 748 791 742 740 900 741 932 744 740 745 755 741 751 803 725 748 736 751 762 814 749 749 752 742 934 759 822 736 899 746 756 734 760 758 758 759 757 757 760 754 761 761 762 764 2 765 765 571 570 767 767 768 768 899 782 770 808 777 772 773 774 773 769 769 781 548 775 776 776 770 777 802 790 743 780 743 903 783 801 723 800 870 783 720 786 774 785 781 788 769 787 784 785 790 798 794 790 802 804 817 878 787 793 777 791 794 875 795 875 797 796 798 778 799 799 800 698 801 782 747 803 804 807 812 811 805 805 833 806 807 807 808 802 850 791 817 852 804 806 816 812 822 755 728 827 819 816 812 852 811 811 755 812 826 813 822 828 838 828 877 828 825 842 830 849 834 833 845 849 830 830 804 823 811 828 847 837 833 854 853 841 837 836 859 838 920 1090 836 887 841 913 842 829 892 850 886 877 824 845 846 846 864 847 859 849 849 849 850 906 850 878 810 853 853 1183 854 1208 860 865 840 858 859 859 858 860 1167 865 862 1155 860 1153 864 864 336 866 717 783 719 869 698 870 871 871 872 869 873 873 870 874 794 796 881 897 922 880 852 878 911 878 880 880 876 878 986 945 890 939 906 1019 898 897 892 894 914 887 879 890 926 1019 890 890 891 891 922 893 888 892 1018 883 895 895 876 792 885 896 900 900 899 899 900 883 928 741 894 904 903 931 904 905 905 905 916 906 907 907 839 909 909 912 910 892 843 911 907 920 917 926 824 916 937 940 916 916 936 924 912 918 918 919 839 1120 1044 1038 922 922 924 925 924 923 923 923 884 919 1035 753 1207 930 1020 903 947 933 902 931 932 930 998 946 934 934 935 935 936 908 940 937 1160 1116 939 320 839 1137 939 952 1004 989 995 959 939 948 994 945 946 946 998 999 976 951 958 962 950 962 295 951 729 729 944 952 952 961 955 943 949 320 948 949 948 948 977 971 958 960 954 961 950 956 955 963 959 965 960 962 966 320 972 985 968 966 978 967 970 975 971 971 992 993 967 320 1013 976 970 975 976 984 982 982 984 978 973 985 980 1013 981 982 969 1014 1015 984 983 985 987 977 986 988 979 983 993 1042 1014 1052 1008 986 993 933 947 930 994 947 996 998 1013 1207 1002 999 999 1017 981 1001 1012 1206 1023 1003 1003 1004 1002 1005 1004 1032 1062 1007 1007 1008 1007 1009 1052 1027 1020 1011 1028 1003 1020 1013 1012 1020 1061 972 983 1075 1054 1017 1017 1018 1033 1019 889 1020 1036 1021 1021 1005 1030 1000 1026 1026 1024 1025 1051 1026 1025 1034 1037 1031 1011 1065 1060 1030 1028 1066 1031 1028 1031 1046 1047 1034 1034 1056 1027 1065 1036 1027 1037 1043 1098 1042 1014 1074 1053 1039 1041 1042 1040 1043 1044 1043 921 1049 1094 1046 1046 1047 1043 1043 1048 1049 1150 1050 1050 1051 1051 1052 1016 1008 990 990 1055 1025 1098 1097 1056 1090 1029 1057 1058 1078 1040 1061 1061 1016 1016 1076 1074 1064 1064 1065 1065 1066 1056 1083 1072 1063 1071 1053 1070 1069 1075 1083 1097 1082 1108 1029 1074 1075 1070 1076 1075 1060 1084 1078 1078 1079 1079 1084 989 1069 1080 1081 1081 1083 1077 1084 1084 1073 1087 1105 1145 1085 1067 1088 1100 1095 1090 1094 882 1092 1096 1122 1144 1049 1093 1086 1067 1090 1111 1097 1099 1098 1035 1099 1104 1092 1100 1101 1090 1096 1086 1104 1089 1104 1105 1105 1176 716 1116 1108 1103 1110 1114 1086 1109 1130 1132 1123 1128 1123 1113 1096 1115 1116 1114 1117 1130 1128 1125 1119 1119 1120 1120 1121 1150 1132 1133 1123 1118 1143 1156 1125 1125 1126 1126 1118 1159 1128 1127 1129 1130 1093 1130 935 1132 1132 1122 1133 1131 1135 1135 1131 1137 1134 1136 920 1136 1139 1138 1138 1145 1139 1122 1138 1169 1143 940 1143 1142 1170 1154 1132 1161 1146 1145 1146 1147 1093 1127 1149 1150 1150 1151 1151 1193 1141 1153 1153 1167 1160 1112 1139 1124 1155 1166 1195 1158 1157 1140 856 1152 1161 1161 855 1162 1181 1157 1165 1162 1165 1166 1166 1208 1176 1170 1176 1169 1142 1170 1170 1204 1163 1203 1202 1202 1174 1186 1176 1176 1172 1177 1200 1179 1179 1165 1184 1181 1202 1182 1175 1157 1187 1187 1187 1185 1171 1189 1201 1188 1190 1191 1191 1197 1196 1193 1194 1152 1178 1188 1196 1196 1198 1201 1193 1200 1189 1200 1201 1201 1202 1202 1203 1203 1204 1204 1205 1205 1206 1206 928 1207 865 1208 1209 1208 1210 1210; 336 325 732 69 119 26 40 22 22 16 22 15 13 15 21 109 13 17 26 17 12 28 50 17 22 21 25 40 9 10 46 43 20 78 26 30 68 20 27 9 59 45 53 53 26 35 36 53 45 42 33 37 46 39 42 27 61 44 57 10 27 20 58 57 46 45 35 62 95 57 57 48 24 27 95 41 62 98 64 55 37 58 56 97 35 53 44 49 117 48 69 59 115 65 110 163 51 60 34 32 87 62 123 85 30 55 69 102 73 101 195 130 75 69 75 67 75 67 151 96 103 67 74 98 77 68 79 83 84 83 92 191 137 161 82 66 82 85 84 81 84 171 88 170 86 92 80 138 144 90 85 89 71 138 89 84 80 95 30 51 94 96 115 62 77 62 99 108 149 100 72 74 80 72 96 142 281 8 264 146 121 71 107 49 109 113 114 119 114 124 111 119 114 115 47 110 116 61 58 108 123 120 148 121 131 111 8 127 8 126 124 127 106 112 126 121 125 106 128 106 105 147 129 132 134 125 132 125 131 167 156 166 242 92 137 141 143 270 144 268 142 140 145 89 139 147 145 146 144 71 89 264 269 106 148 98 7 66 77 151 163 153 198 161 153 164 156 83 209 210 175 157 168 157 165 152 160 155 162 150 149 163 168 81 174 167 81 82 174 87 152 165 400 219 198 183 81 170 172 167 213 335 186 167 158 174 158 174 181 172 165 176 207 179 189 177 182 181 187 186 174 181 180 193 212 189 186 188 182 200 203 192 186 188 195 195 191 184 215 262 86 191 180 194 325 137 204 197 204 217 206 176 238 209 169 225 299 204 280 204 277 180 200 191 326 199 173 206 246 201 218 158 136 158 157 246 207 212 235 176 206 228 234 208 208 204 218 202 223 335 277 202 378 218 276 226 351 227 294 208 302 169 222 226 330 218 330 208 225 229 234 212 211 251 230 239 246 185 234 207 230 235 243 245 200 243 173 233 233 237 229 469 244 243 185 239 257 240 237 241 250 245 230 249 447 247 251 259 334 241 246 248 249 254 504 499 253 256 255 259 257 257 236 236 258 477 259 260 256 70 270 129 264 285 265 263 192 265 281 269 264 284 140 265 269 129 261 319 105 274 131 272 271 266 274 280 281 224 737 304 291 201 291 290 309 202 279 314 285 307 288 263 313 317 315 263 283 322 319 337 321 288 324 317 276 296 279 277 301 292 305 737 277 753 737 731 310 296 301 349 730 299 295 299 310 324 223 297 329 305 297 303 219 306 343 305 294 306 307 298 731 316 296 300 275 300 951 318 284 312 968 283 283 314 284 318 308 282 284 289 275 311 286 288 730 980 320 283 70 340 289 139 325 325 195 139 325 302 293 294 303 227 223 225 331 250 497 259 360 331 335 241 486 361 222 336 347 338 319 969 344 286 339 340 338 195 288 192 357 354 365 344 355 697 354 297 742 350 352 302 351 343 348 348 351 347 352 355 874 348 353 348 343 371 868 378 356 345 362 222 357 437 468 486 371 530 378 343 364 364 364 222 359 373 511 357 542 364 381 357 383 359 222 373 369 389 385 359 457 525 700 362 699 383 700 368 369 404 440 424 368 699 388 425 376 383 376 694 221 382 388 385 490 389 221 385 391 393 430 394 464 397 450 392 395 393 389 403 439 403 427 394 391 395 445 390 409 439 407 633 543 516 408 382 403 396 400 416 653 409 406 522 425 395 406 409 402 410 409 547 407 415 412 426 419 511 427 406 413 412 414 653 528 430 506 414 429 532 508 422 396 524 396 425 659 423 431 515 380 589 522 522 513 429 402 514 402 431 410 506 440 432 410 439 443 435 443 442 443 453 439 399 390 438 468 445 462 436 445 459 448 462 463 459 466 360 486 507 388 453 442 451 221 450 467 455 459 438 457 468 482 481 501 473 472 481 458 443 473 438 390 445 442 444 391 450 464 446 450 455 466 467 471 466 221 364 476 469 443 460 361 471 457 472 438 460 474 240 493 476 496 489 487 460 469 446 466 479 479 441 465 481 488 475 253 477 253 492 500 332 471 477 498 474 489 485 489 479 494 490 480 491 495 479 479 491 481 493 488 507 332 334 255 503 499 503 500 490 5 483 483 502 498 254 514 499 511 506 458 475 476 414 411 509 521 532 589 422 513 533 514 516 426 520 426 519 510 516 426 418 532 515 516 520 536 505 521 428 414 414 521 530 385 423 524 385 382 653 642 528 543 530 693 417 544 425 660 535 512 609 579 510 533 535 539 510 519 421 537 596 535 555 509 538 683 542 542 543 541 381 668 544 381 543 555 540 549 529 548 558 539 538 546 555 549 557 553 569 567 538 549 549 547 549 554 549 568 562 559 562 573 420 554 586 554 594 554 421 563 552 566 567 557 564 573 566 766 566 560 573 765 588 567 702 533 576 534 576 583 619 580 575 586 583 581 579 547 575 583 584 547 585 561 586 584 592 583 582 429 578 429 591 575 591 593 596 567 592 593 592 588 595 609 599 609 606 597 601 597 608 601 605 611 602 599 609 600 615 591 575 767 613 620 600 608 598 610 702 609 611 624 701 616 614 598 607 612 622 638 578 613 618 589 621 636 631 422 621 614 626 641 624 623 628 703 623 632 623 704 401 628 639 630 627 630 673 618 641 641 633 629 641 679 401 631 631 637 631 768 702 638 623 639 647 655 644 656 627 655 642 644 641 642 681 650 646 647 643 648 654 675 656 663 647 664 654 634 645 653 655 648 662 651 676 658 531 666 672 660 620 652 650 686 662 652 681 651 657 634 667 666 672 658 674 718 668 659 669 670 721 680 682 527 659 674 705 667 660 707 651 676 692 662 675 715 727 680 679 645 708 645 681 664 726 671 548 685 686 686 688 686 1089 715 353 697 693 774 690 692 683 689 689 718 696 375 709 772 691 697 689 695 775 869 697 698 870 376 607 616 614 767 645 703 634 620 673 713 711 708 707 671 680 670 726 721 709 706 709 683 771 713 675 771 721 687 688 896 898 544 527 775 1206 761 744 711 710 871 746 723 695 298 688 720 714 709 675 727 902 751 316 941 958 276 295 294 730 733 734 758 759 737 752 346 749 758 346 739 740 738 719 743 720 811 724 742 904 744 933 724 745 740 812 743 748 797 688 747 750 747 754 813 751 750 724 752 724 757 816 754 790 805 751 294 735 737 757 729 759 760 757 762 755 762 761 763 763 571 572 764 560 605 617 617 618 793 801 776 802 771 548 772 772 771 774 787 786 775 548 726 770 875 769 720 784 781 781 719 779 781 698 868 695 869 782 808 789 787 789 783 789 794 784 789 799 789 779 793 778 804 805 806 810 794 795 875 896 795 796 797 797 795 802 789 904 774 785 785 699 785 873 756 791 802 761 816 810 802 746 834 817 746 806 796 808 815 803 815 815 811 818 819 842 827 762 814 829 812 820 809 818 818 804 814 819 827 820 809 846 824 830 879 827 830 829 828 827 831 824 843 829 822 825 807 831 832 831 837 838 846 717 847 838 834 838 857 841 1151 1095 853 906 836 840 815 842 844 829 893 890 907 826 828 833 858 836 841 826 843 851 844 911 851 852 818 836 835 1204 848 1210 848 860 913 840 840 848 864 857 1179 862 865 1148 865 1144 865 857 377 1210 867 869 869 719 873 868 356 868 719 872 698 782 874 870 769 875 878 896 923 885 844 877 851 879 877 876 895 881 993 953 877 946 910 894 778 898 893 885 912 912 844 886 927 1205 888 889 889 885 919 886 879 922 1019 900 811 876 885 795 898 897 898 902 778 792 885 902 1206 780 1206 902 780 930 901 779 798 904 838 887 843 824 913 913 841 914 887 851 907 910 914 935 935 925 838 908 908 913 909 914 918 915 908 917 923 925 864 1109 1045 1035 892 887 923 923 917 924 925 921 927 926 1038 941 1017 932 942 932 1003 934 883 929 903 933 1004 753 744 933 917 920 917 937 1151 920 1166 1110 996 968 1144 1151 945 953 1001 1071 994 978 933 953 996 944 724 939 995 996 959 308 963 282 316 950 731 961 952 941 955 729 957 949 944 970 958 966 957 961 958 970 967 960 948 959 961 954 949 966 970 975 988 971 971 965 956 321 1008 987 321 968 982 973 943 959 964 965 1055 991 985 980 999 974 975 970 970 942 969 978 959 972 980 986 973 1021 974 956 982 1015 978 974 988 967 287 1007 4 959 987 1015 986 1061 1051 1016 1009 967 992 930 996 929 995 997 944 1005 943 1000 998 943 944 1018 943 1023 1001 1207 1005 947 1001 947 1005 1002 1005 1011 990 977 992 972 1008 972 1009 1010 1011 1012 1011 1012 1021 997 1021 1036 1051 978 1050 1069 991 928 1010 928 1034 928 1205 974 1011 974 1012 1023 1023 1017 1023 1010 1023 1026 1061 1024 1028 1010 1027 1032 1022 1029 1056 1022 1030 1032 1039 1025 1036 927 1043 1018 1033 1057 1037 1039 1041 1033 1035 1038 1044 1029 1050 1029 1062 1041 1039 1039 1006 1033 1048 1044 1045 1045 1045 1033 884 884 1047 1048 1043 921 1094 983 942 1006 1009 1006 1055 1064 1062 1054 1016 1037 1059 1098 1066 1096 1065 1035 1060 1079 1042 1014 1042 1052 1063 1006 1075 1009 1055 1031 1056 1025 1065 1082 1081 1076 989 1070 1062 1071 1082 1078 1099 1083 1103 1060 1068 1068 1076 1063 1076 1058 1077 1068 1060 1058 1100 1085 1081 1082 1088 1082 1071 1068 1084 1073 1080 1090 1072 1100 1147 1088 1087 1085 1073 1102 1086 882 1099 1058 1101 1133 1170 915 1099 1102 1085 1073 1120 1058 882 1058 1044 1093 1091 1100 1092 1092 1105 1129 1110 1092 1103 1099 1073 1101 1173 1106 716 1085 1108 1086 1109 1096 1114 1121 1119 1119 1155 1118 716 1091 1111 1102 1126 1120 1149 1138 1112 1107 1109 1111 1117 1111 1149 1107 1121 1109 1125 1122 1159 1113 1118 1113 1114 1107 1156 1118 1155 1111 1091 1149 1129 1135 1133 1117 1146 1117 1134 1134 920 1150 1136 1136 1134 937 1137 1124 1139 1107 1141 1141 1143 1145 1154 938 839 1136 1169 864 1169 1146 1147 1143 1107 1147 1146 1094 1156 1131 915 1131 937 940 1171 1161 1142 938 1168 1161 1125 1156 1158 1156 1124 1188 1124 1159 1154 1159 1165 1141 1140 1203 1179 1163 1158 1152 1154 1163 1160 1163 1167 1174 1209 1179 1140 1153 1144 863 1185 1181 1178 1179 1178 1167 1180 1167 1164 1187 1173 1191 1152 1162 1182 1186 1175 1200 1194 1184 1204 1184 1193 1175 1204 1186 1198 1198 1195 1198 1175 1178 1201 1187 1199 1175 1200 1203 1192 1191 1192 1190 1193 1198 1178 1199 1194 1187 1197 1152 1173 1173 855 1157 1183 889 886 928 929 1017 1002 1210 861 1168 1209 866 1208]

# ╔═╡ 7818342e-0712-4e54-9a8d-6c9e49b44237
pts = [0.0 1.0 0.5 0.25 0.5 0.75 0.25 0.125 0.125 0.0625 0.0625 0.03125 0.03125 0.015625 0.015625 0.0078125 0.046875 0.03125 0.046875 0.09375 0.071875 0.0579545 0.09375 0.046875 0.0895833 0.113871 0.0635417 0.109375 0.1875 0.158692 0.0610677 0.15625 0.130795 0.140625 0.116375 0.12734 0.0937466 0.0976117 0.111631 0.0797852 0.126351 0.071958 0.0797526 0.109378 0.124879 0.149946 0.09375 0.078125 0.103495 0.0621143 0.143427 0.161754 0.145166 0.0852872 0.149533 0.132406 0.101106 0.0872776 0.140625 0.135899 0.0859375 0.166985 0.12319 0.154532 0.171875 0.219015 0.19231 0.170635 0.175616 0.278759 0.214311 0.190299 0.177842 0.205555 0.163449 0.191778 0.190151 0.186817 0.207754 0.220716 0.265872 0.238841 0.22449 0.21988 0.242469 0.268479 0.260821 0.244893 0.249306 0.232143 0.233528 0.253488 0.22136 0.151459 0.138527 0.184004 0.173994 0.188022 0.21875 0.203125 0.204661 0.206034 0.179492 0.1875 0.15625 0.1951 0.197257 0.198483 0.0234375 0.111798 0.138211 0.109375 0.130281 0.147856 0.118965 0.109352 0.0968476 0.117188 0.161289 0.178495 0.178528 0.140625 0.156296 0.137232 0.155069 0.126023 0.163256 0.17496 0.189866 0.187238 0.168136 0.148438 0.132812 0.157062 0.375 0.3125 0.287618 0.269161 0.268831 0.236732 0.256915 0.237703 0.252842 0.217726 0.214669 0.231017 0.216775 0.201714 0.212724 0.205896 0.234375 0.27757 0.258891 0.28125 0.252186 0.279429 0.296875 0.310753 0.292964 0.265625 0.262582 0.242714 0.226274 0.271498 0.299441 0.256438 0.289307 0.284116 0.412617 0.294847 0.277442 0.280804 0.366754 0.329033 0.309899 0.331292 0.311266 0.324585 0.3457 0.355572 0.337192 0.354288 0.327281 0.316055 0.388437 0.359089 0.370708 0.384324 0.368645 0.332542 0.298024 0.341163 0.281295 0.315006 0.326275 0.343179 0.345423 0.314565 0.34375 0.395229 0.420331 0.381174 0.360884 0.371323 0.306657 0.339266 0.358275 0.407872 0.322445 0.304109 0.359375 0.354779 0.34236 0.4375 0.402265 0.389109 0.393529 0.41531 0.401359 0.75 0.59085 0.503995 0.460224 0.437634 0.441616 0.422435 0.445739 0.426534 0.410618 0.383742 0.377114 0.40625 0.388954 0.372154 0.394882 0.418929 0.411662 0.386216 0.402679 0.462945 0.441825 0.400939 0.437897 0.426981 0.426289 0.403931 0.390625 0.421875 0.431152 0.447653 0.418768 0.46875 0.464929 0.453125 0.449909 0.439656 0.430584 0.419201 0.456449 0.436002 0.171875 0.212807 0.244008 0.234861 0.263601 0.216951 0.250409 0.247989 0.217469 0.193483 0.187744 0.164062 0.174341 0.20339 0.318221 0.386307 0.38884 0.390162 0.359802 0.368237 0.229644 0.287989 0.261642 0.274604 0.247654 0.238323 0.21875 0.215952 0.293071 0.368368 0.372848 0.386019 0.440677 0.412702 0.346063 0.342788 0.45139 0.363138 0.336933 0.321563 0.438857 0.45801 0.425817 0.407724 0.486253 0.404051 0.353435 0.312604 0.340638 0.327084 0.32904 0.287014 0.276234 0.272778 0.298318 0.298524 0.28728 0.315673 0.222042 0.256032 0.262951 0.203125 0.278623 0.300403 0.30604 0.287203 0.442952 0.426561 0.481088 0.464108 0.455856 0.496142 0.476918 0.467422 0.494188 0.564866 0.23942 0.230053 0.247875 0.223214 0.195312 0.324823 0.529049 0.210938 0.510142 0.49533 0.467875 0.498069 0.474907 0.498145 0.48341 0.464229 0.523633 0.505985 0.516494 0.543511 0.570337 0.55208 0.537924 0.523278 0.507902 0.574082 0.552095 0.563059 0.528266 0.875 0.8125 0.62584 0.597738 0.582269 0.521523 0.582099 0.609971 0.543141 0.620609 0.594925 0.579835 0.61043 0.625 0.716724 0.61536 0.66665 0.596618 0.612499 0.640931 0.626567 0.621437 0.608478 0.644226 0.641628 0.637121 0.634754 0.662912 0.65784 0.661363 0.691368 0.653818 0.642506 0.6875 0.685211 0.764618 0.686656 0.679349 0.699144 0.733744 0.685315 0.717697 0.677477 0.664064 0.6784 0.738062 0.706789 0.709371 0.734933 0.700815 0.7203 0.695461 0.71875 0.721904 0.9375 0.875 0.762991 0.688843 0.701542 0.712993 0.725225 0.742406 0.730645 0.785858 0.724317 0.706253 0.688618 0.651135 0.66667 0.648307 0.65625 0.5625 0.6101 0.669473 0.631078 0.591775 0.655496 0.617137 0.623312 0.616658 0.563669 0.528676 0.59375 0.640625 0.609761 0.602131 0.558064 0.609375 0.601562 0.580205 0.563576 0.578125 0.703125 0.607596 0.588362 0.625567 0.639033 0.621334 0.60032 0.591389 0.545684 0.536874 0.57443 0.553054 0.603911 0.520876 0.568384 0.588998 0.514446 0.53125 0.537979 0.49346 0.546875 0.546817 0.547893 0.568678 0.582494 0.515282 0.474377 0.479631 0.489556 0.508533 0.488721 0.49855 0.518479 0.515905 0.534906 0.549046 0.533408 0.576258 0.521997 0.483089 0.469843 0.486842 0.503212 0.502098 0.499715 0.469576 0.484375 0.759185 0.709278 0.533953 0.743505 0.8125 0.777615 0.770017 0.774851 0.756436 0.745585 0.744453 0.762787 0.727212 0.78125 0.765625 0.758921 0.771711 0.747945 0.751428 0.667218 0.675258 0.657552 0.685483 0.647581 0.672665 0.664036 0.684678 0.796875 0.815458 0.796124 0.797696 0.780469 0.796875 0.828485 0.814613 0.814232 0.651434 0.631976 0.647667 0.66144 0.632152 0.84375 0.88211 0.66203 0.859375 0.843487 0.859375 0.9375 0.845148 0.899927 0.84432 0.86905 0.87942 0.872247 0.889557 0.96875 0.92065 0.901925 0.919664 0.90625 0.953125 0.924037 0.941061 0.90766 0.921875 0.984375 0.96875 0.939084 0.952984 0.960938 0.83912 0.816595 0.802041 0.803148 0.833133 0.857869 0.8359 0.90625 0.877316 0.856978 0.895829 0.873605 0.921875 0.902668 0.782638 0.798631 0.818632 0.918046 0.945312 0.931195 0.929688 0.917613 0.898568 0.853334 0.885236 0.873421 0.860096 0.890127 0.890625 0.872796 0.833597 0.828188 0.864167 0.853744 0.875482 0.898438 0.882812 0.84375 0.813501 0.837757 0.851769 0.848542 0.813786 0.784354 0.777858 0.759889 0.759123 0.831035 0.828038 0.815019 0.834403 0.859375 0.784622 0.790642 0.80419 0.779793 0.774969 0.819227 0.798874 0.734145 0.776668 0.794675 0.787865 0.808692 0.834298 0.828125 0.801881 0.785988 0.789818 0.807213 0.75663 0.820312 0.800912 0.835938 0.772433 0.78125 0.765199 0.780013 0.754301 0.766374 0.820021 0.796875 0.751476 0.719427 0.695915 0.717443 0.784234 0.773803 0.768984 0.763901 0.733189 0.710673 0.741892 0.682295 0.708469 0.729423 0.705793 0.699787 0.755238 0.736234 0.743335 0.694761 0.758498 0.758769 0.71588 0.734735 0.743973 0.723245 0.682186 0.765625 0.742194 0.748061 0.757812 0.725898 0.656775 0.773438 0.665769 0.675267 0.660236 0.640493 0.623716 0.644525 0.62738 0.58921 0.60784 0.592767 0.850153 0.833577 0.867188 0.750826 0.748442 0.740205 0.737975 0.72448 0.697524 0.688602 0.716264 0.679121 0.729995 0.698827 0.742734 0.375 0.625 0.669339 0.533541 0.685414 0.70463 0.6875 0.520693 0.467766 0.709184 0.699753 0.71561 0.71875 0.416232 0.379072 0.365504 0.384393 0.398606 0.405685 0.460723 0.734375 0.456876 0.479125 0.499689 0.511752 0.50779 0.485184 0.515714 0.491689 0.499591 0.6825 0.694266 0.696106 0.727431 0.707942 0.712109 0.46776 0.442749 0.726562 0.685947 0.69903 0.429804 0.441225 0.414059 0.445872 0.691471 0.703565 0.992188 0.984375 0.953125 0.970293 0.813573 0.800747 0.646619 0.671158 0.674232 0.665398 0.658041 0.630749 0.638127 0.687528 0.66256 0.575396 0.549084 0.528724 0.545266 0.564357 0.550148 0.608393 0.594995 0.571097 0.626366 0.607684 0.587272 0.591224 0.627242 0.600818 0.608053 0.630736 0.622078 0.653678 0.637303 0.568025 0.614556 0.600411 0.582546 0.668676 0.647697 0.658756 0.6751 0.661491 0.667221 0.669356 0.651653 0.614676 0.632033 0.663829 0.703125 0.710938 0.63108 0.677392 0.648709 0.634089 0.679544 0.695312 0.65625 0.663753 0.671875 0.596868 0.679688 0.628255 0.644981 0.649672 0.624223 0.664573 0.664062 0.644675 0.623618 0.642627 0.640625 0.61873 0.626876 0.602551 0.562838 0.587639 0.601009 0.637802 0.602202 0.591188 0.614633 0.633361 0.648438 0.61448 0.617985 0.611796 0.597045 0.605864 0.626436 0.632812 0.5625 0.4375 0.589799 0.5785 0.597984 0.607707 0.59375 0.609375 0.568193 0.572786 0.589718 0.601562 0.617188 0.545577 0.54598 0.569357 0.539301 0.530255 0.572164 0.578473 0.651019 0.596693 0.573732 0.594188 0.582563 0.585357 0.603428 0.453963 0.535734 0.516325 0.567164 0.547778 0.552898 0.564066 0.540382 0.555949 0.557512 0.568344 0.563089 0.540892 0.615917 0.605567 0.585366 0.566932 0.583713 0.552111 0.525375 0.535181 0.52269 0.550352 0.555572 0.577907 0.593041 0.559214 0.580291 0.568569 0.585972 0.559851 0.559268 0.578205 0.502473 0.587619 0.529697 0.545421 0.541716 0.526073 0.502501 0.555071 0.529766 0.517045 0.525923 0.532986 0.512846 0.5071 0.502751 0.493649 0.517998 0.505743 0.474436 0.47761 0.522442 0.542231 0.545429 0.532947 0.446612 0.550214 0.429252 0.36891 0.403701 0.42138 0.430823 0.452055 0.45105 0.381286 0.332106 0.312114 0.330902 0.393928 0.410232 0.355061 0.399845 0.297174 0.372955 0.353988 0.342372 0.342726 0.339201 0.293314 0.362567 0.319824 0.312658 0.277621 0.268853 0.276603 0.27869 0.380008 0.324671 0.308677 0.254309 0.382726 0.35965 0.371425 0.295403 0.320889 0.234375 0.245097 0.388245 0.295738 0.343372 0.362528 0.248721 0.258512 0.226562 0.336334 0.3125 0.28125 0.265625 0.280676 0.265404 0.456845 0.472249 0.439218 0.428027 0.481936 0.415935 0.479488 0.449614 0.487024 0.44381 0.462975 0.468327 0.330599 0.282395 0.293906 0.314309 0.480018 0.414285 0.415677 0.415791 0.34439 0.326591 0.30233 0.488235 0.501325 0.520688 0.392708 0.398677 0.432834 0.456259 0.470505 0.439112 0.450874 0.462851 0.428903 0.384123 0.440039 0.400569 0.412681 0.489103 0.492802 0.449315 0.384619 0.450279 0.470171 0.379505 0.355148 0.363899 0.362972 0.491705 0.466105 0.478422 0.50077 0.507643 0.481509 0.493731 0.352484 0.329667 0.31569 0.296875 0.273438 0.286178 0.414599 0.433166 0.418597 0.421429 0.394979 0.347111 0.304531 0.317599 0.296902 0.399858 0.419327 0.34375 0.365598 0.333149 0.314423 0.324259 0.328125 0.382538 0.368284 0.34988 0.334325 0.3788 0.379833 0.399242 0.35351 0.333478 0.350408 0.365853 0.370593 0.361095 0.39841 0.335938 0.345802 0.359375 0.38408 0.428071 0.416674 0.464587 0.476138 0.37592 0.408257 0.434049 0.438103 0.449127 0.397776 0.408171 0.387386 0.351562 0.432376 0.39359 0.367188 0.46138 0.362777 0.426284 0.407838 0.429671 0.40625 0.390625 0.40537 0.413749 0.390938 0.454344 0.43345 0.445287 0.437048 0.455505 0.493728 0.422276 0.468353 0.415 0.404397 0.442716 0.431649 0.437738 0.446502 0.48669 0.472747 0.474241 0.493686 0.506396 0.512201 0.524166 0.453657 0.462035 0.515374 0.489444 0.517103 0.509187 0.544951 0.477798 0.49265 0.497639 0.421875 0.470166 0.484379 0.540388 0.518222 0.531938 0.542348 0.427642 0.44798 0.463287 0.46041 0.429688 0.509686 0.50468 0.530736 0.48983 0.578125 0.50508 0.491043 0.561285 0.546969 0.527425 0.552562 0.46875 0.53125 0.548381 0.585938 0.497684 0.562989 0.570312 0.534783 0.532484 0.453125 0.484403 0.50207 0.47433 0.485741 0.445312 0.460938 0.510974 0.546875 0.484375 0.515625 0.521456 0.539062 0.497329 0.509655 0.537231 0.525397 0.523438 0.5 0.476562 0.520089 0.512924 0.533731 0.549264 0.461522 0.533158 0.515888 0.498868 0.577064 0.560868 0.588957; 0.0 0.0 0.866025 0.433013 0.0 0.433013 0.0 0.216506 0.0 0.108253 0.0 0.0541266 0.0 0.0270633 0.0 0.0135316 0.0270633 0.0180422 0.0 0.0360844 0.0234549 0.0154179 0.0 0.0811899 0.0180422 0.0219355 0.0541266 0.0 0.0 0.107055 0.0369865 0.0 0.0673842 0.0217187 0.0468784 0.0331801 0.078114 0.0570995 0.0705409 0.0646834 0.100154 0.0867918 0.0469585 0.0901254 0.0832679 0.0835612 0.16238 0.135316 0.119459 0.0719112 0.099363 0.0501576 0.0404132 0.105481 0.0646262 0.0509284 0.103723 0.121736 0.0 0.134639 0.148848 0.0242028 0.119593 0.015244 0.0 0.0481927 0.0811664 0.0736372 0.0942605 0.208569 0.163034 0.131272 0.114795 0.106791 0.138623 0.0976767 0.0423759 0.0620168 0.0663734 0.133234 0.0897929 0.0730504 0.0641614 0.0876339 0.0973855 0.137692 0.114088 0.1323 0.167274 0.115644 0.1515 0.148981 0.103675 0.124288 0.114699 0.160988 0.0391198 0.0212382 0.0 0.0 0.145371 0.124084 0.144508 0.32476 0.270633 0.222132 0.190162 0.172008 0.0405949 0.143911 0.176606 0.189443 0.155995 0.149201 0.17218 0.159574 0.135798 0.202975 0.169329 0.181088 0.204956 0.24357 0.193729 0.200423 0.220727 0.188034 0.207572 0.224981 0.25775 0.23917 0.251348 0.257101 0.230038 0.237975 0.0 0.0 0.1711 0.162188 0.18644 0.214134 0.205078 0.189277 0.1832 0.200944 0.182111 0.17169 0.222604 0.206521 0.0241302 0.0387687 0.0167194 0.0414259 0.0239699 0.0 0.0498663 0.0207147 0.0 0.033966 0.0323689 0.0 0.0387419 0.0335974 0.0334414 0.0655555 0.0600939 0.0748316 0.0824609 0.055902 0.124847 0.116093 0.12174 0.102454 0.0737245 0.0869798 0.0783111 0.056542 0.100745 0.0713469 0.0719914 0.139582 0.115859 0.096416 0.103067 0.122324 0.100406 0.117875 0.104009 0.133103 0.130562 0.138443 0.144489 0.197531 0.149593 0.139887 0.169531 0.156763 0.177308 0.0511292 0.0 0.117868 0.203895 0.195398 0.191125 0.164004 0.163206 0.0286555 0.0479789 0.161195 0.0175396 0.0168626 0.0171233 0.0324069 0.0446235 0.0 0.141979 0.156244 0.177521 0.182192 0.196852 0.0 0.153011 0.173106 0.183233 0.18846 0.154236 0.14652 0.172059 0.1664 0.0450098 0.0340822 0.01762 0.0 0.0597768 0.0576773 0.0451987 0.0804768 0.102077 0.08006 0.0719558 0.107307 0.12602 0.0888751 0.0988803 0.1314 0.112869 0.0229418 0.0 0.0 0.0267413 0.111322 0.0160577 0.0 0.0599681 0.0178333 0.0414551 0.0651234 0.0467778 0.0617058 0.0829229 0.0818573 0.297696 0.292106 0.259023 0.237471 0.233334 0.25779 0.243589 0.22678 0.240194 0.303263 0.280819 0.284165 0.268477 0.270626 0.249745 0.264979 0.229812 0.211457 0.221351 0.20652 0.276731 0.309379 0.288007 0.2611 0.276661 0.303141 0.378886 0.321097 0.234351 0.246012 0.230638 0.247283 0.255246 0.2557 0.285074 0.241078 0.221295 0.268925 0.263634 0.220066 0.207246 0.202775 0.231494 0.221947 0.196079 0.239733 0.254657 0.286659 0.222023 0.235716 0.27966 0.283654 0.296944 0.276796 0.262743 0.296099 0.250102 0.26758 0.305771 0.348119 0.321275 0.351823 0.225508 0.213418 0.188625 0.191489 0.237298 0.247553 0.173753 0.158421 0.140233 0.132922 0.141643 0.125497 0.154163 0.238451 0.329524 0.35461 0.315958 0.337765 0.338291 0.188261 0.210791 0.365354 0.19619 0.253825 0.249575 0.223808 0.219258 0.207394 0.237205 0.232756 0.241275 0.239664 0.224284 0.229083 0.193011 0.207445 0.176432 0.151113 0.144714 0.216511 0.188701 0.16172 0.192263 0.216506 0.32476 0.196214 0.200693 0.203158 0.169024 0.173783 0.173295 0.156518 0.253792 0.238173 0.231834 0.18993 0.0 0.116334 0.220861 0.151004 0.219454 0.238201 0.168812 0.178757 0.143332 0.155548 0.146094 0.0721237 0.109313 0.128847 0.122348 0.137582 0.0933828 0.133208 0.108311 0.0910977 0.0 0.106631 0.215787 0.0501131 0.121991 0.117405 0.169396 0.0783564 0.145283 0.0924652 0.0710857 0.0638744 0.0667086 0.092573 0.0653457 0.094763 0.0782704 0.100881 0.177477 0.0 0.0802701 0.108253 0.0 0.132536 0.155765 0.145767 0.16581 0.0342752 0.146235 0.130357 0.0862418 0.0533918 0.0429538 0.0250824 0.0346311 0.0474794 0.0538657 0.0 0.0 0.0384225 0.0282745 0.0415071 0.0860026 0.0175819 0.0634369 0.0214457 0.0933591 0.118647 0.116471 0.0 0.0 0.119234 0.137144 0.140108 0.0 0.0193654 0.133499 0.0478233 0.0175621 0.0 0.0782489 0.05902 0.078537 0.0164266 0.108462 0.102861 0.118919 0.125413 0.140349 0.149309 0.0241363 0.0536608 0.132496 0.0315069 0.0364348 0.0554673 0.0 0.0450134 0.0861202 0.0 0.0824984 0.103999 0.0921814 0.10346 0.0256347 0.0912041 0.0725681 0.109822 0.116577 0.0470847 0.0670944 0.0926962 0.0741422 0.0938718 0.0621885 0.070749 0.0742582 0.040751 0.124155 0.040632 0.0217412 0.0998989 0.0161221 0.033528 0.0232986 0.0 0.0832665 0.0202953 0.0194744 0.0815175 0.0 0.0390938 0.107211 0.0638717 0.0650214 0.0448223 0.0220822 0.0485953 0.0161507 0.0 0.0156626 0.0316735 0.0920226 0.114725 0.0982323 0.179301 0.164985 0.165341 0.242557 0.190331 0.211993 0.196631 0.195141 0.0211688 0.0577838 0.0672972 0.0460899 0.0228503 0.0 0.0281494 0.0397319 0.0176295 0.238868 0.237974 0.214745 0.224949 0.22118 0.0 0.0694024 0.300048 0.0266873 0.0180715 0.0 0.0 0.0531547 0.0325118 0.0356519 0.0489455 0.031206 0.016306 0.0502513 0.0541266 0.0733655 0.0660842 0.0481826 0.0 0.0811899 0.022409 0.0599557 0.017145 0.0 0.0270633 0.0 0.0382197 0.0479131 0.0676582 0.112017 0.090644 0.0827573 0.130039 0.0736217 0.0743822 0.0928546 0.16238 0.107558 0.0991651 0.090505 0.0877129 0.135316 0.116505 0.12435 0.106511 0.111053 0.0961363 0.0947215 0.086106 0.121785 0.111799 0.139902 0.167913 0.125508 0.149822 0.125878 0.155339 0.189443 0.134491 0.143177 0.126149 0.19221 0.145093 0.172602 0.175911 0.202975 0.270633 0.200236 0.189422 0.183174 0.21374 0.168268 0.156137 0.140548 0.165974 0.14904 0.206954 0.240661 0.222517 0.22424 0.24357 0.25807 0.212985 0.244029 0.234481 0.189673 0.264384 0.227871 0.256115 0.205455 0.193349 0.175201 0.184208 0.256484 0.297696 0.261467 0.29667 0.27721 0.290741 0.280785 0.311228 0.30914 0.284165 0.271654 0.378886 0.336717 0.318988 0.239669 0.253957 0.280662 0.351823 0.310197 0.217502 0.215859 0.191502 0.337567 0.357581 0.304076 0.320578 0.234368 0.246253 0.213219 0.225924 0.205263 0.385048 0.29073 0.232759 0.19696 0.194969 0.357939 0.267013 0.351686 0.37479 0.266722 0.278159 0.293725 0.313111 0.287286 0.405949 0.374478 0.394567 0.419481 0.414956 0.269451 0.392418 0.283656 0.260421 0.253088 0.25746 0.292683 0.287157 0.272821 0.27605 0.269054 0.257217 0.198474 0.17 0.230038 0.263494 0.180098 0.33284 0.317603 0.294152 0.351553 0.319719 0.336266 0.304744 0.346858 0.334021 0.411017 0.649519 0.649519 0.239383 0.30181 0.395023 0.309632 0.541266 0.272835 0.317105 0.397825 0.375596 0.362001 0.487139 0.307926 0.302873 0.288528 0.284251 0.298384 0.277052 0.283329 0.460076 0.264879 0.269866 0.274435 0.294136 0.34015 0.29686 0.317655 0.321835 0.306443 0.448168 0.422161 0.406548 0.438025 0.456281 0.427458 0.299484 0.308304 0.473608 0.481307 0.44028 0.286051 0.272761 0.29143 0.291596 0.463983 0.477382 0.0135316 0.0 0.0184953 0.0171513 0.147733 0.158135 0.348651 0.366123 0.340798 0.321242 0.336339 0.322475 0.305007 0.364605 0.35223 0.373933 0.342052 0.33488 0.320404 0.295373 0.304332 0.353928 0.315189 0.327009 0.343031 0.332487 0.343894 0.362824 0.411324 0.393323 0.374288 0.366585 0.390511 0.393717 0.39778 0.346142 0.310572 0.294391 0.300843 0.415495 0.41601 0.438953 0.432038 0.467102 0.452338 0.398973 0.512195 0.457444 0.437401 0.491232 0.514203 0.500671 0.486227 0.513127 0.480519 0.465446 0.496645 0.527734 0.595392 0.533911 0.568329 0.556947 0.554798 0.53946 0.530141 0.555401 0.512555 0.549807 0.581861 0.450572 0.568048 0.578506 0.622456 0.599438 0.584631 0.580893 0.631868 0.610232 0.598568 0.502035 0.529609 0.489895 0.551921 0.554958 0.608924 0.625403 0.527244 0.495262 0.509655 0.475791 0.61403 0.635987 0.757772 0.757772 0.637741 0.624789 0.623259 0.64271 0.703646 0.676582 0.672107 0.65102 0.665751 0.690114 0.663051 0.259236 0.281975 0.262548 0.244519 0.285931 0.279769 0.248514 0.372699 0.427772 0.457043 0.458893 0.473414 0.442521 0.443976 0.532215 0.41471 0.498576 0.416806 0.48161 0.531707 0.472824 0.448408 0.457002 0.435776 0.507414 0.490504 0.431168 0.426883 0.411363 0.407035 0.394535 0.389389 0.405826 0.376076 0.394153 0.355124 0.374225 0.357907 0.539059 0.542011 0.577867 0.587855 0.524614 0.52316 0.554268 0.604986 0.557737 0.564529 0.572216 0.551209 0.546363 0.50754 0.591492 0.530104 0.516031 0.527739 0.539437 0.511832 0.491825 0.466056 0.43272 0.400047 0.369121 0.391776 0.357087 0.346234 0.330179 0.570861 0.567001 0.591454 0.622159 0.337277 0.619866 0.324847 0.427459 0.37887 0.354178 0.340991 0.321807 0.371187 0.344145 0.333556 0.317697 0.303273 0.322771 0.336221 0.311993 0.35511 0.357061 0.322935 0.333851 0.383696 0.357503 0.318109 0.333079 0.352444 0.370293 0.337542 0.347907 0.392096 0.331111 0.371084 0.370522 0.352073 0.414713 0.372479 0.400602 0.37271 0.38646 0.391717 0.394169 0.405949 0.359586 0.38494 0.37436 0.413017 0.405157 0.391822 0.412133 0.392418 0.39858 0.541266 0.487139 0.460076 0.433344 0.427118 0.353183 0.368077 0.357444 0.379043 0.38937 0.368955 0.41977 0.404643 0.405078 0.387635 0.386254 0.404191 0.465405 0.410533 0.42287 0.442364 0.454551 0.43382 0.405057 0.389173 0.441017 0.425463 0.468112 0.437031 0.450879 0.44704 0.422719 0.406715 0.420303 0.428582 0.439402 0.469679 0.450461 0.471 0.450223 0.486124 0.436468 0.460759 0.449965 0.481216 0.465075 0.500287 0.444223 0.482961 0.492758 0.465116 0.481482 0.450224 0.466818 0.506782 0.514145 0.535537 0.492944 0.513564 0.519518 0.547309 0.426704 0.448785 0.458667 0.514203 0.473608 0.454045 0.487869 0.490699 0.523964 0.505382 0.510858 0.458649 0.49174 0.478607 0.440158 0.478312 0.468629 0.595392 0.509004 0.507147 0.507648 0.525075 0.568329 0.561108 0.492239 0.499227 0.488 0.533518 0.517149 0.531007 0.550992 0.544135 0.527365 0.524679 0.548678 0.575932 0.606988 0.581861 0.56696 0.622456 0.58626 0.570939 0.548795 0.569366 0.554346 0.607219 0.584863 0.537651 0.517507 0.553693 0.549879 0.565827 0.628628 0.608924 0.554718 0.573054 0.635987 0.652696 0.594291 0.634816 0.625419 0.605365 0.703646 0.676582 0.653066 0.600561 0.644851 0.623123 0.664136 0.640901 0.621134 0.59549 0.629375 0.651332 0.709532 0.680552 0.668837 0.693128 0.680688 0.58774 0.571743 0.591691 0.634367 0.61124 0.609884 0.594554 0.619589 0.608023 0.673654 0.691171 0.683084 0.692007 0.651171 0.636228 0.65332 0.670648 0.65267 0.673684 0.730709 0.585946 0.572181 0.606912 0.733499 0.64044 0.680311 0.709337 0.713137 0.740279 0.724014 0.744241 0.708693 0.694145 0.699932 0.729397 0.730709 0.72415 0.710688 0.704765 0.696293 0.667734 0.667675 0.811899 0.811899 0.731578 0.717177 0.775348 0.722097 0.744241 0.75581 0.7197 0.784836 0.753972 0.744987 0.775044 0.795622 0.771304 0.798367 0.796831 0.784836 0.838962 0.838962 0.775495 0.798367 0.817679 0.76158 0.771821 0.790674 0.82543 0.833888 0.82543 0.749416 0.812997 0.739289 0.748005 0.759976 0.467442 0.414924 0.417588 0.691043 0.687084 0.681182]

# ╔═╡ 7ef14a3d-0bba-4df4-8e4b-ca93ac66e6ac
md"""
To generate a new mesh, uncomment and enable the cells below:
"""

# ╔═╡ 1087358c-ba74-4fec-aa43-f1fee0a8dd24
# ╠═╡ disabled = true
#=╠═╡
# using Triangulate
  ╠═╡ =#

# ╔═╡ 2882ebb8-a8d3-40b7-9667-2cd1f640c686
# ╠═╡ disabled = true
#=╠═╡
function example_domain_bcdt_area(; maxarea = 0.05)
    triin = Triangulate.TriangulateIO()
    triin.pointlist = Matrix{Cdouble}([0.0 0.0; 1.0 0.0; 0.5 cos(π/6)]')
    triin.segmentlist = Matrix{Cint}([1 2; 2 3; 3 1]')
    triin.segmentmarkerlist = Vector{Int32}([1, 2, 3])
    area = string(maxarea)
    (triout, vorout) = Triangulate.triangulate("pa$(area)DQ", triin)

	triout
end
  ╠═╡ =#

# ╔═╡ 9f8eaf46-3101-4f4d-bcf5-f4e930847537
# ╠═╡ disabled = true
#=╠═╡
z = example_domain_bcdt_area(maxarea=0.0003)
  ╠═╡ =#

# ╔═╡ 0c7cc866-c132-4304-af1e-128197580de7
# ╠═╡ disabled = true
#=╠═╡
tris = z.pointlist
  ╠═╡ =#

# ╔═╡ f07d54b6-f3e9-4e9c-95c3-e13a373e314f
# ╠═╡ disabled = true
#=╠═╡
pts = z.trianglelist
  ╠═╡ =#

# ╔═╡ 51b3fecd-f48b-4ff2-af5d-0909c761792a
md"""
To store a new triangle mesh, use the "Copy Output" function (in the cell context menu) to get code for the mesh below. Paste it in the hardcoded definition above.
"""

# ╔═╡ d887cfe6-f7a8-4765-a554-288ffb5122ba
#=╠═╡
repr(z.pointlist; context=IOContext(stdout, :compact=>true)) |> Text
  ╠═╡ =#

# ╔═╡ 7c980558-f97e-4581-bdb5-92faa0c1decf
#=╠═╡
repr(z.trianglelist; context=IOContext(stdout, :compact=>true)) |> Text
  ╠═╡ =#

# ╔═╡ 32f38d02-7579-4bcf-ad76-49fbff60cad9
md"""
### Barycentric coordinate conversion
Convert 2D screen coordinates to 3D triangle coordinates (in the domain of the dirichlet pdf).
"""

# ╔═╡ cb90b58b-6b54-4dea-bf08-422b71c22a5c
function tweak(p)
	# to fix numerical instabilities
	q = p .+ 0.001 * one(eltype(p))
	q ./ sum(q)
end

# ╔═╡ ddc83c32-3b8a-4ec1-88f0-b4ad5fc93e2d
function cartesian_to_barycentric(p)
	# https://en.wikipedia.org/wiki/Barycentric_coordinate_system
	T = [
		.5 - 0       1 - 0
		cos(π/6) - 0 0 - 0
	]
	
	result = T \ p

	parameter = [result..., (1-sum(result))]
	# slightly change the value to fix numerical instabilities
	tweak(parameter)
end

# ╔═╡ 3a633562-2c17-4b30-be76-792ed67d61cd
pts_barycentric = tweak.(cartesian_to_barycentric.(eachcol(pts)))

# ╔═╡ 68a2aa44-1c8c-42d5-b7f9-b96aee922480
function plot_dirichlet_pdf(α)
	vals = [pdf(Dirichlet(α), p) for p in pts_barycentric]
	mesh_heatmap_superfast(pts, tris, vals; )
end

# ╔═╡ 886f8df5-e2cc-417b-bfbc-068a37f2de04
plot_dirichlet_pdf(α₀)

# ╔═╡ 28d63bb9-af87-4b97-a650-d58d95c4b74c
let
	# Extract parameters 
	αN = params(resultsDC.posteriors[:θ]) |> only
	
	# Generate filled contour plot
	var"show prior" ? 
		TwoColumn(
			header("Prior", plot_dirichlet_pdf(α₀)),
			header("Posterior", plot_dirichlet_pdf(αN))
		) : 
			header("Posterior", plot_dirichlet_pdf(αN))
		
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractPlutoDingetjes = "6e696c72-6542-2067-7265-42206c756150"
BmlipTeachingTools = "656a7065-6f73-6c65-7465-6e646e617262"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Memoization = "6fafb56a-5788-4b4e-91ca-c0cea6611c73"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
RxInfer = "86711068-29c9-4ff7-b620-ae75d7495b3d"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[compat]
AbstractPlutoDingetjes = "~1.4.0"
BmlipTeachingTools = "~1.4.1"
Distributions = "~0.25.127"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
Memoization = "~0.2.2"
Plots = "~1.41.6"
RxInfer = "~5.5.0"
SpecialFunctions = "~2.8.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "a537fdfe6cd417df17dd5017a8370c481618c433"

[[deps.ADTypes]]
git-tree-sha1 = "d9aaef7c63466eee4de23b4d9dad03629df54bea"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.22.1"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "daa72978cd7a624246e894a4f4f067706d4e17e2"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.0"
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
git-tree-sha1 = "3d0cabd25fab32390e3bcb82cd67e700aebd9816"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.25.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
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
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
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
git-tree-sha1 = "e0b47732a192dd59b9d079a06d04235e2f833963"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.12.2"
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
git-tree-sha1 = "e644fb61dcc7b0df1be931230d2b9bc3a96d912f"
uuid = "b4ee3484-f114-42fe-b91c-797d54a0c67e"
version = "1.5.9"
weakdeps = ["FastCholesky"]

    [deps.BayesBase.extensions]
    FastCholeskyExt = "FastCholesky"

[[deps.BitFlags]]
git-tree-sha1 = "bbe1079eecf9c9fbb52765193ad2bae27ae09bc8"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.10"

[[deps.BitSetTuples]]
deps = ["TupleTools"]
git-tree-sha1 = "aa19428fb6ad21db22f8568f068de4f443d3bacc"
uuid = "0f2f92aa-23a3-4d05-b791-88071d064721"
version = "1.1.5"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "0f606a9894e2bcda541ceb82a91a13c5d450ed97"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.9.3"

    [deps.BlockArrays.extensions]
    BlockArraysAdaptExt = "Adapt"
    BlockArraysBandedMatricesExt = "BandedMatrices"

    [deps.BlockArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"

[[deps.BmlipTeachingTools]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "PlutoTeachingTools", "PlutoUI", "Reexport"]
git-tree-sha1 = "721865ca80c702e053b7d3958c5de5295ad84eca"
uuid = "656a7065-6f73-6c65-7465-6e646e617262"
version = "1.4.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "1a3ad7e16a321667698a19e77362b35a1e94c544"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.1"

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

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

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
git-tree-sha1 = "c761b00e7755700f9cdf5b02039939d1359330e1"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.1.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "f1697a56da59e8a2cefcbbfe71c13354a6f18c61"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.1.0"

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
git-tree-sha1 = "21d088c496ea22914fe80906eb5bce65755e5ec8"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.1"

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

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6fb53a69613a0b2b68a0d12671717d307ab8b24e"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.5"

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
git-tree-sha1 = "a55766a9c8f66cf19ffcdbdb1444e249bb4ace33"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.6"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "79a2aca180a85c690c58a020d47b426954b590f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.16.0"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "2147a95a217cc8a78ec96ee03581adf129468e49"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.18"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = ["GPUArraysCore", "Adapt"]
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceHyperHessiansExt = "HyperHessians"
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
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
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
    HyperHessians = "06b494a0-c8e0-40cc-ad32-d99506a00a6c"
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
git-tree-sha1 = "3c8a0a9a6d4a10bdfb6b751bd2b6051ed3e25fd4"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.127"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.DomainIntegrals]]
deps = ["CompositeTypes", "DomainSets", "FastGaussQuadrature", "FunctionMaps", "GaussQuadrature", "HCubature", "IntervalSets", "LinearAlgebra", "QuadGK", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "e90efe58cabfd58984ba5ba8edaf440251233384"
uuid = "cc6bae93-f070-4015-88fd-838f9505a86c"
version = "0.5.4"

[[deps.DomainSets]]
deps = ["CompositeTypes", "FunctionMaps", "IntervalSets", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4599e0cd684f3ff6cbbab73c77553a3d01a8d74d"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.7.18"

    [deps.DomainSets.extensions]
    DomainSetsMakieExt = "Makie"
    DomainSetsRandomExt = "Random"

    [deps.DomainSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

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
git-tree-sha1 = "c307cd83373868391f3ac30b41530bc5d5d05d08"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.1+0"

[[deps.ExponentialFamily]]
deps = ["BayesBase", "BlockArrays", "Distributions", "DomainSets", "FastCholesky", "FillArrays", "ForwardDiff", "HCubature", "HypergeometricFunctions", "IntervalSets", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "PositiveFactorizations", "Random", "SparseArrays", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "TinyHugeNumbers"]
git-tree-sha1 = "a6b9907170f8e7958e01ff0d31b6838022f4db2f"
uuid = "62312e5e-252a-4322-ace9-a5f4bf9b357b"
version = "2.5.0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

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
git-tree-sha1 = "4916117dd032ec5959b7633aedbbac408ca5ddeb"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.3.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "8e9c059d6857607253e837730dbf780b6b151acd"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.19.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "2e5742cda07276e1834281ada4cc71b6bfc87586"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.31.1"

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
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "2c5d0b0e12088cde2cf84afb2784415b1ea3dfee"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.4.1"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.FunctionMaps]]
deps = ["CompositeTypes", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "31bd99a57edf98990d1c21486032963955450e8d"
uuid = "a85aefff-f8ca-4649-a888-c8e5398bc76c"
version = "0.1.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "9e0fb9e54594c47f278d75063980e43066e26e20"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "f954322d5de03ec630d177cda203dcd92b6be399"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.26"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "6fada551286ab6ea4ca1628cb2de9f166a2ec966"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.26+0"

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

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.GraphPPL]]
deps = ["BitSetTuples", "DataStructures", "Dictionaries", "MacroTools", "MetaGraphsNext", "NamedTupleTools", "Static", "StaticArrays", "TupleTools", "Unrolled"]
git-tree-sha1 = "ea4b12cf4e8b76e4c075898724b37915808cace1"
uuid = "b3f8163a-e979-4e85-b43e-1f63d8c8b42c"
version = "4.8.0"

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
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Inflate", "LinearAlgebra", "Random", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "7eb45fe833a5b7c51cf6d89c5a841d5967e44be3"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.14.0"

    [deps.Graphs.extensions]
    GraphsSharedArraysExt = "SharedArrays"

    [deps.Graphs.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
    SharedArrays = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HCubature]]
deps = ["Combinatorics", "DataStructures", "LinearAlgebra", "QuadGK", "StaticArrays"]
git-tree-sha1 = "8ee627fb73ecba0b5254158b04d4745611b404a1"
uuid = "19dc6840-f33b-545b-b366-655c7e3ffd49"
version = "1.8.0"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

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
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

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
git-tree-sha1 = "79d6bd28c8d9bccc2229784f1bd637689b256377"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.14"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JLD2]]
deps = ["ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "941f87a0ae1b14d1ac2fa57245425b23a9d7a516"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.4"

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

    [deps.JLD2.weakdeps]
    UnPack = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c89d196f5ffb64bfbf80985b699ea913b0d2c211"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

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
git-tree-sha1 = "405e938379dbbdb24c0d93bd5b9600ecce03d11c"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "2.9.10"

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
version = "8.15.0+0"

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
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Printf"]
git-tree-sha1 = "cef1ba655e8c1f65af9d96c4fffe18bf1a3a3291"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.7.1"

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
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Memoization]]
deps = ["MacroTools"]
git-tree-sha1 = "7dbf904fa6c4447bd1f1d316886bfbe29feacf45"
uuid = "6fafb56a-5788-4b4e-91ca-c0cea6611c73"
version = "0.2.2"

[[deps.MetaGraphsNext]]
deps = ["Graphs", "JLD2", "SimpleTraits"]
git-tree-sha1 = "1983a06112db00c54e171f5cff8788c8efbdae85"
uuid = "fa8bd995-216d-47f1-8a91-f3b68fbeb377"
version = "0.7.5"

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
version = "2025.11.4"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "FiniteDiff", "LinearAlgebra"]
git-tree-sha1 = "b3f76b463c7998473062992b246045e6961a074e"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "8.0.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NamedTupleTools]]
git-tree-sha1 = "90914795fc59df44120fe3fff6742bb0d7adb1d0"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.14.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["ADTypes", "EnumX", "FillArrays", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "Statistics"]
git-tree-sha1 = "6fe140aab6c042a73c9d5dc280b87b32eee44f9f"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "2.2.1"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "a680e58292816791211582a4d8e44353835e991f"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.38"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "32a4e09c5f29402573d673901778a0e03b0807b9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.6"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
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
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "cb20a4eacda080e517e4deb9cfb6c7c518131265"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.6"

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

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "90b41ced6bacd8c01bd05da8aed35c5458891749"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

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
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

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
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.ReactiveMP]]
deps = ["BayesBase", "DiffResults", "Distributions", "DomainIntegrals", "DomainSets", "ExponentialFamily", "FastCholesky", "FastGaussQuadrature", "FixedArguments", "ForwardDiff", "HCubature", "LazyArrays", "LinearAlgebra", "MacroTools", "MatrixCorrectionTools", "Optim", "PolyaGammaHybridSamplers", "PositiveFactorizations", "Random", "Rocket", "SpecialFunctions", "StatsBase", "StatsFuns", "TinyHugeNumbers", "Tullio", "TupleTools", "UUIDs"]
git-tree-sha1 = "e495df252e8564d65b215a2005684fe3a0deda93"
uuid = "a194aa59-28ba-4574-a09c-4a745416d6e3"
version = "6.3.0"

    [deps.ReactiveMP.extensions]
    ReactiveMPOptimisersExt = "Optimisers"
    ReactiveMPProjectionExt = "ExponentialFamilyProjection"

    [deps.ReactiveMP.weakdeps]
    ExponentialFamilyProjection = "17f509fa-9a96-44ba-99b2-1c5f01f0931b"
    Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"

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

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Rocket]]
deps = ["DataStructures", "Sockets", "Unrolled"]
git-tree-sha1 = "12c8d9bb85176acf23057b679a894ea59eb922c8"
uuid = "df971d30-c9d6-4b37-b8ff-e965b2cb3a40"
version = "1.9.0"

    [deps.Rocket.extensions]
    RocketObservablesExt = "Observables"

    [deps.Rocket.weakdeps]
    Observables = "510215fc-4207-5dde-b226-833fc4488ee2"

[[deps.RxInfer]]
deps = ["Base64", "BayesBase", "DataStructures", "Dates", "Distributions", "DomainSets", "ExponentialFamily", "FastCholesky", "GraphPPL", "HTTP", "JSON", "LinearAlgebra", "Logging", "MacroTools", "Optim", "Preferences", "ProgressMeter", "Random", "ReactiveMP", "Reexport", "Rocket", "SparseArrays", "Static", "Statistics", "TupleTools", "UUIDs"]
git-tree-sha1 = "4098a13ad0db656071d6e71ff085cbec69300412"
uuid = "86711068-29c9-4ff7-b620-ae75d7495b3d"
version = "5.5.0"

    [deps.RxInfer.extensions]
    PrettyTablesExt = "PrettyTables"
    ProjectionExt = "ExponentialFamilyProjection"
    TensorBoardLoggerExt = "TensorBoardLogger"

    [deps.RxInfer.weakdeps]
    ExponentialFamilyProjection = "17f509fa-9a96-44ba-99b2-1c5f01f0931b"
    PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
    TensorBoardLogger = "899adc3e-224a-11e9-021f-63837185c80f"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLPublic]]
git-tree-sha1 = "912a1a941afb2a232183a485d86a437a25f88520"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.2.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

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
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "6547cbdd8ce32efba0d21c5a40fa96d1a3548f9f"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.8.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "cbc2fed5b692d12e199397471ba6c5988f69b629"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.4.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

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
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "e4d7a1a0edc20af42689ea6f4f3587a2175d50ee"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.12"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

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

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

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
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.Tullio]]
deps = ["DiffRules", "LinearAlgebra", "Requires"]
git-tree-sha1 = "de0febfe1243e89f352abd4ca0e9de6c8e6190c5"
uuid = "bc48ee85-29a4-5162-ae0b-a64e1601d4bc"
version = "0.3.9"

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

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unrolled]]
deps = ["MacroTools"]
git-tree-sha1 = "6cc9d682755680e0f0be87c56392b7651efc2c7b"
uuid = "9602ed7d-8fef-5bc8-8597-8f21381861e8"
version = "0.1.5"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

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
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

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
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

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
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

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
git-tree-sha1 = "ed349d26affcacafbc7fc2941ace1fb98f71e715"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+1"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

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
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

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
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

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
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

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
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╟─71fa9b4b-ca28-4517-b6d4-0135cce1b800
# ╟─779f3ea6-36dd-11f0-0323-613cd06c6f3c
# ╟─feef03fb-f9c5-4a76-8727-2d690c4dac1f
# ╠═10bdf484-848d-44a1-af9b-64c0c11395f7
# ╟─34a1be54-4641-455d-ab3e-8d4d08aaf966
# ╟─dfa63b82-3593-45b1-b13f-c47274ce724c
# ╟─de99bbcb-ce5b-44f1-85ef-ad26a9238509
# ╟─4dfba22d-1d61-4325-90a0-52afed1787ee
# ╟─f05721c1-e085-4c7d-92c8-292aa17c8040
# ╠═35d8e213-09ef-4419-a5b8-ac9a1b4e3c1f
# ╟─5cf8f6fe-31cc-4226-89dd-6e8fe6527442
# ╟─6dae4878-8765-4bae-b8df-399d43cdf53c
# ╟─0cb98ad6-12b5-4644-810a-eaf0a8436084
# ╟─ffc0d2c1-cbb6-496b-91c1-39dc505a8df1
# ╟─c6d8d802-ee4b-4eac-84cb-a1769d2ab175
# ╟─928f748d-60eb-45bb-bc8d-0a99b3f11b01
# ╟─0735f44e-2b14-41f8-a7fd-ab18b20a6d87
# ╟─90210954-a4b0-4e98-b00e-667479bfe60d
# ╟─5163221b-9572-4cfe-950a-ac435ba57c19
# ╟─2cd6a20d-80bb-4aaf-8017-c657ed1315bc
# ╟─388981ac-ee33-46b1-8162-1f51e479b5d7
# ╟─613e50b4-535e-4420-8ecd-c1ddc89c262e
# ╟─bc347329-d4f1-4999-89ce-fd50e4d890e1
# ╠═5411235f-e823-4db4-b4c4-1532d4bc8927
# ╟─307acd4b-72a7-4aa5-add8-8269708adaa9
# ╟─c72aabb2-779f-421c-b828-277c3b4d1233
# ╟─b34f2fce-3944-4c9c-bd8b-f48aca983979
# ╟─6671664b-1efc-437e-8690-bd8d4c9a7277
# ╟─4fd197db-cac8-4a39-a9eb-9399f945a4e1
# ╟─a8da905e-16df-4fc0-a4ef-ff10cbd60a85
# ╟─9d9542b3-cadf-4ada-95a9-d1873061df8d
# ╟─aa1528c5-1c46-4c45-a8e8-8a899e8124c8
# ╟─4a0d19ef-1756-4d14-a787-0b31763577f5
# ╟─0a536620-fc4a-41c6-8f34-7015b400c910
# ╠═4973ed66-7e7f-402b-b636-0a7f8771e743
# ╟─ff49a7f8-7f83-4fe0-a470-0f2c265fc619
# ╟─617339f1-a22a-40ff-aa17-6258cdb8ef6c
# ╟─e73893ad-d6d4-496d-ae39-20650ee175ab
# ╟─861a82c4-87ea-4a01-b511-9847786f347f
# ╟─e2aeb352-6a50-45a0-89f9-c68e287245ca
# ╟─ec2fb935-0e29-4188-b507-b7009452a99e
# ╠═886f8df5-e2cc-417b-bfbc-068a37f2de04
# ╟─8442878d-068c-4cee-b51a-782190703f58
# ╟─4df18a8f-7483-45db-9988-8579dc4b9103
# ╟─41e0eb82-bf76-49e0-8742-fcb30e675905
# ╠═2f0932a8-7f5e-41f8-aea6-882a147aadf3
# ╟─68a37b6f-6bad-4752-85c4-ad929fc59e16
# ╟─c600e2cc-1094-4177-a533-26cba3ff4eaa
# ╟─28d63bb9-af87-4b97-a650-d58d95c4b74c
# ╟─cea0cb2a-8e11-443f-8e2f-c49b09d9fdf5
# ╟─ab3fb0b2-242f-46eb-93e9-6f70f18df707
# ╟─13a5594d-b675-43c5-ad7c-1b9e0e0a9572
# ╟─ac1321cb-764a-48d3-8ac4-9cedfe34d370
# ╠═9c1da54d-edad-499a-91ae-88483b5d5a72
# ╟─174aedea-93e2-4e21-aac1-21a4c27cede9
# ╟─abe4105c-c42b-4d97-bc15-ef9741c23fcf
# ╟─73025b50-feac-4723-8a65-024689232112
# ╠═0af6ca08-cece-460b-9166-5b8168fb9b73
# ╟─caf8ede0-a410-4b4a-81dd-4342b8a2385a
# ╟─75abee86-a43c-4a53-8edb-cf943f4e570c
# ╟─71b7e9cb-a6ce-4982-b00c-b98dcd08888f
# ╟─7cc095a9-fbf3-4402-bba6-f25fdb1347b1
# ╟─27991fcd-6754-42fe-8e47-d52187065afb
# ╟─86f213ce-b2b2-48ce-bd94-ccd863e3e2d7
# ╠═12d5f6d1-f214-448e-9c2e-da691b997d60
# ╟─67b0f499-9492-48fb-9540-b6600abffc3a
# ╟─017e4cec-544f-46aa-b612-1b1ba83bb792
# ╟─66b2711a-f4b9-43fe-8f0f-112d1365bbbe
# ╠═0bb66b2a-5d41-4f6b-b245-0360274b9296
# ╟─df362c96-d688-42c1-acfc-c270c33d5db7
# ╟─5fa37c24-c90e-4e21-921e-03422943f225
# ╠═6d7f9ad2-8dc4-450d-afb6-c55deaeed564
# ╟─5c315bdb-7375-459e-a9e2-0d180271ed78
# ╟─ff70f05f-7259-4d56-9a33-19ebd3483c63
# ╟─7e6ac0f6-710e-4388-acc2-b2b255a961ca
# ╟─02156dca-2fcb-4654-80a0-3666a7807ea3
# ╟─5ef49dcd-16ff-4183-a28b-061572d98b98
# ╟─d8218c14-e005-4e34-a468-44ebb790ef11
# ╟─7bb464b2-0cd2-484e-aacd-f01ff92fc30d
# ╟─948a442d-64c3-4a8b-ad9f-b98f5db82438
# ╟─f1395ab5-637c-4894-999c-4f45489cfccb
# ╟─02e2ddee-afb9-43a5-98b1-420e1e6169d8
# ╠═35de8df9-ab15-4ea7-a099-6f6523635267
# ╠═c2dfe525-508e-4af7-9224-7f56051c651d
# ╠═a3eccf21-dab7-48ae-8091-e3f4569a2a04
# ╟─acf9c5bd-e9a6-44c2-9baf-0d1a0b41c708
# ╟─7db33856-3f31-481b-9622-d673ae16c5e7
# ╟─68a2aa44-1c8c-42d5-b7f9-b96aee922480
# ╟─6f170031-948a-4a02-8912-913e531ac1a1
# ╟─57aa818c-eb7d-44c4-a94c-a61b59a98192
# ╟─7818342e-0712-4e54-9a8d-6c9e49b44237
# ╟─7ef14a3d-0bba-4df4-8e4b-ca93ac66e6ac
# ╠═1087358c-ba74-4fec-aa43-f1fee0a8dd24
# ╠═2882ebb8-a8d3-40b7-9667-2cd1f640c686
# ╠═9f8eaf46-3101-4f4d-bcf5-f4e930847537
# ╠═0c7cc866-c132-4304-af1e-128197580de7
# ╠═f07d54b6-f3e9-4e9c-95c3-e13a373e314f
# ╟─51b3fecd-f48b-4ff2-af5d-0909c761792a
# ╠═d887cfe6-f7a8-4765-a554-288ffb5122ba
# ╠═7c980558-f97e-4581-bdb5-92faa0c1decf
# ╟─32f38d02-7579-4bcf-ad76-49fbff60cad9
# ╟─ddc83c32-3b8a-4ec1-88f0-b4ad5fc93e2d
# ╟─cb90b58b-6b54-4dea-bf08-422b71c22a5c
# ╠═3a633562-2c17-4b30-be76-792ed67d61cd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
