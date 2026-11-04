### A Pluto.jl notebook ###
# v1.0.3

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

# ╔═╡ 109f3a18-7a85-5225-abb3-036a7d6f5319
begin
	using BmlipTeachingTools
	using Random
	using LinearAlgebra
	using Statistics
	using Distributions
	using RxInfer
	using Plots
	Plots.default(label="", grid=false, linewidth=3, margin=10Plots.pt, size=(700,300))
end

# ╔═╡ f4618cf4-1940-520c-88d8-2a1d79f76127
title("Probabilistic Programming 5: Bayesian control & active inference")

# ╔═╡ 490f293f-4996-5b82-9524-4ec329e9ef93
md"""

#### Goal
  - Learn how to formulate control as Bayesian inference, using goal priors over future states.
  - Learn how to run an agent in a closed loop with its environment: plan, act, perceive.
  - Learn how the expected free energy combines goal-seeking with information-seeking, and why that matters for an agent that has to learn while it acts.

#### Materials
  - Mandatory
    - This notebook
    - Lecture notes on [intelligent agents and active inference](https://bmlip.github.io/course/lectures/Intelligent%20Agents%20and%20Active%20Inference.html)
    - Lecture notes on [dynamic models](https://bmlip.github.io/course/lectures/Dynamic%20Models.html)
  - Optional
    - Levine (2018), [Reinforcement learning and control as probabilistic inference: tutorial and review](https://arxiv.org/abs/1805.00909)
    - Friston et al. (2015), [Active inference and epistemic value](https://doi.org/10.1080/17588928.2015.1020053)
    - Kouw (2023), [Information-seeking polynomial NARX model-predictive control through expected free energy minimization](https://arxiv.org/abs/2312.15046)
    - [Differences between Julia and Matlab / Python](https://docs.julialang.org/en/v1/manual/noteworthy-differences/index.html).
"""

# ╔═╡ 5eb11e24-f3e7-53f0-be55-cc187d2c2f99
TableOfContents()

# ╔═╡ 08a90d11-2a35-560e-980a-5981a0ce119b
md"""
In the previous sessions, the data was handed to us. In this session, our models get a body: we build **agents** that choose actions, and these actions change the data they will observe next. We will see that choosing an action can be cast as inference, just like everything else in this course.

We will build two agents:
1. a **Bayesian controller**, which infers the forces that bring a robot to its charging dock (_planning as inference_), and
2. an **active inference agent**, which has to figure out how its own motor works while it tries to reach the dock, and which seeks out the information it needs to do so.
"""

# ╔═╡ 1dcf887d-bcf8-5601-83b7-216b388a519d
NotebookCard("https://bmlip.github.io/course/lectures/Intelligent%20Agents%20and%20Active%20Inference.html")

# ╔═╡ 5b157b82-2f20-5344-a985-847ba5da3178
challenge_statement("Parking a robot"; header_level=1, big=true)

# ╔═╡ f803e8b1-8fed-5bf5-98a7-1e7988bd8905
md"""
A small robot drives along a straight rail in a warehouse. At the end of its shift, it has to park at its charging dock, 5 meters from where it is now. The robot can push itself forward or backward with its motor, and it measures its position with a noisy sensor.

We model the robot as a mass ``m`` with friction coefficient ``c``, driven by a motor force ``u`` (in Newtons):

```math
m \ddot{s} = - c \dot{s} + u \, ,
```

where ``s`` is the robot's position. With the state ``x = [s \ \ \dot{s}]^{\top}`` (position and velocity) and a simple discretization with time step ``\Delta t``, we obtain a linear Gaussian state-space model, just like in [Probabilistic Programming 4](https://bmlip.github.io/course/probprog/PP4%20-%20Bayesian%20filtering%20and%20smoothing.html):

```math
\begin{aligned}
x_k &= A x_{k-1} + B u_k + w_k \, , \qquad w_k \sim \mathcal{N}(0, Q) \\
y_k &= C^{\top} x_k + r_k \, , \qquad \ \ \ r_k \sim \mathcal{N}(0, \sigma_y^2) \, ,
\end{aligned}
```

with

```math
A = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 - \frac{c}{m} \Delta t \end{bmatrix} , \qquad B = \begin{bmatrix} 0 \\ \frac{\Delta t}{m} \end{bmatrix} , \qquad C = \begin{bmatrix} 1 \\ 0 \end{bmatrix} .
```

The robot's task is to reach the state ``x_* = [5 \ \ 0]^{\top}``: parked at the dock and standing still.
"""

# ╔═╡ bf6d4247-0ac3-59d6-9b2a-b1020d688fb1
begin
	# Physical parameters
	Δt = 0.2           # time step (s)
	mass = 1.0         # mass of the robot (kg)
	friction = 0.5     # friction coefficient (N⋅s/m)

	# State transition, state x = [position, velocity]
	A = [1.0  Δt;
	     0.0  1.0 - friction/mass*Δt]

	# Control matrix (2×1): the force is a one-dimensional input
	B = reshape([0.0, Δt/mass], 2, 1)

	# Emission vector: the sensor measures position
	C = [1.0, 0.0]

	# Noise
	Q = diagm([1e-4, 1e-4])   # process noise covariance
	σ_y = 0.2                 # standard deviation of the position sensor (m)

	# Goal: parked at the dock, standing still
	dock = 5.0
	x_dock = [dock, 0.0]

	# Length of a simulation (number of time steps)
	N_steps = 50
end

# ╔═╡ 21a58e3e-a745-5561-bb57-b7be6efb5fa7
md"""
The real robot is simulated by a `RobotOnRail` (see the [Appendix](#Appendix)). The function `apply_force!(robot, u)` applies the force `u` during one time step, and returns a noisy measurement of the robot's new position. The robot's true state, `robot.state`, is hidden from the agent: we only use it to evaluate how well the agent does.
"""

# ╔═╡ 76066720-ec4a-5b75-9f1a-8009865bfa99
md"""
## Perception

The agent maintains a belief ``q(x_t) = \mathcal{N}(x_t \mid m_t, S_t)`` about the robot's current state. After applying force ``u_t`` and receiving measurement ``y_t``, it updates this belief with Bayes' rule:

```math
q(x_t) \propto \underbrace{p(y_t \mid x_t)}_{\text{likelihood}} \int \underbrace{p(x_t \mid x_{t-1}, u_t)}_{\text{state transition}} \, \underbrace{q(x_{t-1})}_{\text{previous belief}} \, \mathrm{d}x_{t-1} \, .
```

You will recognize this as one step of a Kalman filter (see the [lecture on dynamic models](https://bmlip.github.io/course/lectures/Dynamic%20Models.html)). In RxInfer, we only need to write down the model for a single time step:
"""

# ╔═╡ dcabac88-b287-575f-871b-230829feff3e
@model function robot_filter(y, m_prev, S_prev, u_prev, A, B, C, Q, σ2_y)
	"One step of Bayesian filtering: update the belief after acting and observing"

	# Belief about the previous state
	x_prev ~ MvNormalMeanCovariance(m_prev, S_prev)

	# State transition, given the force that was applied
	x ~ MvNormalMeanCovariance(A * x_prev + B * u_prev, Q)

	# Position measurement
	y ~ NormalMeanVariance(dot(C, x), σ2_y)
end

# ╔═╡ 6b21a578-986f-54b4-aed2-e7c6558a135c
md"""
We wrap one perception step in a function, which returns the mean and covariance matrix of the updated belief:
"""

# ╔═╡ d0df0199-b1df-51e9-840a-524ab8a3290a
function perceive(m_prev, S_prev, u_applied, y)
	result = infer(
		model = robot_filter(m_prev=m_prev, S_prev=S_prev, u_prev=u_applied, A=A, B=B, C=C, Q=Q, σ2_y=σ_y^2),
		data  = (y = y,),
	)
	m, S = mean_cov(result.posteriors[:x])

	# Return a symmetric covariance matrix (to avoid round-off problems)
	return m, (S + S') / 2
end

# ╔═╡ 8893d502-0c43-593e-99b9-c7c66fd7598c
md"""
## Planning as inference

How should the robot choose its forces? In Bayesian control, we do not write down a cost function and hand it to an optimizer. Instead, we extend the agent's generative model into the future and add two kinds of prior beliefs:
- a **control prior** ``p(u_k) = \mathcal{N}(u_k \mid 0, \sigma_u^2)``, which states that small forces are more plausible than large ones (large forces cost energy and wear out the motor), and
- a **goal prior** ``\hat{p}(x_k) = \mathcal{N}(x_k \mid x_*, \Sigma_*)``, which states where the agent expects to be in the future: at the dock.

For a planning horizon of ``T`` steps, counting from the current time (``k=0``), the agent's model of the future is

```math
\underbrace{q(x_0)}_{\text{current belief}} \prod_{k=1}^{T} \underbrace{p(x_k \mid x_{k-1}, u_k)}_{\text{state transition}} \, \underbrace{p(u_k)}_{\text{control prior}} \, \underbrace{\hat{p}(x_k)}_{\text{goal prior}} \, .
```

Inference in this model yields a posterior distribution over the future forces ``u_{1:T}``: the forces that are consistent with both the robot's dynamics and its goals. The agent applies the most probable first force, observes the result, updates its belief, and plans again. This is known as _receding horizon_ control, the same idea that underlies model predictive control.
"""

# ╔═╡ f1cfa47a-ae48-5bb1-86f0-8d49578f825a
md"""
!!! info "Goal priors as virtual observations"
	A Gaussian density is symmetric in its argument and its mean: ``\mathcal{N}(x_k \mid x_*, \Sigma_*) = \mathcal{N}(x_* \mid x_k, \Sigma_*)``. So, a goal prior on ``x_k`` is equivalent to pretending that we will *observe* the goal state ``x_*`` at time ``k``, with "measurement noise" ``\Sigma_*``. In RxInfer, we can therefore add the goal prior as a likelihood for a variable `x_goal[k]`, to which we feed the value ``x_*`` as data.
"""

# ╔═╡ ed2e3c11-c405-598f-84d1-9f5b168f6d1e
@model function robot_planner(x_goal, m_now, S_now, A, B, Q, m_u, S_u, S_goal, T)
	"Planning as inference: which forces bring the robot to its goal?"

	# Belief about the current state
	x_0 ~ MvNormalMeanCovariance(m_now, S_now)

	x_kmin1 = x_0
	for k in 1:T

		# Control prior
		u[k] ~ MvNormalMeanCovariance(m_u, S_u)

		# Predicted state transition
		x[k] ~ MvNormalMeanCovariance(A * x_kmin1 + B * u[k], Q)

		# Goal prior, written as a virtual observation of the goal state
		x_goal[k] ~ MvNormalMeanCovariance(x[k], S_goal)

		# Update recursive aux
		x_kmin1 = x[k]
	end
end

# ╔═╡ 58819fa7-b1d9-54f8-8b55-d99cd9aaaa49
md"""
Note that the force ``u_k`` is a 1-dimensional vector here, so that the ``2 \times 1`` matrix ``B`` maps it into the state space. The function below sets up the priors and calls `infer`, starting from the agent's current belief ``q(x_0) = \mathcal{N}(m_t, S_t)``:
"""

# ╔═╡ b39ae303-a54a-5b39-8594-865102bf88c4
md"""
### What does the robot plan?

At the start, the robot believes it is standing still at position 0, but it is not entirely sure about that:
"""

# ╔═╡ 4689e861-d70d-5d3e-82e6-fc5255fbf4f0
begin
	m_init = [0.0, 0.0]          # mean of the initial belief (position, velocity)
	S_init = diagm([1.0, 0.1])   # covariance matrix of the initial belief
end

# ╔═╡ 99966210-853f-5640-8ad5-f3f307aae6d6
md"""
Let's look at the plan the robot makes from this initial belief:
"""

# ╔═╡ 3ee7ec5c-a11c-556f-825f-34af544b461e
md"""
Let's compare both plans. With the sliders, you can change the standard deviations of the goal prior (``\Sigma_*`` is diagonal, with one standard deviation for the position and one for the velocity) and of the control prior (``\sigma_u``), as well as the planning horizon ``T``.
"""

# ╔═╡ 18b3356a-9530-5e5b-ad18-59e8c185c5f0
begin
	agent_bonds = PlutoUI.ExperimentalLayout.vbox([
		@bindname(goal_std_position, Slider([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]; default=0.5, show_value=true))
		@bindname(goal_std_velocity, Slider([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]; default=0.5, show_value=true))
		@bindname(control_std, Slider([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]; default=1.0, show_value=true))
		@bindname(horizon, Slider([2, 5, 10, 15, 20, 30, 40]; default=20, show_value=true))
	])
end

# ╔═╡ 169f73d5-55b8-5a10-9928-febbe5f8c6ea
"Infer a posterior over the next `horizon` forces, starting from the belief N(m_now, S_now)"
function plan_forces(m_now, S_now)
	return infer(
		model = robot_planner(
			m_now  = m_now,
			S_now  = S_now,
			A      = A,
			B      = B,
			Q      = Q,
			m_u    = zeros(1),
			S_u    = fill(control_std^2, 1, 1),
			S_goal = diagm([goal_std_position^2, goal_std_velocity^2]),
			T      = horizon,
		),
		data = (x_goal = [x_dock for k in 1:horizon],),
	)
end

# ╔═╡ 2613ddd8-a146-5e6b-b54c-cfaa2179c20f
first_plan = plan_forces(m_init, S_init)

# ╔═╡ 247a905f-8ad7-5bb4-b5f3-fe0e1c405cfb
md"""
That is strange: the robot hardly pushes (its first planned force is only $(round(mean(first_plan.posteriors[:u][1])[1], digits=2)) N), yet it plans to be near the dock right away. The explanation lies in the planner's posterior belief about the *current* position: ``s_0 \approx`` $(round(mean(first_plan.posteriors[:x_0])[1], digits=2)) m, even though the robot hasn't moved yet!

Inference does not distinguish between *wanting* to observe the goal and *expecting* to observe it. The goal priors act as virtual observations of the dock, and the posterior explains these observations in the cheapest way available. Pushing hard is expensive under the control prior. But the robot is not sure where it is, and it is cheaper to believe that it is *already close to the dock*. This is **wishful thinking**, a well-known pitfall of planning as inference: conditioning on the goals biases the beliefs about the present towards the goals.

The remedy is to keep perception and planning apart. Updating the belief about the current state is the job of perception, which uses actual measurements. The planner takes this belief as given: it plans from the agent's best estimate of the current state, ``m_t``, and we replace the covariance matrix ``S_t`` by a tiny one. For a linear Gaussian system with quadratic costs, this is exactly the right thing to do: by the *separation principle* of LQG control, the optimal controller acts on the estimated state as if it were the true state. (The expected free energy in Part 2 avoids wishful thinking in a different way: it compares the *predicted* states ``q(x \mid u)``, which are not conditioned on the goals, with the goal prior.)
"""

# ╔═╡ 7b20c8f9-95b3-4661-84ec-969d4a990824
"Plan from the point estimate m_now of the current state, ignoring the uncertainty S_now"
plan_from_estimate(m_now, S_now) = plan_forces(m_now, 1e-6*diageye(2))

# ╔═╡ 4db0ebda-7d89-53d9-b860-dfc528b990ab
plan_estimate = plan_from_estimate(m_init, S_init)

# ╔═╡ f29ac4aa-8122-5d8a-9b00-c59467d659e1
md"""
From its point estimate, the robot plans to push hard at first, and then lets friction (and a bit of braking) slow it down as it approaches the dock. Note that a plan is a probability distribution: the shaded areas show ``\pm 2`` standard deviations of the posterior over future positions and forces.

👉 What happens to the planned forces when you widen the goal prior? And when you narrow the control prior? What does the horizon do?
"""

# ╔═╡ 03f8e3dc-b619-52a2-a89c-549e0b6ebcf7
md"""
## Closing the loop

An agent repeats three steps, over and over:
1. **plan**: infer a posterior over future forces, starting from the point estimate of the current state, and pick the most probable first force,
2. **act**: apply this force to the robot, and receive a new measurement,
3. **perceive**: update the belief about the robot's state.

This is called the *action-perception loop*. Here is the code:
"""

# ╔═╡ e1954792-9742-57b9-90ee-6178bcf87aaf
agent_bonds

# ╔═╡ 18f16a73-8009-5da4-a5fe-7991279223ce
md"""
### What was the robot thinking?

At every time step, the robot makes a new plan for the next ``T`` steps, but it only executes the first force of that plan. Move the slider to see the plan the robot made at each moment.
"""

# ╔═╡ 1208952c-581b-5cec-b8bb-ecdb8610e607
@bindname t_view Slider(1:N_steps; default=1, show_value=true)

# ╔═╡ 3fce0f5a-bbcd-5438-a4d2-8b3ad492c72e
exercise_statement("Park the robot", header_level=2)

# ╔═╡ a8ee6a06-3b56-55f5-8f12-cfb92d44103e
md"""
The robot's motor overheats when it delivers more than 3 N. Tune the agent's priors (and horizon) such that, in the simulation above,
1. the force never exceeds 3 N in absolute value,
2. the robot never overshoots the dock by more than 25 cm, and
3. from ``t = 8`` s onwards, the robot stays within 25 cm of the dock.
"""

# ╔═╡ 355f35de-61b8-507b-8e13-8493ea0c2d98
agent_bonds

# ╔═╡ a6a8b6cc-a6fb-550b-8c25-f4a243ad0607
hide_solution(md"""
For example: a control prior with standard deviation ``\sigma_u = 0.5`` N, and a goal prior with standard deviations of 1.0 m (position) and 1.0 m/s (velocity).

A narrower control prior makes large forces implausible: it acts as a *soft* limit on the force. Wider goal priors make the robot less eager to be at the dock *right now*, so it doesn't need large forces. But if you make the control prior too narrow, or the goal priors too wide, the robot becomes too lazy to arrive in time.
""")

# ╔═╡ 7af487a5-a345-50f8-b313-8a7b472a923a
exercise_statement("A goal at the horizon"; prefix="Model ", header_level=2)

# ╔═╡ b43ecfd4-36d7-5b8e-8e22-148ec2219996
md"""
Our planner places a goal prior on *every* future state: the robot wants to be at the dock as soon as possible. Alternatively, we could only specify where the robot should be at the *end* of its planning horizon, like a deadline.

👉 Specify a planning model with a goal prior on the final state ``x_T`` only, and run it in the action-perception loop. Try a few different horizons. What happens? Can you explain it?
"""

# ╔═╡ 20ca5627-5927-53de-9537-529577e48142
hint(md"""
- Move the goal prior out of the for-loop. After the loop, `x_kmin1` refers to the final state. The data is now a single goal state: `data = (x_goal = x_dock,)`.
- `run_robot` accepts any planning function that takes the current belief `(m, S)` and returns the result of `infer`. For example: `run_robot((m, S) -> infer(...))`.
- Use `plot_simulation(...)` to visualize the result.
""")

# ╔═╡ 5e01c014-cf56-53d6-b8ca-9ba29e774800
### YOUR CODE HERE

# ╔═╡ 949750b6-27e4-5f00-b7b0-5ce9d416324c
hide_solution(md"""
```julia
@model function robot_planner_final(x_goal, m_now, S_now, A, B, Q, m_u, S_u, S_goal, T)
	x_0 ~ MvNormalMeanCovariance(m_now, S_now)
	x_kmin1 = x_0
	for k in 1:T
		u[k] ~ MvNormalMeanCovariance(m_u, S_u)
		x[k] ~ MvNormalMeanCovariance(A * x_kmin1 + B * u[k], Q)
		x_kmin1 = x[k]
	end

	# Goal prior on the final state only
	x_goal ~ MvNormalMeanCovariance(x_kmin1, S_goal)
end
```

```julia
sim_final = run_robot((m, S) -> infer(
	model = robot_planner_final(
		m_now  = m,
		S_now  = 1e-6*diageye(2),   # plan from the point estimate
		A      = A,
		B      = B,
		Q      = Q,
		m_u    = zeros(1),
		S_u    = fill(control_std^2, 1, 1),
		S_goal = diagm([goal_std_position^2, goal_std_velocity^2]),
		T      = horizon,
	),
	data = (x_goal = x_dock,),
))
```

```julia
plot_simulation(sim_final)
```

The robot *procrastinates*. At every time step, it plans to arrive at the dock ``T`` steps later, which only requires a gentle push now. But after executing that push, it makes a new plan, and the deadline moves one step further into the future. The longer the horizon, the lazier the robot. With goal priors on every future state, every step away from the dock counts, which gives the robot a sense of urgency.
""")

# ╔═╡ 160a9e12-e432-5ba0-9c3e-6998a3f2ff39
md"""
## Where are the epistemic terms?

In the lecture, an active inference agent selects policies by minimizing the expected free energy (EFE), which consists of *risk*, *ambiguity* and *novelty* terms. Our parking robot did fine without the last two. That is no coincidence: the robot knows its own dynamics (there is nothing to learn, so no novelty), and the noise of its sensor does not depend on where it is (so all policies are equally ambiguous). For this robot, minimizing the EFE yields (practically) the same forces as our Bayesian controller.

Epistemic terms start to matter when the agent's uncertainty *depends on its actions*. Our second robot is a case in point.
"""

# ╔═╡ a9567b09-6cd2-5640-925e-3f914246ead8
challenge_statement("The rewired robot"; header_level=1, big=true)

# ╔═╡ b242f2e6-8536-5be9-9443-30a8a05fd0a4
md"""
A second robot has just returned from maintenance. The technician replaced its motor driver, but forgot to write down how the motor was connected. We don't know the gain of the motor, not even its sign: the wires may have been swapped.

This robot has a built-in speed controller. The command ``u_k \in [-2, 2]`` sets its speed for one time step, and its position ``x_k`` evolves as

```math
x_k = x_{k-1} + \theta u_k + e_k \, , \qquad e_k \sim \mathcal{N}(0, \sigma_e^2) \, ,
```

where ``\theta`` is the unknown motor gain (in meters per command unit, per time step) and ``e_k`` is a small random disturbance, for example due to wheel slip. The robot measures its position very accurately, so we treat ``x_k`` as observed.

The robot starts at ``x_0 = 0`` and has to drive to its dock at ``x_* = 5``. It can only do so if it figures out how its motor works.
"""

# ╔═╡ 7aa4fb0a-4cfd-590e-8ead-869739113c94
md"""
## The agent's model

The agent's generative model consists of the dynamics above, a prior over the motor gain, a goal prior and a control prior:

```math
\begin{aligned}
p(x_k \mid x_{k-1}, u_k, \theta) &= \mathcal{N}(x_k \mid x_{k-1} + \theta u_k, \sigma_e^2) \\
p(\theta) &= \mathcal{N}(\theta \mid m_0, v_0) \\
\hat{p}(x_k) &= \mathcal{N}(x_k \mid x_*, s_*^2) \\
p(u_k) &= \mathcal{N}(u_k \mid 0, \sigma_u^2) \, .
\end{aligned}
```

With prior mean ``m_0 = 0``, the agent considers a positive and a negative motor gain equally likely. In the code, ``s_*`` is called `goal_std`, ``m_0`` is `prior_mean`, ``v_0`` is `prior_var` and ``\sigma_u`` is `σ_command`. With the sliders, you can change the goal prior and the prior mean of the agent, as well as the robot's *true* motor gain (which the agent doesn't know).
"""

# ╔═╡ 782a35af-56da-57fd-bdc3-8b04a02ca15f
begin
	σ_e = 0.05          # standard deviation of the disturbance per time step (m)
	x_start = 0.0       # start position (m)
	x_target = 5.0      # position of the dock (m)
	u_limit = 2.0       # commands are limited to [-u_limit, u_limit]
	σ_command = 2.0     # standard deviation of the control prior
	prior_var = 1.0     # prior variance of the motor gain
	N_rewired = 20      # number of time steps

	# The candidate commands that the agent considers
	actions = range(-u_limit, u_limit; length=401)
end

# ╔═╡ 10eb6485-58c3-57d9-8e88-c672f15e1266
begin
	rewired_bonds = PlutoUI.ExperimentalLayout.vbox([
		@bindname(goal_std, Slider([0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]; default=0.5, show_value=true))
		@bindname(prior_mean, Slider([-1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0]; default=0.0, show_value=true))
		@bindname(true_gain, Slider([-1.0, -0.5, -0.2, 0.2, 0.5, 1.0]; default=-0.5, show_value=true))
	])
end

# ╔═╡ 60ccc0eb-d4d4-5b88-9edc-dee0811cee3d
md"""
## Learning the motor gain

After each step, the agent observes the displacement ``\Delta x_k = x_k - x_{k-1} = \theta u_k + e_k``, and updates its belief ``q(\theta) = \mathcal{N}(\theta \mid m_\theta, v_\theta)``. This is Bayesian linear regression with a single regressor, ``u_k`` (see [Probabilistic Programming 2](https://bmlip.github.io/course/probprog/PP2%20-%20Bayesian%20regression%20and%20classification.html)), performed online: the posterior after one step is the prior for the next.
"""

# ╔═╡ e9fb117b-e5ac-510f-aa2c-cbb69ea89364
@model function motor_learning(Δx, u, m_θ, v_θ, σ2)
	"One step of online learning of the motor gain θ"

	# Current belief about the motor gain
	θ ~ NormalMeanVariance(m_θ, v_θ)

	# Displacement caused by command u
	Δx ~ NormalMeanVariance(u * θ, σ2)
end

# ╔═╡ 5fae9622-a1bd-5010-a914-de09783d39d0
"Update the belief N(m_θ, v_θ) about the motor gain, after command u caused displacement Δx"
function update_motor_belief(m_θ, v_θ, u, Δx)

	# A zero command carries no information about the motor gain
	if abs(u) < 1e-8
		return m_θ, v_θ
	end

	result = infer(
		model = motor_learning(u=u, m_θ=m_θ, v_θ=v_θ, σ2=σ_e^2),
		data  = (Δx = Δx,),
	)
	return mean_var(result.posteriors[:θ])
end

# ╔═╡ 19dba654-d3ec-50bf-8fe6-d4330b080d1e
md"""
## Predicting the consequences of an action

To choose a command, the agent predicts where it will end up. Integrating over its uncertainty about the motor gain gives the predictive distribution

```math
q(x_{t+1} \mid u) = \int p(x_{t+1} \mid x_t, u, \theta) \, q(\theta) \, \mathrm{d}\theta = \mathcal{N}\big(x_{t+1} \mid x_t + m_\theta u, \ \sigma_e^2 + v_\theta u^2 \big) \, .
```

Note that the predicted variance grows with ``|u|``: the bigger the command, the more the outcome depends on the unknown motor gain.
"""

# ╔═╡ 7a69a7e1-d1be-587a-8c09-cfddc02a60fa
"Predictive distribution of the next position, given command u: returns (mean, variance)"
function predict_position(u, x, m_θ, v_θ)
	return x + m_θ*u, σ_e^2 + v_θ*u^2
end

# ╔═╡ 5189d002-4858-5d54-b1d7-24bed3c32df1
md"""
## Two ways to choose an action

Both agents look one step ahead. They consider 401 candidate commands between ``-2`` and ``2``, score each command, and execute the best one.

#### Agent 1: decision-theoretic

The first agent follows classical Bayesian decision theory: it minimizes its *expected cost*. The cost of ending up at position ``x`` is the surprise under its goal prior, ``-\log \hat{p}(x) = \frac{(x - x_*)^2}{2 s_*^2} + \text{const.}``, and the cost of a command is ``P(u) = -\log p(u)``:

```math
J(u) = \underbrace{\mathbb{E}_{q(x_{t+1} \mid u)}\big[ -\log \hat{p}(x_{t+1}) \big]}_{\text{expected cost (cross-entropy)}} + P(u) \, .
```

#### Agent 2: active inference

The second agent minimizes the expected free energy plus the cost of the command, ``G(u) + P(u)``, as in Eq. (Q*) of the lecture. (The complexity term ``C(u)`` is zero here, because we evaluate each command under the predictive distribution.) For our robot, the position is observed exactly, and ``\theta`` only appears in the dynamics. In the EFE of Eq. (G1), the ambiguity term then does not depend on the command and the novelty term is zero, so the EFE equals the risk (up to a constant):

```math
G(u) = \underbrace{\mathrm{KL}\big[ q(x_{t+1} \mid u) \,\|\, \hat{p}(x_{t+1}) \big]}_{\text{risk}} = \underbrace{\mathbb{E}_{q(x_{t+1} \mid u)}\big[ -\log \hat{p}(x_{t+1}) \big]}_{\text{expected cost}} - \underbrace{H\big[ q(x_{t+1} \mid u) \big]}_{\text{entropy of predicted position}} \, .
```

So, the active inference agent minimizes the same expected cost as agent 1, *minus* the entropy of its predicted position: it likes commands with uncertain outcomes! But not just any uncertainty. The entropy of the predicted position is

```math
H\big[ q(x_{t+1} \mid u) \big] = \underbrace{\tfrac{1}{2} \log 2 \pi e \sigma_e^2}_{\text{noise, independent of } u} + \underbrace{I\big[ \theta; x_{t+1} \mid u \big]}_{\text{information gain}} \, ,
```

where ``I[\theta; x_{t+1} \mid u]`` is the mutual information between the motor gain and the next position: how much the agent expects to learn about its motor by executing command ``u``. This is the epistemic prior ``\tilde{p}(u) = \exp(H[q(x \mid u)])`` of Eq. (E1) in the lecture at work.

Both objectives are easy to compute in closed form for Gaussian distributions:
"""

# ╔═╡ 81a2cfb2-dd5c-5ed8-bbf9-0fd7dfa3f489
begin
	"Expected cost: cross-entropy between the predicted position N(μ, v) and the goal prior"
	expected_cost(μ, v) = ((v + (μ - x_target)^2) / goal_std^2 + log(2π * goal_std^2)) / 2

	"Entropy of a Gaussian distribution with variance v"
	entropy_gaussian(v) = log(2π * ℯ * v) / 2

	"Cost of a command: negative log of the control prior (up to a constant)"
	control_cost(u) = u^2 / (2 * σ_command^2)
end

# ╔═╡ 238f441b-236a-538e-a1c6-9a405ba5690e
begin
	"Decision-theoretic agent: expected cost + control cost"
	function decision_theoretic_objective(u, x, m_θ, v_θ)
		μ, v = predict_position(u, x, m_θ, v_θ)
		return expected_cost(μ, v) + control_cost(u)
	end

	"Active inference agent: expected free energy (= expected cost - entropy) + control cost"
	function active_inference_objective(u, x, m_θ, v_θ)
		μ, v = predict_position(u, x, m_θ, v_θ)
		return expected_cost(μ, v) - entropy_gaussian(v) + control_cost(u)
	end
end

# ╔═╡ 4f444f67-f8ac-5b63-9faf-a299cfeb824e
md"""
### The first decision

Let's compare both objectives for the very first decision, when the agent hasn't learned anything yet. The top panel shows the objectives (relative to their minimum value), the bottom panel the corresponding distributions over commands, ``q(u) \propto \exp(-J(u))`` and ``q(u) \propto \exp(-G(u) - P(u))``. The dashed lines mark the chosen commands.
"""

# ╔═╡ 19d547a3-78da-503a-9337-7be7748c8ea4
rewired_bonds

# ╔═╡ 46cb62ea-94ed-5bf8-afab-b7661d90a21e
let
	J_dt  = [decision_theoretic_objective(u, x_start, prior_mean, prior_var) for u in actions]
	G_aif = [active_inference_objective(u, x_start, prior_mean, prior_var) for u in actions]

	p1 = plot(actions, J_dt .- minimum(J_dt); color="orange", label="decision-theoretic agent: J(u)", ylabel="objective - minimum", ylims=(-0.5, 10), legend=:top)
	plot!(p1, actions, G_aif .- minimum(G_aif); color="blue", label="active inference agent: G(u) + P(u)")
	vline!(p1, [actions[argmin(J_dt)]]; color="orange", linestyle=:dash, label="")
	vline!(p1, [actions[argmin(G_aif)]]; color="blue", linestyle=:dash, label="")

	q_dt = exp.(minimum(J_dt) .- J_dt)
	q_aif = exp.(minimum(G_aif) .- G_aif)
	p2 = plot(actions, q_dt ./ sum(q_dt); color="orange", label="", xlabel="command u", ylabel="q(u)")
	plot!(p2, actions, q_aif ./ sum(q_aif); color="blue", label="")

	plot(p1, p2; layout=(2,1), size=(700,500))
end

# ╔═╡ a620f32e-e26b-54c6-be70-b1a12788e3e8
md"""
With the default settings (prior mean ``m_0 = 0``), the best command of the decision-theoretic agent is ``u = 0``: any command is as likely to move the robot away from the dock as towards it, and it only adds uncertainty about where the robot will end up. The active inference agent prefers a moderate command in either direction instead: the information it expects to gain outweighs the expected cost.
"""

# ╔═╡ 57a70848-2552-5abc-a804-a36f17211c62
md"""
## Running the agents

Both agents run in the same action-perception loop. The only difference is the objective they use to score the candidate commands:
"""

# ╔═╡ ca35e9ed-85d6-54ba-a59e-6a6dd42e5a99
rewired_bonds

# ╔═╡ 992f259b-7f5c-5258-a49e-3c4ad88a2760
md"""
With the default settings, the decision-theoretic agent never moves (the small movements you see are disturbances). It is stuck with its initial belief, because a command of ``u = 0`` reveals nothing about the motor gain. In adaptive control, this is known as the *turn-off* phenomenon of cautious controllers.

The active inference agent first makes a modest probing move, learns the sign and the size of its motor gain from the outcome, and then drives to the dock. Once it knows its motor gain, the entropy term hardly depends on the command anymore, and the agent behaves like a goal-directed controller.
"""

# ╔═╡ 567f51bd-8e31-58b0-acab-da3023ac03c3
exercise_statement("Information gain", header_level=2)

# ╔═╡ ac13794a-1077-52b6-8e73-f56f0ccfd735
md"""
Show that the entropy of the predicted position decomposes as

```math
H\big[q(x_{t+1} \mid u)\big] = \tfrac{1}{2} \log 2 \pi e \sigma_e^2 + I\big[\theta; x_{t+1} \mid u\big] \, , \qquad \text{with} \qquad I\big[\theta; x_{t+1} \mid u\big] = \tfrac{1}{2} \log \Big( 1 + \frac{v_\theta u^2}{\sigma_e^2} \Big) \, .
```

What happens to the information gain when the agent becomes more certain about its motor gain?
"""

# ╔═╡ 2695186f-15c3-53ad-b399-bcf1c6e34d40
hint(md"""
Use ``I[\theta; x] = H[x] - H[x \mid \theta]``, and the entropy of a Gaussian distribution, ``H\big[\mathcal{N}(\mu, v)\big] = \tfrac{1}{2} \log 2 \pi e v``.
""")

# ╔═╡ 47ee63fe-9266-5443-99b8-a088e4ab43d6
hide_solution(md"""
The predictive distribution is Gaussian with variance ``\sigma_e^2 + v_\theta u^2``, so ``H[q(x_{t+1} \mid u)] = \tfrac{1}{2}\log 2\pi e(\sigma_e^2 + v_\theta u^2)``. Given ``\theta``, the next position is distributed as ``\mathcal{N}(x_t + \theta u, \sigma_e^2)``, whose entropy ``\tfrac{1}{2}\log 2\pi e \sigma_e^2`` does not depend on ``\theta``. Hence,

```math
I[\theta; x_{t+1} \mid u] = H[x_{t+1} \mid u] - H[x_{t+1} \mid \theta, u] = \tfrac{1}{2} \log \frac{\sigma_e^2 + v_\theta u^2}{\sigma_e^2} = \tfrac{1}{2} \log \Big(1 + \frac{v_\theta u^2}{\sigma_e^2}\Big) \, .
```

The ratio ``v_\theta u^2 / \sigma_e^2`` is a signal-to-noise ratio: the variance of the outcome caused by the agent's uncertainty about ``\theta``, relative to the variance caused by the disturbance. As the agent learns, ``v_\theta \to 0``, the information gain vanishes for every command, and the active inference agent behaves like the decision-theoretic agent.
""")

# ╔═╡ 7da0c20e-bb2a-5219-a1a3-7b02b6e57b09
exercise_statement("When does curiosity stop?", header_level=2)

# ╔═╡ 4cfca320-5a9c-5fd4-91cf-05930d3a3842
md"""
The active inference agent does not explore unconditionally. Set the prior mean to ``m_0 = 0``, and make the goal prior narrower with the `goal_std` slider. Find the **largest** value of `goal_std` for which the active inference agent does **not** explore, that is, for which its first command is ``u = 0``. How does this value compare to the standard deviation of the disturbance, ``\sigma_e``?
"""

# ╔═╡ 416933d7-e4eb-55a8-88ed-607006e2951c
rewired_bonds

# ╔═╡ f3a70297-518f-5533-95ad-71e1a05afc49
hide_solution(md"""
With ``m_\theta = 0``, the objective of the active inference agent depends on the command only through ``w = u^2``:

```math
G(u) + P(u) = \frac{v_\theta w}{2 s_*^2} - \tfrac{1}{2}\log\Big(1 + \frac{v_\theta w}{\sigma_e^2}\Big) + \frac{w}{2\sigma_u^2} + \text{const.}
```

This function is convex in ``w``. So, the agent explores if and only if the function decreases at ``w = 0``, that is, if

```math
\frac{v_\theta}{\sigma_e^2} > \frac{v_\theta}{s_*^2} + \frac{1}{\sigma_u^2} \, .
```

With ``v_\theta = 1`` and ``\sigma_u = 2``, this means ``s_* > 0.05002``: the agent only explores if its goal prior is wider than (about) the disturbance, ``\sigma_e = 0.05``. Intuitively, the risk term wants the predicted position to be about as spread out as the goal prior. An agent whose goals are stricter than the noise it cannot avoid has no reason to add any uncertainty, not even to learn.
""")

# ╔═╡ b345a533-f09b-54d9-965e-ce53b25d6139
exercise_statement("A scientist without goals", header_level=2)

# ╔═╡ 55f1e530-af0e-5df8-bbb3-bcd3d9da68fc
md"""
What does an agent do if it has no goal at all? Remove the goal prior by making it infinitely wide (``s_* \to \infty``). The expected cost then no longer depends on the command, and the agent's objective reduces to ``-H\big[q(x_{t+1} \mid u)\big] + P(u)``: it only cares about information, and about the cost of its commands.

👉 Write the objective function of this agent, run it with `run_rewired_robot`, and compare it with the active inference agent using `plot_rewired`. What does this agent do, and why does it stop?
"""

# ╔═╡ b1157d47-a80e-5e7f-8f36-6de81880e52a
hint(md"""
An objective is a function of `(u, x, m_θ, v_θ)`. You can reuse `predict_position`, `entropy_gaussian` and `control_cost`. To compare two agents, call `plot_rewired(history_1, history_2; labels=["...", "..."])`.
""")

# ╔═╡ 8a05d41d-b09f-5602-8220-bcca5f2a6878
### YOUR CODE HERE

# ╔═╡ 2607c35f-9632-59c5-b0c8-753ea3a9cd21
hide_solution(md"""
```julia
function curious_objective(u, x, m_θ, v_θ)
	μ, v = predict_position(u, x, m_θ, v_θ)
	return -entropy_gaussian(v) + control_cost(u)
end
```

```julia
history_curious = run_rewired_robot(curious_objective)
```

```julia
plot_rewired(history_curious, history_aif; labels=["no goal", "active inference"])
```

The goal-free agent makes one big probing move, which tells it the motor gain with high precision. After that, any further information is not worth the cost of a command, so it stops: it has learned what it wanted to know, and it has no reason to go anywhere. As the lecture puts it: without goal priors, an active inference agent behaves like an automated scientist. The goal prior turns it into an engineer.
""")

# ╔═╡ fb51c8cd-958b-52b8-bc26-328337ad8c44
md"""
# Summary

- In **Bayesian control**, goals are prior beliefs about future states, and actions are latent variables. Choosing an action means inferring it, by message passing in a generative model of the future (*planning as inference*).
- An agent runs an **action-perception loop**: plan, act, perceive, repeat. Perception and planning are both inference, but we keep them apart, to avoid wishful thinking.
- For a linear Gaussian system, Bayesian control with Gaussian goal and control priors is equivalent to linear-quadratic optimal control (see the [Appendix](#Appendix)).
- The **expected free energy** scores *beliefs* about future states. Its risk term contains the entropy of the predicted states, which rewards actions that are informative about unknown parameters. This makes an active inference agent explore when it needs to, and stop exploring once it has learned enough.
- How much an active inference agent explores depends on its goal prior: an agent with goals stricter than the noise it cannot avoid (``s_* \lesssim \sigma_e``) does not explore, and an agent without goals explores until it has learned enough, and then stops.

Curious how this scales up? The same ideas drive the NARX- and ARX-based active inference agents in [Kouw (2023)](https://arxiv.org/abs/2312.15046) and [Nisslbeck & Kouw (2024)](https://arxiv.org/abs/2410.10415), which learn the dynamics of systems such as a pendulum while controlling them. [De Vries et al. (2025)](https://arxiv.org/abs/2504.14898) show how minimizing the expected free energy itself can be realized by message passing.
"""

# ╔═╡ 75b92f65-6746-4f83-8669-09c3fc7a9393
challenge_statement("Real world example: active inference in traffic and mobility"; header_level=1, big=true)

# ╔═╡ 69df80d0-a05b-49f5-a576-fe3ae8b629f3
md"""
The ideas in this notebook are not limited to robots. In the [AiMTT](https://aimtt.nl) programme (*AI Learning Initiative for Multi-modal Traffic and Transportation*), TU/e and TU Delft work together with NDW, Rijkswaterstaat and industry partners on active inference for traffic management. Traffic lights, ramp meters and travellers increasingly make their own decisions, each based on their own, incomplete information. Active inference offers a principled way to build such decision makers: agents that form expectations, take their uncertainty into account, and adjust when reality deviates from their predictions.

In [one of the AiMTT experiments](http://biaslab.github.io/aim-tt), a population of simulated travellers chooses every day between two routes between the Valburg and Velperbroek interchanges near Arnhem. The travellers have to *learn* how congested each route is while they use it, and they have to *adapt* when the network changes, for example when road construction halves the capacity of one of the routes for a few weeks.

Below, we strip this experiment down to a single commuter, and we use exactly the same tools as for the rewired robot.
"""

# ╔═╡ 09279d95-27f2-4eaa-9267-6b9209f60632
md"""
## The commuter's problem

Every morning, a commuter drives from home to work. There are two routes:
- route **A**, the motorway, which usually takes about 22 minutes, and
- route **B**, a secondary road, which usually takes about 32 minutes.

The commuter has just moved, and doesn't know these travel times: they can only find out by driving. Moreover, they only experience the route they drive: there is no way to know how long the other route would have taken today. The travel times also vary from day to day, with a standard deviation of about 2 minutes.

To make things interesting, there are **road works on route A** from day 31 to day 70, during which the motorway takes about 45 minutes. Nobody tells the commuter when the road works start or end.

The road network is simulated by a `CommuteNetwork` (see the [Appendix](#Appendix)). The function `drive!(network, route)` returns the travel time (in minutes) of the next day's trip along `route` (1 = A, 2 = B).
"""

# ╔═╡ d6c9a2b1-e636-409a-a9a2-7e5389bcb4e6
md"""
## The commuter's model

The commuter keeps a belief about the mean travel time ``\theta_{r,d}`` of each route ``r \in \{A, B\}`` on day ``d``. Their generative model consists of a prior, a state transition, a likelihood and a goal prior:

```math
\begin{aligned}
p(\theta_{r,0}) &= \mathcal{N}(\theta_{r,0} \mid 30, 10^2) \\
p(\theta_{r,d} \mid \theta_{r,d-1}) &= \mathcal{N}(\theta_{r,d} \mid \theta_{r,d-1}, \sigma_\text{drift}^2) \\
p(y_d \mid \theta_{r,d}) &= \mathcal{N}(y_d \mid \theta_{r,d}, \sigma_y^2) \\
\hat{p}(y_d) &= \mathcal{N}(y_d \mid y_*, s_*^2) \, .
\end{aligned}
```

- The prior says: "a trip probably takes about half an hour", for both routes.
- The state transition says that travel times **may change** from day to day.
- The likelihood describes the travel time ``y_d`` of today's trip, *if* route ``r`` was driven on day ``d``.
- The goal prior says that the commuter wants to be at work in ``y_* = 20`` minutes.

The state transition is the crucial ingredient. A commuter who believes that travel times never change (``\sigma_\text{drift} = 0``) is learning a fixed parameter, as in [Probabilistic Programming 2](https://bmlip.github.io/course/probprog/PP2%20-%20Bayesian%20regression%20and%20classification.html). A commuter with ``\sigma_\text{drift} > 0`` is tracking a hidden state, as in [Probabilistic Programming 4](https://bmlip.github.io/course/probprog/PP4%20-%20Bayesian%20filtering%20and%20smoothing.html): they expect that the road network can change.

In the code, ``\sigma_\text{drift}`` is called `commute_drift_std`, ``\sigma_y`` is `commute_obs_std` and ``s_*`` is `commute_goal_std`. With the sliders, you can change ``\sigma_\text{drift}`` and ``s_*``.
"""

# ╔═╡ cc583d4c-cc9c-4bcd-a29d-6abfe12d7072
begin
	N_days = 120                  # number of days
	commute_prior_mean = 30.0     # prior mean of the travel time of each route (min)
	commute_prior_var = 10.0^2    # prior variance of the travel time of each route (min²)
	commute_obs_std = 2.0         # standard deviation of the travel time of a single trip (min)
	commute_target = 20.0         # goal prior: the commuter wants to be at work in 20 minutes
end

# ╔═╡ 61a5c9b4-8e5e-4b6a-8e0f-d3ff0866d318
begin
	commute_bonds = PlutoUI.ExperimentalLayout.vbox([
		@bindname(commute_drift_std, Slider([0.0, 0.5, 1.0, 1.5, 2.0, 3.0]; default=1.5, show_value=true))
		@bindname(commute_goal_std, Slider([5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]; default=15.0, show_value=true))
	])
end

# ╔═╡ 07a717c9-5ec2-43cf-b0a3-17691a3ba5de
md"""
After driving route ``r`` on day ``d``, the commuter updates their beliefs ``q(\theta_{r,d}) = \mathcal{N}(\theta_{r,d} \mid m_r, v_r)`` about both routes. This is one step of a Kalman filter, as for the parking robot, with one twist: only the route that was driven is observed. In RxInfer, we simply pass `missing` as the observation of the other route.

The posterior for the route that was not driven is then just the prediction, ``\mathcal{N}(\theta_{r,d} \mid m_r, v_r + \sigma_\text{drift}^2)``: its mean stays the same, but its variance **grows** with every day that the commuter stays away from that route.
"""

# ╔═╡ 4eec31d0-a8f4-49e1-8f88-42c1d1b94188
@model function commute_learning(y, m_prev, v_prev, σ2_drift, σ2_obs)
	"One day of learning the travel times of both routes"

	for r in 1:2
		# Belief about yesterday's travel time of route r
		θ_prev[r] ~ NormalMeanVariance(m_prev[r], v_prev[r])

		# The travel time may have changed overnight
		θ[r] ~ NormalMeanVariance(θ_prev[r], σ2_drift)

		# Today's trip (missing for the route that was not driven)
		y[r] ~ NormalMeanVariance(θ[r], σ2_obs)
	end
end

# ╔═╡ 5290535c-3444-481f-a0bc-760f8230abe8
"Update the beliefs N(m, v) about the travel times of both routes, after driving `route` and observing travel time y"
function update_route_beliefs(m, v, route, y)

	# Only the route that was driven is observed
	observations = [r == route ? y : missing for r in 1:2]

	result = infer(
		model = commute_learning(m_prev=m, v_prev=v, σ2_drift=commute_drift_std^2, σ2_obs=commute_obs_std^2),
		data  = (y = observations,),
	)
	return mean.(result.posteriors[:θ]), var.(result.posteriors[:θ])
end

# ╔═╡ e4e2c95f-5484-4e90-afec-ada06ff8bb8c
md"""
## Choosing a route

To choose a route, the commuter predicts tomorrow's travel time on each route:

```math
q(y \mid r) = \mathcal{N}\big(y \mid m_r, \ v_r + \sigma_\text{drift}^2 + \sigma_y^2\big) \, .
```

The commuter scores both routes in the same way as the agents of the rewired robot. There are only two candidate actions now, and there is no control cost (both routes take the same effort to drive):

```math
\begin{aligned}
J(r) &= \mathbb{E}_{q(y \mid r)}\big[ -\log \hat{p}(y) \big] && \text{(decision-theoretic commuter)} \\
G(r) &= \mathbb{E}_{q(y \mid r)}\big[ -\log \hat{p}(y) \big] - H\big[ q(y \mid r) \big] && \text{(active inference commuter)}
\end{aligned}
```

As before, the entropy contains the information gain: ``H[q(y \mid r)] = \tfrac{1}{2}\log 2\pi e \sigma_y^2 + I[\theta_r; y \mid r]``. The uncertainty about a route grows while it is not driven, so the longer the commuter stays away from a route, the more they expect to learn by driving it.
"""

# ╔═╡ 8c873a95-c5a2-484d-ade3-bc129515438a
"Predictive distribution of tomorrow's travel time on a route with belief N(m_r, v_r): returns (mean, variance)"
function predict_travel_time(m_r, v_r)
	return m_r, v_r + commute_drift_std^2 + commute_obs_std^2
end

# ╔═╡ 61613276-a1dd-4fc6-8bd4-423f3cce7cdc
begin
	"Expected cost: cross-entropy between the predicted travel time N(μ, v) and the goal prior"
	route_expected_cost(μ, v) = ((v + (μ - commute_target)^2) / commute_goal_std^2 + log(2π * commute_goal_std^2)) / 2

	"Decision-theoretic commuter: expected cost"
	function route_dt_objective(m_r, v_r)
		μ, v = predict_travel_time(m_r, v_r)
		return route_expected_cost(μ, v)
	end

	"Active inference commuter: expected free energy (= expected cost - entropy)"
	function route_aif_objective(m_r, v_r)
		μ, v = predict_travel_time(m_r, v_r)
		return route_expected_cost(μ, v) - entropy_gaussian(v)
	end
end

# ╔═╡ 2216a8fb-81d9-4376-8a0a-5c1dfabdf66d
md"""
## Driving to work for 120 days

Both commuters run the same action-perception loop: plan (score both routes), act (drive to work), and learn (update the beliefs about the travel times):
"""

# ╔═╡ b1626647-7841-4620-982b-b352d78c1770
commute_bonds

# ╔═╡ 4d863121-0381-4af0-a231-bd404060cd0f
md"""
With the default settings (``\sigma_\text{drift} = 1.5`` and ``s_* = 15``), the plots tell three stories:

1. **Learning the network** (days 1 to 30). Both commuters start with the same prior. The decision-theoretic commuter drives route A on the first day (both routes look the same, and `argmin` picks the first one), finds it fast, and never tries route B. Their belief about route B stays at the prior mean, with an ever-growing uncertainty (the wide orange band). The active inference commuter tries route B on the second day, and checks it again every few days, whenever their uncertainty about route B has grown enough. This curiosity has a price: about 10 extra minutes per check (bottom panel).
2. **Road works** (days 31 to 70). Both commuters notice the road works within a few days, and switch to route B. They can only do so this quickly because their model allows the travel times to change (see the exercise below).
3. **After the road works** (days 71 to 120). The decision-theoretic commuter remembers route A as slow, and never drives it again. So they never find out that the road works are over, and they lose 10 minutes every day, for the rest of the simulation. This is the *turn-off* phenomenon of the rewired robot again, but now it happens halfway: a change in the world has made old knowledge stale. The uncertainty of the active inference commuter about route A keeps growing, until driving it becomes worth the risk. They find out that the road works are over, and return to route A.

In the long run, curiosity pays off: the active inference commuter loses much less time than the decision-theoretic commuter (see the table).
"""

# ╔═╡ c12907ed-9c2a-46f2-9962-0be558d86a3e
exercise_statement("A world that never changes", header_level=2)

# ╔═╡ 8e7317f4-d17d-4137-b419-460d27c6778a
md"""
👉 Set `commute_drift_std` to 0, so that the commuters believe that the travel time of each route is a fixed (but unknown) number. How do the commuters respond to the road works now? Can you explain it?
"""

# ╔═╡ 5301881d-f47c-48df-8345-25b2267688e4
commute_bonds

# ╔═╡ 29d0b601-2d6d-48f8-ac2f-eb1f6e75a51f
hide_solution(md"""
With ``\sigma_\text{drift} = 0``, the travel time of a route is a fixed parameter. The posterior mean is then (roughly) the average of all trips along that route, and the posterior variance shrinks as ``\sigma_y^2 / n`` after ``n`` trips. After 30 trips along route A, the commuter is very sure that it takes 22 minutes, and each trip through the road works moves the average only a little. For example, after 20 days of road works, the average of all trips along route A is still only ``(30 \cdot 22 + 20 \cdot 45) / 50 \approx 31`` minutes.

So, both commuters keep driving through the road works for weeks. The decision-theoretic commuter only switches to route B after more than a month. And when the road works are over, the static model makes things worse a second time: the commuter's belief about route A is stuck at the average of all their trips (about 34 minutes), and they are just as certain about it as about route B. There is nothing left to learn, so the decision-theoretic commuter never goes back to route A. With ``\sigma_\text{drift} = 0``, the decision-theoretic commuter loses almost twice as much time as with ``\sigma_\text{drift} = 1.5``.

A model that cannot represent change, cannot detect it either: to adapt, an agent must believe that the world can change. (The agents in the AiMTT experiments use a similar trick: they inflate their uncertainty about routes they have not driven recently.)

Note that adapting has a price as well: a commuter who believes that travel times drift keeps forgetting what they know about the other route, and has to check it every now and then. This pays off here, because the road works last for almost six weeks. If the road works only lasted a few days, a stubborn commuter could actually lose less time!
""")

# ╔═╡ 2753fe1b-48c4-4b66-bde2-e5d067bbdaca
exercise_statement("Strict commuters", header_level=2)

# ╔═╡ 8a77b17b-de60-465a-b7f7-cecf2a9ad21c
md"""
👉 Set `commute_drift_std` back to 1.5, and make the goal prior narrower: set `commute_goal_std` to 10. What does the active inference commuter do after the road works? Can you explain it?
"""

# ╔═╡ cc0c9e01-9de6-42e0-a03d-509988c4f468
commute_bonds

# ╔═╡ c8706bc6-4133-4bb2-a6c8-1c1fe5b7b56c
hint(md"""
Write ``G(r)`` as a function of the mean ``\mu_r`` and the variance ``V_r`` of the predicted travel time. For which value of ``V_r`` is ``G(r)`` minimal?
""")

# ╔═╡ 25802cd8-bca5-4813-b5c4-6836ee0c84cb
hide_solution(md"""
The strict active inference commuter stays on route B after the road works, just like the decision-theoretic commuter.

Up to a constant, the expected free energy of route ``r`` is

```math
G(r) = \frac{(\mu_r - y_*)^2 + V_r}{2 s_*^2} - \tfrac{1}{2} \log V_r \, .
```

Uncertainty is attractive, but only up to a point: as a function of ``V_r``, ``G(r)`` is minimal at ``V_r = s_*^2``. Compared to the well-known route B (with ``V_B \approx 8``), the uncertainty about route A can lower its score by at most

```math
\Delta G_\text{max} = \tfrac{1}{2} \log \frac{s_*^2}{V_B} - \frac{s_*^2 - V_B}{2 s_*^2} \, .
```

At the end of the road works, the commuter believes that route A takes ``\mu_A \approx 44`` minutes and route B ``\mu_B \approx 32`` minutes. This costs route A an extra ``\big((\mu_A - y_*)^2 - (\mu_B - y_*)^2\big) / (2 s_*^2)``:

| ``s_*`` | maximal curiosity bonus ``\Delta G_\text{max}`` | penalty for the slow memory of route A |
|:--:|:--:|:--:|
| 15 | ≈ 1.2 | ≈ 1.0 |
| 10 | ≈ 0.8 | ≈ 2.2 |

With ``s_* = 15``, curiosity eventually wins. With ``s_* = 10``, it never does: a strict commuter cares so much about the mean travel time that the stale memory of a slow route A can never be outweighed by curiosity. This is the same lesson as in the exercise *When does curiosity stop?*: the stricter the goals, the less an active inference agent explores.
""")

# ╔═╡ ccfd1039-4346-4ff8-a670-e8e0a2857aa8
md"""
## From one commuter to a whole corridor

In the [AiMTT experiments](http://biaslab.github.io/aim-tt), the travel times are not fixed numbers. 2000 active inference travellers share the same two routes, and the travel time of a route depends on how many of them choose it, through the queues that form on the road. Each traveller only observes the route they drove, learns the free-flow travel time, the capacity and the queue of each route, and chooses a route by minimizing the expected free energy, just like our commuter.

Compared to classical route-choice models (a logit model and a reinforcement-learning model), the population of active inference travellers produces more stable route splits, and adapts more smoothly to road construction and to random incidents. You can read more about these experiments, and about the programme at [aimtt.nl](https://aimtt.nl).
"""

# ╔═╡ cce2a7a0-abc0-5d36-87f6-c54391429e78
md"""
# Appendix
"""

# ╔═╡ c787491a-252f-5bd7-9860-1a6aadf205de
md"""
### Simulated environments
"""

# ╔═╡ 889c61d3-3173-5f7c-aac1-71a245e0f752
begin
	"Simulated robot on a rail (environment of Part 1)"
	mutable struct RobotOnRail
		state::Vector{Float64}    # true state [position, velocity], hidden from the agent
		rng::Xoshiro
	end
	RobotOnRail(; seed=1) = RobotOnRail([0.0, 0.0], Xoshiro(seed))

	"Apply force u during one time step, and return a noisy measurement of the new position"
	function apply_force!(robot::RobotOnRail, u::Real)
		w = cholesky(Q).L * randn(robot.rng, 2)
		robot.state = A*robot.state + B[:,1]*u + w
		return dot(C, robot.state) + σ_y*randn(robot.rng)
	end
end

# ╔═╡ e30bfe40-abaa-53da-bf2c-214be145a48b
"Run the action-perception loop for N_steps time steps, with a given planning function"
function run_robot(planner; seed=1)

	# The environment: a simulated robot on a rail
	robot = RobotOnRail(seed=seed)

	# The agent's initial belief about the robot's state
	m, S = m_init, S_init

	# Bookkeeping
	sim = (
		positions    = Float64[],            # true positions (hidden from the agent)
		observations = Float64[],            # measured positions
		m            = Vector{Float64}[],    # belief means
		S            = Matrix{Float64}[],    # belief covariance matrices
		forces       = Float64[],            # applied forces
		plans        = Vector{Float64}[],    # planned positions at each time step
	)

	for t in 1:N_steps

		# 1. Plan: infer the forces, and pick the most probable first force
		plan = planner(m, S)
		u = mean(plan.posteriors[:u][1])     # a 1-element vector

		# 2. Act: apply the force and receive a measurement
		y = apply_force!(robot, u[1])

		# 3. Perceive: update the belief about the robot's state
		m, S = perceive(m, S, u, y)

		# Store results
		push!(sim.positions, robot.state[1])
		push!(sim.observations, y)
		push!(sim.m, m)
		push!(sim.S, S)
		push!(sim.forces, u[1])
		push!(sim.plans, [mean(q)[1] for q in plan.posteriors[:x]])
	end
	return sim
end

# ╔═╡ 37da6c61-47cd-5efe-a7f5-faea068a83c4
sim = run_robot(plan_from_estimate)

# ╔═╡ 78680031-c66c-5e47-beed-d23baf55148d
let
	positions = sim.positions
	times = Δt .* (1:length(positions))

	checks = [
		"the force never exceeds 3 N" => maximum(abs.(sim.forces)) <= 3.0,
		"overshoot of at most 25 cm" => maximum(positions) - dock <= 0.25,
		"within 25 cm of the dock from t = 8 s onwards" => all(abs.(positions[times .>= 8.0 - 1e-9] .- dock) .<= 0.25),
	]
	report = Markdown.parse(join(["- $(ok ? "✅" : "❌") $(name)" for (name, ok) in checks], "\n"))

	all(last.(checks)) ? correct(report) : keep_working(report)
end

# ╔═╡ 90d3f2cd-d510-5694-8c39-62e1635bdf4c
begin
	"Simulated robot with an unknown motor gain (environment of Part 2)"
	mutable struct RewiredRobot
		position::Float64
		θ::Float64            # true motor gain, hidden from the agent
		rng::Xoshiro
	end
	RewiredRobot(θ; seed=2) = RewiredRobot(x_start, θ, Xoshiro(seed))

	"Send speed command u (clipped to the allowed range), and return the new position"
	function send_command!(robot::RewiredRobot, u::Real)
		u = clamp(u, -u_limit, u_limit)
		robot.position += robot.θ*u + σ_e*randn(robot.rng)
		return robot.position
	end
end

# ╔═╡ 6b05cbc4-f3b8-53e6-909d-0979e7efbd56
"Run the action-perception loop of the rewired robot, for an agent with the given objective"
function run_rewired_robot(objective)

	# The environment: a robot with an unknown motor gain
	robot = RewiredRobot(true_gain)
	x = robot.position

	# The agent's prior belief about its motor gain
	m_θ, v_θ = prior_mean, prior_var

	# Bookkeeping
	history = (x = [x], u = Float64[], m_θ = [m_θ], v_θ = [v_θ])

	for t in 1:N_rewired

		# 1. Plan: score all candidate commands, and pick the best one
		scores = [objective(u, x, m_θ, v_θ) for u in actions]
		u = actions[argmin(scores)]

		# 2. Act: send the command, and observe the new position
		x_new = send_command!(robot, u)

		# 3. Learn: update the belief about the motor gain
		m_θ, v_θ = update_motor_belief(m_θ, v_θ, u, x_new - x)
		x = x_new

		# Store results
		push!(history.x, x)
		push!(history.u, u)
		push!(history.m_θ, m_θ)
		push!(history.v_θ, v_θ)
	end
	return history
end

# ╔═╡ 962aaa17-1c69-59a2-bb18-8347c3336590
history_dt = run_rewired_robot(decision_theoretic_objective)

# ╔═╡ a21cf15e-e22b-57b9-8de6-83916e2f3b42
history_aif = run_rewired_robot(active_inference_objective)

# ╔═╡ 344f7abb-8a57-5756-920d-04b6ebdbabfc
let
	u_first = history_aif.u[1]
	explores = abs(u_first) > 1e-6

	if prior_mean != 0.0
		keep_working(md"Set the prior mean back to 0 for this exercise.")
	elseif !explores && goal_std == 0.05
		correct(md"Well done! At this value, the active inference agent no longer explores.")
	elseif !explores
		keep_working(md"The agent does not explore at this value. Is there a larger value for which it still does not explore?")
	else
		keep_working(md"The agent still explores at this value: its first command is u = $(round(u_first, digits=2)).")
	end
end

# ╔═╡ c446cf9f-12c2-4a6f-a0e1-0cd0bb4c3f46
begin
	# The true road network (hidden from the commuter)
	tt_usual = [22.0, 32.0]      # usual travel times of route A (motorway) and route B (secondary road), in minutes
	tt_roadworks = 45.0          # travel time of route A during the road works (min)
	roadworks_days = 31:70       # days with road works on route A
	tt_noise_std = 2.0           # day-to-day variation of the travel time (min)

	"True mean travel time of a route (1 = A, 2 = B) on a given day"
	commute_true_tt(route, day) = (route == 1 && day in roadworks_days) ? tt_roadworks : tt_usual[route]

	"Simulated road network with two routes from home to work (environment of the traffic example)"
	mutable struct CommuteNetwork
		day::Int
		rng::Xoshiro
	end
	CommuteNetwork(; seed=2) = CommuteNetwork(0, Xoshiro(seed))

	"Drive along route 1 (A) or 2 (B) on the next day, and return the travel time (min)"
	function drive!(network::CommuteNetwork, route::Int)
		network.day += 1
		return commute_true_tt(route, network.day) + tt_noise_std*randn(network.rng)
	end
end

# ╔═╡ 3068b8c8-a397-4bd8-9f81-83d5062f3aca
"Run the action-perception loop of a commuter with the given objective, for N_days days"
function run_commuter(objective; seed=2)

	# The environment: a road network with two routes
	network = CommuteNetwork(seed=seed)

	# The commuter's prior beliefs about the travel times of route A and route B
	m = fill(commute_prior_mean, 2)
	v = fill(commute_prior_var, 2)

	# Bookkeeping
	history = (
		route = Int[],        # chosen routes (1 = A, 2 = B)
		tt    = Float64[],    # experienced travel times
		extra = Float64[],    # extra travel time, compared to the faster route on that day
		m     = [m],          # belief means
		v     = [v],          # belief variances
	)

	for day in 1:N_days

		# 1. Plan: score both routes, and pick the best one
		scores = [objective(m[r], v[r]) for r in 1:2]
		route = argmin(scores)

		# 2. Act: drive to work, and observe the travel time
		y = drive!(network, route)

		# 3. Learn: update the beliefs about the travel times
		m, v = update_route_beliefs(m, v, route, y)

		# Store results (the true travel times are only used for evaluation)
		push!(history.route, route)
		push!(history.tt, y)
		push!(history.extra, commute_true_tt(route, day) - minimum(commute_true_tt(r, day) for r in 1:2))
		push!(history.m, m)
		push!(history.v, v)
	end
	return history
end

# ╔═╡ bb9b2685-375b-4d5a-91b6-dbe83d2feeb1
commute_dt = run_commuter(route_dt_objective)

# ╔═╡ e29f44d3-0b71-444e-bb76-80207694ccf6
commute_aif = run_commuter(route_aif_objective)

# ╔═╡ f28dc0f1-5366-5395-869c-75a260e98aa9
md"""
### Plotting functions
"""

# ╔═╡ ade5e467-96f1-5730-9796-c3eddc09e001
"Plot the planned positions and forces of one or more planning results"
function plot_plans(results, labels; colors=["blue", "red", "purple"])
	p1 = plot(ylabel="position (m)", ylims=(-0.5, dock+2), legend=:bottomright)
	p2 = plot(xlabel="time (s)", ylabel="force (N)")
	for (i, result) in enumerate(results)
		T = length(result.posteriors[:u])
		times = Δt .* (1:T)
		pos_mean = [mean(q)[1] for q in result.posteriors[:x]]
		pos_std  = [sqrt(var(q)[1]) for q in result.posteriors[:x]]
		u_mean   = [mean(q)[1] for q in result.posteriors[:u]]
		u_std    = [sqrt(var(q)[1]) for q in result.posteriors[:u]]
		plot!(p1, times, pos_mean; ribbon=2 .* pos_std, fillalpha=0.15, color=colors[i], label=labels[i])
		plot!(p2, times, u_mean; ribbon=2 .* u_std, fillalpha=0.15, color=colors[i], marker=:circle, markersize=3, label="")
	end
	hline!(p1, [dock]; color="green", linestyle=:dash, label="dock")
	plot(p1, p2; layout=(2,1), size=(700,500))
end

# ╔═╡ a1c34b56-5424-5bf9-8bd7-1fe05459f090
plot_plans([first_plan], ["planned position"])

# ╔═╡ 9fb427b6-263a-560e-9e35-89990b77530c
plot_plans([first_plan, plan_estimate], ["plan from full belief", "plan from point estimate"])

# ╔═╡ e4876603-a1f0-54db-a40c-ceed18b6c9dd
"Plot the true, measured and estimated positions, and the applied forces, of a simulation"
function plot_simulation(sim)
	N = length(sim.positions)
	times = Δt .* (1:N)
	m_pos = [m[1] for m in sim.m]
	s_pos = [sqrt(S[1,1]) for S in sim.S]

	p1 = plot(times, m_pos; ribbon=2 .* s_pos, color="blue", fillalpha=0.2, label="estimated position", ylabel="position (m)", legend=:bottomright)
	plot!(p1, times, sim.positions; color="red", label="true position")
	scatter!(p1, times, sim.observations; color="black", markersize=2, alpha=0.3, label="measurements")
	hline!(p1, [dock]; color="green", linestyle=:dash, label="dock")

	p2 = plot(times, sim.forces; color="purple", marker=:circle, markersize=2, label="force", xlabel="time (s)", ylabel="force (N)")
	hline!(p2, [-3.0, 3.0]; color="gray", linestyle=:dot, label="± 3 N")

	plot(p1, p2; layout=(2,1), size=(700,500))
end

# ╔═╡ fb3b4276-3784-538b-9ddb-ce93a77c7716
plot_simulation(sim)

# ╔═╡ bbfaf3f4-cd13-5bc5-b593-fbac3dd1afcd
"Plot the trajectory up to time step t, and the plan that the agent made at that moment"
function plot_snapshot(sim, t)
	N = length(sim.positions)
	T = length(sim.plans[t])
	past_times = Δt .* (0:t-1)
	past_positions = [0.0; sim.positions[1:t-1]]
	plan_times = Δt .* ((t-1) .+ (1:T))

	p = plot(past_times, past_positions; color="red", marker=:circle, markersize=2, label="true position so far", xlabel="time (s)", ylabel="position (m)", xlims=(0, Δt*(N+T)), ylims=(-0.5, dock+2), legend=:bottomright)
	plot!(p, plan_times, sim.plans[t]; color="blue", linestyle=:dash, label="plan made at t = $(round((t-1)*Δt, digits=1)) s")
	hline!(p, [dock]; color="green", linestyle=:dot, label="dock")
	return p
end

# ╔═╡ 0b5382a0-da1c-5b52-a42c-43ec91119188
plot_snapshot(sim, t_view)

# ╔═╡ cc6f1751-897d-582f-8b2a-ef3764823f09
"Compare the positions, beliefs and commands of two agents controlling the rewired robot"
function plot_rewired(h1, h2; labels=["decision-theoretic", "active inference"])
	N = length(h1.u)
	steps = 0:N

	p1 = plot(steps, h1.x; color="orange", marker=:circle, markersize=3, label=labels[1], ylabel="position (m)", legend=:right)
	plot!(p1, steps, h2.x; color="blue", marker=:circle, markersize=3, label=labels[2])
	hline!(p1, [x_target]; color="green", linestyle=:dash, label="dock")

	p2 = plot(steps, h1.m_θ; ribbon=2 .* sqrt.(h1.v_θ), color="orange", fillalpha=0.2, label="", ylabel="belief about θ")
	plot!(p2, steps, h2.m_θ; ribbon=2 .* sqrt.(h2.v_θ), color="blue", fillalpha=0.2, label="")
	hline!(p2, [true_gain]; color="black", linestyle=:dot, label="true θ")

	p3 = plot(1:N, h1.u; color="orange", marker=:circle, markersize=3, label="", xlabel="time step", ylabel="command u")
	plot!(p3, 1:N, h2.u; color="blue", marker=:circle, markersize=3, label="")

	plot(p1, p2, p3; layout=(3,1), size=(700,750))
end

# ╔═╡ e43f046f-cd46-5d21-89a4-4367de89af57
plot_rewired(history_dt, history_aif)

# ╔═╡ 67dc62b5-f28a-4b31-b2a1-640a7c0d2c80
"Compare the route choices, beliefs and extra travel times of two commuters"
function plot_commute(h1, h2; labels=["decision-theoretic", "active inference"], colors=["orange", "blue"])
	days = 1:N_days
	shade_roadworks!(p) = vspan!(p, [first(roadworks_days) - 0.5, last(roadworks_days) + 0.5]; color="gray", alpha=0.15, label="")

	# Which route did each commuter take?
	p1 = plot(ylabel="route", yticks=([1, 2], ["B", "A"]), ylims=(0.5, 2.5), legend=:outertopright, title="route choice (gray: road works on route A)", titlefontsize=10)
	shade_roadworks!(p1)
	scatter!(p1, days, 3 .- h1.route .+ 0.08; color=colors[1], markersize=3, markerstrokewidth=0, label=labels[1])
	scatter!(p1, days, 3 .- h2.route .- 0.08; color=colors[2], markersize=3, markerstrokewidth=0, label=labels[2])

	# What did each commuter believe?
	function belief_panel(h, label, color)
		p = plot(ylabel="travel time (min)", ylims=(10, 60), legend=:outertopright, title="beliefs of the $(label) commuter", titlefontsize=10)
		shade_roadworks!(p)
		for (r, style, name) in [(1, :solid, "A"), (2, :dash, "B")]
			μ = [m[r] for m in h.m[2:end]]
			σ = [sqrt(v[r]) for v in h.v[2:end]]
			plot!(p, days, μ; ribbon=2 .* σ, fillalpha=0.15, color=color, linestyle=style, label="belief about $(name)")
			plot!(p, days, [commute_true_tt(r, d) for d in days]; color="black", linewidth=1, linestyle=style, label="true $(name)")
		end
		return p
	end
	p2 = belief_panel(h1, labels[1], colors[1])
	p3 = belief_panel(h2, labels[2], colors[2])

	# How much time did each commuter lose?
	p4 = plot(xlabel="day", ylabel="minutes", legend=:outertopright, title="cumulative extra travel time (compared to always taking the faster route)", titlefontsize=10)
	shade_roadworks!(p4)
	plot!(p4, days, cumsum(h1.extra); color=colors[1], label=labels[1])
	plot!(p4, days, cumsum(h2.extra); color=colors[2], label=labels[2])

	plot(p1, p2, p3, p4; layout=(4,1), size=(800,1000))
end

# ╔═╡ 9b9dcb16-6321-4937-9d81-edd9bc6f156d
plot_commute(commute_dt, commute_aif)

# ╔═╡ 8d91235c-3349-4562-a305-2465516e8af8
"Summarize the route choices of two commuters in a table"
function commute_summary(h1, h2; labels=["decision-theoretic", "active inference"])
	last_day = last(roadworks_days)
	function row(h, label)
		days_on_A = count(==(1), h.route[roadworks_days])
		back = findfirst(==(1), h.route[last_day+1:end])
		back_on_A = back === nothing ? "never" : "day $(last_day + back)"
		return "| $(label) | $(days_on_A) of $(length(roadworks_days)) | $(back_on_A) | $(round(Int, sum(h.extra))) |"
	end
	Markdown.parse(join([
		"| commuter | days on route A during the road works | back on route A after the road works | total extra travel time (min) |",
		"|:--|:--:|:--:|:--:|",
		row(h1, labels[1]),
		row(h2, labels[2]),
	], "\n"))
end

# ╔═╡ f54dceb2-0956-4c04-ac9e-10c638850668
commute_summary(commute_dt, commute_aif)

# ╔═╡ 8c721823-37c3-4316-a090-bb7a8ac36b7b
commute_summary(commute_dt, commute_aif)

# ╔═╡ e0278cd0-3280-4a3f-9fa3-93e753a95585
commute_summary(commute_dt, commute_aif)

# ╔═╡ ec4514f4-7df5-5634-8f39-9685615df687
md"""
### Bayesian control and LQR

If you have taken a course on optimal control, the Bayesian controller may remind you of the *linear-quadratic regulator* (LQR). Taking the negative logarithm of the goal and control priors gives quadratic costs:

```math
-\log \hat{p}(x_k) - \log p(u_k) = \tfrac{1}{2} (x_k - x_*)^{\top} \Sigma_*^{-1} (x_k - x_*) + \tfrac{1}{2} \sigma_u^{-2} u_k^2 + \text{const.}
```

In fact, if the planner knows the current state and assumes that there is no process noise (``Q \to 0``), then the most probable force sequence is exactly the solution of the linear-quadratic optimal control problem with state cost ``\Sigma_*^{-1}`` and control cost ``\sigma_u^{-2}``, over the same horizon. Let's verify this numerically (the function `lqr_forces` below computes the linear-quadratic solution):
"""

# ╔═╡ b85beb41-21d4-5d5f-b620-12474f3e4325
md"""
With process noise (``Q > 0``), the two differ slightly: the Bayesian controller assumes that the noise may *help* it reach the goal, another form of wishful thinking ([Levine, 2018](https://arxiv.org/abs/1805.00909)). For the small process noise of our robot, this difference is negligible.
"""

# ╔═╡ 425b4a40-bcc3-5a94-ac12-caa6c26c9bfe
"""
Solve the linear-quadratic control problem over horizon T, starting from state x_now:

	minimize  ∑ₖ (xₖ - x_dock)ᵀ W (xₖ - x_dock) + R uₖ²   subject to   xₖ = A xₖ₋₁ + B uₖ ,

with W = Σ*⁻¹ (from the goal prior) and R = σᵤ⁻² (from the control prior).
"""
function lqr_forces(x_now, T)
	n = length(x_now)

	# Stack the predicted states: x_{1:T} = Φ x_now + Γ u_{1:T}
	Φ = zeros(n*T, n)
	Γ = zeros(n*T, T)
	for k in 1:T
		Φ[(k-1)*n+1:k*n, :] = A^k
		for j in 1:k
			Γ[(k-1)*n+1:k*n, j] = (A^(k-j) * B)[:, 1]
		end
	end

	# Quadratic costs from the goal prior and the control prior
	W = kron(diageye(T), inv(diagm([goal_std_position^2, goal_std_velocity^2])))
	R = diageye(T) ./ control_std^2
	X_goal = repeat(x_dock, T)

	# Least-squares solution
	return (Γ'*W*Γ + R) \ (Γ'*W*(X_goal - Φ*x_now))
end

# ╔═╡ d29a1629-fe4a-58af-8ad0-f4ce4a70e74d
let
	m_now = [0.0, 0.0]

	# Bayesian control with (almost) no process noise and a known current state
	result = infer(
		model = robot_planner(
			m_now  = m_now,
			S_now  = 1e-8*diageye(2),
			A      = A,
			B      = B,
			Q      = 1e-8*diageye(2),
			m_u    = zeros(1),
			S_u    = fill(control_std^2, 1, 1),
			S_goal = diagm([goal_std_position^2, goal_std_velocity^2]),
			T      = horizon,
		),
		data = (x_goal = [x_dock for k in 1:horizon],),
	)
	u_bayes = [mean(q)[1] for q in result.posteriors[:u]]

	# Linear-quadratic optimal control
	u_lqr = lqr_forces(m_now, horizon)

	times = Δt .* (1:horizon)
	plot(times, u_lqr; color="black", linewidth=8, alpha=0.25, label="linear-quadratic control")
	plot!(times, u_bayes; color="purple", marker=:circle, markersize=3, label="Bayesian control", xlabel="time (s)", ylabel="planned force (N)")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BmlipTeachingTools = "656a7065-6f73-6c65-7465-6e646e617262"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
RxInfer = "86711068-29c9-4ff7-b620-ae75d7495b3d"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
BmlipTeachingTools = "~1.4.1"
Distributions = "~0.25.132"
Plots = "~1.41.6"
RxInfer = "~6.1.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.7"
manifest_format = "2.0"
project_hash = "21861a25f0cb434a33c88fe07c38cb82fc4e5148"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "7063ad1083578215c7c4bf410368150abe8d5524"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.45"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

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

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "b4a2d2dfcaedf8f0cb93a970c4ccba6f632f14cc"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.15.1"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AutoregressiveMessagePassingRules]]
deps = ["BayesBase", "Compat", "Distributions", "ExponentialFamily", "FastCholesky", "LinearAlgebra", "MessagePassingRulesBase", "StatsFuns"]
git-tree-sha1 = "4c9f3f4edd4580e385535bce39a6a39f8f1a6aa9"
uuid = "bbb8872a-15f9-43f6-924d-37f12d04ed53"
version = "1.0.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BayesBase]]
deps = ["Distributions", "DomainSets", "LinearAlgebra", "Random", "SpecialFunctions", "StaticArrays", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "TinyHugeNumbers"]
git-tree-sha1 = "cadec5ad37b53e1198f36524b30297fc0df98aae"
uuid = "b4ee3484-f114-42fe-b91c-797d54a0c67e"
version = "1.5.12"
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
git-tree-sha1 = "876d4c4b941cc8b079da79c88d9c2dee52b9dd13"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.10.3"

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
git-tree-sha1 = "3496e2b359c27793d12bec54ec94be7983a6ceb2"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.2"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "d4101e848e8d3f585d61d244c2fe0c80a70e6b3b"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.1.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "970758a3d591a2a5c2a907c53f2e2f8c1b1d3537"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.9"

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

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
deps = ["PrecompileTools"]
git-tree-sha1 = "7c4ea0adfb7238bf4678aa36325863fab6163f6f"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.2.2"

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
version = "1.3.1+2"

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d5e5fa1f0d6ca517edbbcd734efa93c6639405b0"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.7.1"

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
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

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

[[deps.DeltaMessagePassingRules]]
deps = ["BayesBase", "Distributions", "ExponentialFamily", "MessagePassingRulesApproximations", "MessagePassingRulesBase"]
git-tree-sha1 = "eb4eafdd796377c6dfc1e842363291c5db8525c6"
uuid = "ea5de7a7-24c9-4a61-90ce-ef77c844b13f"
version = "1.0.0"

    [deps.DeltaMessagePassingRules.extensions]
    DeltaMessagePassingRulesProjectionExt = "ExponentialFamilyProjection"

    [deps.DeltaMessagePassingRules.weakdeps]
    ExponentialFamilyProjection = "17f509fa-9a96-44ba-99b2-1c5f01f0931b"

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

[[deps.DiscreteTransitionMessagePassingRules]]
deps = ["BayesBase", "Distributions", "ExponentialFamily", "LinearAlgebra", "MessagePassingRulesBase"]
git-tree-sha1 = "1bf03a659aa103303f73a2ef40d74c206b02305a"
uuid = "039b7295-e235-4523-a933-c224bd9a7e58"
version = "1.0.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "2ddbc341251ed713d39e503b504593a5d6789551"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.132"

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

[[deps.DomainSets]]
deps = ["CompositeTypes", "FunctionMaps", "IntervalSets", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "6ac879a64381881dc24f96a15e13e80204439906"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.8.3"

    [deps.DomainSets.extensions]
    DomainSetsMakieExt = "Makie"
    DomainSetsRandomExt = "Random"
    DomainSetsRecipesBaseExt = "RecipesBase"

    [deps.DomainSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "4e468f521e1f9f86891cb07186de5df90360a666"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.12"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c307cd83373868391f3ac30b41530bc5d5d05d08"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.1+0"

[[deps.ExponentialFamily]]
deps = ["BayesBase", "BlockArrays", "Distributions", "DomainSets", "FastCholesky", "FillArrays", "ForwardDiff", "HCubature", "HypergeometricFunctions", "IntervalSets", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "PositiveFactorizations", "Random", "SparseArrays", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "TinyHugeNumbers"]
git-tree-sha1 = "22a03af5898b5f988efdd835f40af831164f7d4b"
uuid = "62312e5e-252a-4322-ace9-a5f4bf9b357b"
version = "2.7.1"

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
git-tree-sha1 = "02f0dc1afe2c0bb73a0881727a4bf771a19ec190"
uuid = "2d5283b6-8564-42b6-bb00-83ed8e915756"
version = "1.5.0"

    [deps.FastCholesky.extensions]
    ReactantExt = "Reactant"
    StaticArraysCoreExt = "StaticArraysCore"

    [deps.FastCholesky.weakdeps]
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "4916117dd032ec5959b7633aedbbac408ca5ddeb"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.3.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6621fef488e496356c9c9625d0562c12a6070819"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.20.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "417289ac840d201005d9ed16789a46defccbe36d"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.18.1"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

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
git-tree-sha1 = "3b0f72e2ffef1a139ac450725c9ecd01c0a3c050"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.4.6"
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

[[deps.GCVMessagePassingRules]]
deps = ["BayesBase", "Distributions", "ExponentialFamily", "MessagePassingRulesApproximations", "MessagePassingRulesBase", "StandardMessagePassingRules", "StatsFuns"]
git-tree-sha1 = "b9d896eff3b7dd887b119675f96d210abc31f2a9"
uuid = "2bb838d1-ae36-49a2-8c64-82a0acccfa93"
version = "1.0.0"

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

[[deps.Gamma]]
git-tree-sha1 = "86f86b6168a016ed88e4ae4e64577b98c3b59e8e"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.1.0"

[[deps.GaussianCouplingMessagePassingRules]]
deps = ["BayesBase", "Distributions", "ExponentialFamily", "MessagePassingRulesBase"]
git-tree-sha1 = "73948ad868fa022db11c678aaf1415ddb762f219"
uuid = "a57ce80c-d854-4eb3-973d-f0d0550df039"
version = "1.0.0"

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
git-tree-sha1 = "2d10297851848d1b6dc1dc335e549068e7890477"
uuid = "b3f8163a-e979-4e85-b43e-1f63d8c8b42c"
version = "4.9.0"

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
git-tree-sha1 = "dbb2b976f605b220dfe139d6db9f3859680da09e"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.15.0"

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
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "31bb6c92405c084617facc1d7ed9eb6c402d061e"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.30"

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
git-tree-sha1 = "0db6aea5b64caa1c47e9fdd27394f0296f05e8bf"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.15"

    [deps.IntervalSets.extensions]
    IntervalSetsMakieExt = "Makie"
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.InverseFunctions]]
git-tree-sha1 = "5cc6ad3e2bfc75a11eed92632ee341d37f4a4536"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.18"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "d22e8bdb3f03c300a103ff23c0dd5a654c6701de"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.7"

[[deps.JLD2]]
deps = ["ChunkCodecCore", "ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "9ce2e7c49ae4a7035b7d60db3a553b1f6c16875d"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.7"

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
git-tree-sha1 = "9939eb6f799a963b3392ac4a1d49d3fc0646eb80"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.11.0"

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

[[deps.MessagePassingRulesApproximations]]
deps = ["Compat", "FastCholesky", "FastGaussQuadrature", "ForwardDiff", "LinearAlgebra"]
git-tree-sha1 = "0904181cd2a25ed85da53a3426af17f0fb7bb193"
uuid = "36043009-14b3-4068-9fb9-f1e492c92d7d"
version = "1.0.0"

[[deps.MessagePassingRulesBase]]
deps = ["BayesBase", "Compat", "FastCholesky", "IrrationalConstants", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "386ffd59dc3b21d3d36542c41f7b0326c821813f"
uuid = "4a5b64c7-30c9-4471-82f2-bbe8464b157d"
version = "1.1.2"

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

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "676c0f1c1c34224d6869675047a54b42fdc7b8c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.5"

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
git-tree-sha1 = "5caab1056e0734b87f62ae3a3b520d47d3e994e8"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.7.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

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
git-tree-sha1 = "123266c25174ef6c8d4718920abc206452cf8de6"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.41"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

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
git-tree-sha1 = "aef1987d90dcc2e2cbd919ad10b2d25bcce4f418"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Printf"]
git-tree-sha1 = "ce5ba3edfdeba107a721748d53b5f5887f415853"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.13.0"

    [deps.ProgressMeter.extensions]
    ProgressMeterDistributedExt = "Distributed"
    ProgressMeterIJuliaExt = "IJulia"

    [deps.ProgressMeter.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

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
deps = ["BayesBase", "Compat", "Distributions", "LinearAlgebra", "MacroTools", "MessagePassingRulesBase", "Random", "Rocket", "TinyHugeNumbers", "TupleTools", "UUIDs"]
git-tree-sha1 = "944722ce6bfb8b1cdfb05a123afe9fe397fbd725"
uuid = "a194aa59-28ba-4574-a09c-4a745416d6e3"
version = "7.0.0"

    [deps.ReactiveMP.extensions]
    ReactiveMPReactantExt = "Reactant"

    [deps.ReactiveMP.weakdeps]
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

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
git-tree-sha1 = "6d40b2fe70437b01397d2a4d5b020008da4e7019"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.2+0"

[[deps.Rocket]]
deps = ["DataStructures", "Sockets", "Unrolled"]
git-tree-sha1 = "ad8f448adaaedd25803f12bcff8d59a8a1dd8bbb"
uuid = "df971d30-c9d6-4b37-b8ff-e965b2cb3a40"
version = "1.11.0"

    [deps.Rocket.extensions]
    RocketObservablesExt = "Observables"

    [deps.Rocket.weakdeps]
    Observables = "510215fc-4207-5dde-b226-833fc4488ee2"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "e6ffa7bd4f8d0b8744d18f219304738d952f9a37"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.11"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.RxInfer]]
deps = ["AutoregressiveMessagePassingRules", "Base64", "BayesBase", "DataStructures", "Dates", "DeltaMessagePassingRules", "DiscreteTransitionMessagePassingRules", "Distributions", "DomainSets", "ExponentialFamily", "FastCholesky", "GCVMessagePassingRules", "GaussianCouplingMessagePassingRules", "GraphPPL", "HTTP", "JSON", "LinearAlgebra", "Logging", "MacroTools", "MessagePassingRulesApproximations", "MessagePassingRulesBase", "PrecompileTools", "Preferences", "ProgressMeter", "Random", "ReactiveMP", "Reexport", "Rocket", "SoftDotMessagePassingRules", "SparseArrays", "StandardMessagePassingRules", "Static", "Statistics", "TupleTools", "UUIDs"]
git-tree-sha1 = "cbb0595f742056ccab20beba5cf2a1253294c6dd"
uuid = "86711068-29c9-4ff7-b620-ae75d7495b3d"
version = "6.1.0"

    [deps.RxInfer.extensions]
    OptimExt = "Optim"
    PrettyTablesExt = "PrettyTables"
    ProjectionExt = "ExponentialFamilyProjection"
    ReactantExt = "Reactant"
    TensorBoardLoggerExt = "TensorBoardLogger"

    [deps.RxInfer.weakdeps]
    ExponentialFamilyProjection = "17f509fa-9a96-44ba-99b2-1c5f01f0931b"
    Optim = "429524aa-4258-5aef-a3af-852621145aeb"
    PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    TensorBoardLogger = "899adc3e-224a-11e9-021f-63837185c80f"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLPublic]]
deps = ["PrecompileTools"]
git-tree-sha1 = "74685afb51732a464fbce79a72708f7c4203ceb7"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.3.0"

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

[[deps.SoftDotMessagePassingRules]]
deps = ["BayesBase", "Distributions", "ExponentialFamily", "LinearAlgebra", "MessagePassingRulesBase", "StatsFuns"]
git-tree-sha1 = "8ccebfa9b93d9908178cc0e97c4f4a480708740d"
uuid = "fdfab25d-afbd-4817-865a-cd280c067f5e"
version = "1.0.0"

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
git-tree-sha1 = "429071b23f4c9a13fb6582f807cc2ef454082408"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StandardMessagePassingRules]]
deps = ["BayesBase", "Compat", "Distributions", "DomainSets", "ExponentialFamily", "FastCholesky", "LinearAlgebra", "LogExpFunctions", "MatrixCorrectionTools", "MessagePassingRulesApproximations", "MessagePassingRulesBase", "SpecialFunctions", "StatsFuns"]
git-tree-sha1 = "ca7e44b8d9c49483dfab3fc5e91346d032cb7d45"
uuid = "d166223a-7879-43b1-add3-64af0a178e9e"
version = "1.0.0"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "474a5283ad435618090122872eea6a8165ea6bcf"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.4.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "8a8a52f887f38d44014e24ee46b8a7311c625f6d"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.24"

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
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
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
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "0aa97471d55945556e8d2859568b8317cef98351"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.3"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ad435e30854ed0f50f4b7f67edd71664afecc5d8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.10.0"

    [deps.StructUtils.extensions]
    StructUtilsDataFramesExt = ["DataFrames"]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazilyInitializedFields = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
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
git-tree-sha1 = "d5b24fd37550762d78280fb37c1800c3ea7e95cb"
uuid = "783c9a47-75a3-44ac-a16b-f1ab7b3acf04"
version = "1.0.4"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TupleTools]]
git-tree-sha1 = "a1c7e924f746f0ebc792425e7d4d2ea5b4470697"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.6.1"

[[deps.URIs]]
deps = ["Serialization"]
git-tree-sha1 = "0392e5e3c5327b90a487b0ff3e818c4aecd3660b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.1"

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
# ╟─f4618cf4-1940-520c-88d8-2a1d79f76127
# ╟─490f293f-4996-5b82-9524-4ec329e9ef93
# ╟─5eb11e24-f3e7-53f0-be55-cc187d2c2f99
# ╠═109f3a18-7a85-5225-abb3-036a7d6f5319
# ╟─08a90d11-2a35-560e-980a-5981a0ce119b
# ╟─1dcf887d-bcf8-5601-83b7-216b388a519d
# ╟─5b157b82-2f20-5344-a985-847ba5da3178
# ╟─f803e8b1-8fed-5bf5-98a7-1e7988bd8905
# ╠═bf6d4247-0ac3-59d6-9b2a-b1020d688fb1
# ╟─21a58e3e-a745-5561-bb57-b7be6efb5fa7
# ╟─76066720-ec4a-5b75-9f1a-8009865bfa99
# ╠═dcabac88-b287-575f-871b-230829feff3e
# ╟─6b21a578-986f-54b4-aed2-e7c6558a135c
# ╠═d0df0199-b1df-51e9-840a-524ab8a3290a
# ╟─8893d502-0c43-593e-99b9-c7c66fd7598c
# ╟─f1cfa47a-ae48-5bb1-86f0-8d49578f825a
# ╠═ed2e3c11-c405-598f-84d1-9f5b168f6d1e
# ╟─58819fa7-b1d9-54f8-8b55-d99cd9aaaa49
# ╠═169f73d5-55b8-5a10-9928-febbe5f8c6ea
# ╟─b39ae303-a54a-5b39-8594-865102bf88c4
# ╠═4689e861-d70d-5d3e-82e6-fc5255fbf4f0
# ╟─99966210-853f-5640-8ad5-f3f307aae6d6
# ╠═2613ddd8-a146-5e6b-b54c-cfaa2179c20f
# ╟─a1c34b56-5424-5bf9-8bd7-1fe05459f090
# ╟─247a905f-8ad7-5bb4-b5f3-fe0e1c405cfb
# ╠═7b20c8f9-95b3-4661-84ec-969d4a990824
# ╠═4db0ebda-7d89-53d9-b860-dfc528b990ab
# ╟─3ee7ec5c-a11c-556f-825f-34af544b461e
# ╟─18b3356a-9530-5e5b-ad18-59e8c185c5f0
# ╟─9fb427b6-263a-560e-9e35-89990b77530c
# ╟─f29ac4aa-8122-5d8a-9b00-c59467d659e1
# ╟─03f8e3dc-b619-52a2-a89c-549e0b6ebcf7
# ╠═e30bfe40-abaa-53da-bf2c-214be145a48b
# ╠═37da6c61-47cd-5efe-a7f5-faea068a83c4
# ╟─e1954792-9742-57b9-90ee-6178bcf87aaf
# ╟─fb3b4276-3784-538b-9ddb-ce93a77c7716
# ╟─18f16a73-8009-5da4-a5fe-7991279223ce
# ╟─1208952c-581b-5cec-b8bb-ecdb8610e607
# ╟─0b5382a0-da1c-5b52-a42c-43ec91119188
# ╟─3fce0f5a-bbcd-5438-a4d2-8b3ad492c72e
# ╟─a8ee6a06-3b56-55f5-8f12-cfb92d44103e
# ╟─355f35de-61b8-507b-8e13-8493ea0c2d98
# ╟─78680031-c66c-5e47-beed-d23baf55148d
# ╟─a6a8b6cc-a6fb-550b-8c25-f4a243ad0607
# ╟─7af487a5-a345-50f8-b313-8a7b472a923a
# ╟─b43ecfd4-36d7-5b8e-8e22-148ec2219996
# ╟─20ca5627-5927-53de-9537-529577e48142
# ╠═5e01c014-cf56-53d6-b8ca-9ba29e774800
# ╟─949750b6-27e4-5f00-b7b0-5ce9d416324c
# ╟─160a9e12-e432-5ba0-9c3e-6998a3f2ff39
# ╟─a9567b09-6cd2-5640-925e-3f914246ead8
# ╟─b242f2e6-8536-5be9-9443-30a8a05fd0a4
# ╟─7aa4fb0a-4cfd-590e-8ead-869739113c94
# ╠═782a35af-56da-57fd-bdc3-8b04a02ca15f
# ╟─10eb6485-58c3-57d9-8e88-c672f15e1266
# ╟─60ccc0eb-d4d4-5b88-9edc-dee0811cee3d
# ╠═e9fb117b-e5ac-510f-aa2c-cbb69ea89364
# ╠═5fae9622-a1bd-5010-a914-de09783d39d0
# ╟─19dba654-d3ec-50bf-8fe6-d4330b080d1e
# ╠═7a69a7e1-d1be-587a-8c09-cfddc02a60fa
# ╟─5189d002-4858-5d54-b1d7-24bed3c32df1
# ╠═81a2cfb2-dd5c-5ed8-bbf9-0fd7dfa3f489
# ╠═238f441b-236a-538e-a1c6-9a405ba5690e
# ╟─4f444f67-f8ac-5b63-9faf-a299cfeb824e
# ╟─19d547a3-78da-503a-9337-7be7748c8ea4
# ╟─46cb62ea-94ed-5bf8-afab-b7661d90a21e
# ╟─a620f32e-e26b-54c6-be70-b1a12788e3e8
# ╟─57a70848-2552-5abc-a804-a36f17211c62
# ╠═6b05cbc4-f3b8-53e6-909d-0979e7efbd56
# ╠═962aaa17-1c69-59a2-bb18-8347c3336590
# ╠═a21cf15e-e22b-57b9-8de6-83916e2f3b42
# ╟─ca35e9ed-85d6-54ba-a59e-6a6dd42e5a99
# ╟─e43f046f-cd46-5d21-89a4-4367de89af57
# ╟─992f259b-7f5c-5258-a49e-3c4ad88a2760
# ╟─567f51bd-8e31-58b0-acab-da3023ac03c3
# ╟─ac13794a-1077-52b6-8e73-f56f0ccfd735
# ╟─2695186f-15c3-53ad-b399-bcf1c6e34d40
# ╟─47ee63fe-9266-5443-99b8-a088e4ab43d6
# ╟─7da0c20e-bb2a-5219-a1a3-7b02b6e57b09
# ╟─4cfca320-5a9c-5fd4-91cf-05930d3a3842
# ╟─416933d7-e4eb-55a8-88ed-607006e2951c
# ╟─344f7abb-8a57-5756-920d-04b6ebdbabfc
# ╟─f3a70297-518f-5533-95ad-71e1a05afc49
# ╟─b345a533-f09b-54d9-965e-ce53b25d6139
# ╟─55f1e530-af0e-5df8-bbb3-bcd3d9da68fc
# ╟─b1157d47-a80e-5e7f-8f36-6de81880e52a
# ╠═8a05d41d-b09f-5602-8220-bcca5f2a6878
# ╟─2607c35f-9632-59c5-b0c8-753ea3a9cd21
# ╟─fb51c8cd-958b-52b8-bc26-328337ad8c44
# ╟─75b92f65-6746-4f83-8669-09c3fc7a9393
# ╟─69df80d0-a05b-49f5-a576-fe3ae8b629f3
# ╟─09279d95-27f2-4eaa-9267-6b9209f60632
# ╟─d6c9a2b1-e636-409a-a9a2-7e5389bcb4e6
# ╠═cc583d4c-cc9c-4bcd-a29d-6abfe12d7072
# ╟─61a5c9b4-8e5e-4b6a-8e0f-d3ff0866d318
# ╟─07a717c9-5ec2-43cf-b0a3-17691a3ba5de
# ╠═4eec31d0-a8f4-49e1-8f88-42c1d1b94188
# ╠═5290535c-3444-481f-a0bc-760f8230abe8
# ╟─e4e2c95f-5484-4e90-afec-ada06ff8bb8c
# ╠═8c873a95-c5a2-484d-ade3-bc129515438a
# ╠═61613276-a1dd-4fc6-8bd4-423f3cce7cdc
# ╟─2216a8fb-81d9-4376-8a0a-5c1dfabdf66d
# ╠═3068b8c8-a397-4bd8-9f81-83d5062f3aca
# ╠═bb9b2685-375b-4d5a-91b6-dbe83d2feeb1
# ╠═e29f44d3-0b71-444e-bb76-80207694ccf6
# ╟─b1626647-7841-4620-982b-b352d78c1770
# ╟─9b9dcb16-6321-4937-9d81-edd9bc6f156d
# ╟─f54dceb2-0956-4c04-ac9e-10c638850668
# ╟─4d863121-0381-4af0-a231-bd404060cd0f
# ╟─c12907ed-9c2a-46f2-9962-0be558d86a3e
# ╟─8e7317f4-d17d-4137-b419-460d27c6778a
# ╟─5301881d-f47c-48df-8345-25b2267688e4
# ╟─8c721823-37c3-4316-a090-bb7a8ac36b7b
# ╟─29d0b601-2d6d-48f8-ac2f-eb1f6e75a51f
# ╟─2753fe1b-48c4-4b66-bde2-e5d067bbdaca
# ╟─8a77b17b-de60-465a-b7f7-cecf2a9ad21c
# ╟─cc0c9e01-9de6-42e0-a03d-509988c4f468
# ╟─e0278cd0-3280-4a3f-9fa3-93e753a95585
# ╟─c8706bc6-4133-4bb2-a6c8-1c1fe5b7b56c
# ╟─25802cd8-bca5-4813-b5c4-6836ee0c84cb
# ╟─ccfd1039-4346-4ff8-a670-e8e0a2857aa8
# ╟─cce2a7a0-abc0-5d36-87f6-c54391429e78
# ╟─c787491a-252f-5bd7-9860-1a6aadf205de
# ╟─889c61d3-3173-5f7c-aac1-71a245e0f752
# ╟─90d3f2cd-d510-5694-8c39-62e1635bdf4c
# ╟─c446cf9f-12c2-4a6f-a0e1-0cd0bb4c3f46
# ╟─f28dc0f1-5366-5395-869c-75a260e98aa9
# ╟─ade5e467-96f1-5730-9796-c3eddc09e001
# ╟─e4876603-a1f0-54db-a40c-ceed18b6c9dd
# ╟─bbfaf3f4-cd13-5bc5-b593-fbac3dd1afcd
# ╟─cc6f1751-897d-582f-8b2a-ef3764823f09
# ╟─67dc62b5-f28a-4b31-b2a1-640a7c0d2c80
# ╟─8d91235c-3349-4562-a305-2465516e8af8
# ╟─ec4514f4-7df5-5634-8f39-9685615df687
# ╠═d29a1629-fe4a-58af-8ad0-f4ce4a70e74d
# ╟─b85beb41-21d4-5d5f-b620-12474f3e4325
# ╟─425b4a40-bcc3-5a94-ac12-caa6c26c9bfe
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
