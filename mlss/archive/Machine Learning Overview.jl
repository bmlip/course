### A Pluto.jl notebook ###
# v0.20.19

#> [frontmatter]
#> image = "https://github.com/bmlip/course/blob/v2/assets/figures/scientific-inquiry-loop.png?raw=true"
#> description = "What type of problem can be solved using Machine Learning? Can we apply the Bayesian approach?"
#> 
#>     [[frontmatter.author]]
#>     name = "BMLIP"
#>     url = "https://github.com/bmlip"

using Markdown
using InteractiveUtils

# ╔═╡ a5d43e01-8f73-4c48-b565-f10eb807a9ab
using BmlipTeachingTools

# ╔═╡ 3ceb490e-d294-11ef-1883-a50aadd2d519
title("Machine Learning Overview")

# ╔═╡ d7d20de9-53c6-4e30-a1bd-874fca52f017
PlutoUI.TableOfContents()

# ╔═╡ 3cebc804-d294-11ef-32bd-29507524ddb2
md"""
## Preliminaries

##### Goal

  * Top-level overview of machine learning

##### Materials

  * Mandatory  

      * this notebook
  * Optional

      * Study Bishop pp. 1-4

"""

# ╔═╡ 3cebf2d4-d294-11ef-1fde-bf03ecfb9b99
md"""
## What is Machine Learning?

Machine Learning relates to **building models from data and using these models in applications**.

"""

# ╔═╡ 3cec06e6-d294-11ef-3359-5740f25965da
md"""
##### Problem

 - Suppose we want to develop an algorithm for a complex process about which we have little knowledge (so hand-programming is not possible).

"""

# ╔═╡ 3cec1032-d294-11ef-1b9d-237c491b2eb2
md"""
##### Solution

 - Get the computer to develop the algorithm by itself by showing it examples of the behavior that we want.

"""

# ╔═╡ 3cec1832-d294-11ef-1317-07fe5c4e69c2
md"""
Practically, we choose a library of models, and write a program that picks a model and tunes it to fit the data.

"""

# ╔═╡ 3cec20f4-d294-11ef-1012-c19579a786e4
md"""
This field is known in various scientific communities with slight variations under different names such as machine learning, statistical inference, system identification, data mining, source coding, data compression, data science, etc.

"""

# ╔═╡ 3cec3062-d294-11ef-3dd6-bfc5588bdf1f
md"""
## Machine Learning and the Scientific Method


The **scientific method** (or scientific inquiry loop) is a systematic approach for building models of the world. It comprises three stages:

  1.	**Hypothesis formulation / Experimental ("Trial") design** – a trial is designed and executed, based on an analysis of the uncertainties in the current model.

  2.	**Observation / Data collection** – the world’s response to the trial is measured in the form of new observations.

  3.	**Analysis / Model updating** – the model is revised in light of the observations.

The cycle repeats, incrementally improving the model and our understanding. In an engineering context, the model can be used for various applications, such as predicting the future, or recognizing observations as objects.

![](https://github.com/bmlip/course/blob/v2/assets/figures/scientific-inquiry-loop.png?raw=true)


Machine learning can be viewed analogously, as the process of constructing models of the world. While one may debate whether the experimental design stage is part of the machine learning field, the model updating stage, driven by observations, clearly is. This explains why machine learning methods are applied so widely across the sciences and engineering disciplines.

In this course, we will revisit this scientific inquiry loop and progressively annotate it with the mathematical formulas that underlie each stage of the scientific method.

"""

# ╔═╡ a3ae8443-0100-41f7-a91a-c7d128064b88
keyconcept("",
"Machine learning is about building models of the environment and is therefore an integral part of the scientific inquiry loop. Consequently, we see machine learning applied across the sciences, engineering, and society at large.")

# ╔═╡ 3cec43d4-d294-11ef-0a9f-43eb506527a6
md"""
## Machine Learning is Difficult

##### Modeling (Learning) Problems

  * Is there any regularity in the data anyway?
  * What is our prior knowledge and how to express it mathematically?
  * How to pick the model library?
  * How to tune the models to the data?
  * How to measure the generalization performance?

"""

# ╔═╡ 3cec5b96-d294-11ef-39e0-15e93768d2b1
md"""
##### Quality of Observed Data

  * Not enough data
  * Too much data?
  * Available data may be messy (measurement noise, missing data points, outliers)

"""

# ╔═╡ 3cec86cc-d294-11ef-267d-7743fd241c64
md"""
# A Machine Learning Taxonomy

Machine learning methods can be grouped, in broad terms, into four categories:

 - Supervised learning

 - Unsupervised learning

 - Trial design / decision-making

 - Other frameworks
   - Other stuff, like preference learning, learning to rank, etc., can often be (re-)formulated as special cases of either a supervised, unsupervised, or trial design problem.

We will briefly introduce these categories below and, in the forthcoming lectures, expand on how each can be framed within the Bayesian machine learning perspective.

![](https://github.com/bmlip/course/blob/v2/assets/figures/ml-taxonomy.png?raw=true)


"""

# ╔═╡ a66c3b31-0f1f-41ff-b19c-0926ea1c2ff5
keyconcept("",
"The three major paradigms of machine learning are supervised learning, unsupervised learning, and trial design. Most other applications can be reframed within one of these three frameworks." )

# ╔═╡ 6f95adeb-d0a9-47fd-900a-0e55d497bab6
md"""
## Supervised Learning
"""

# ╔═╡ 3ced0d0c-d294-11ef-3000-7b63362a2351
md"""

**Supervised learning** is about learning functions. [Functions describe the world!!](https://youtu.be/BWZTlfrneD8?si=FNhUu7QH9O9xx2Bp)

In supervised learning, we are given observations of desired input–output behavior,

```math
D = \{(x_1, y_1), \dots, (x_N, y_N)\},
```

where ``x_n`` are inputs and ``y_n`` are the corresponding outputs. The goal is to estimate the conditional probability distribution ``p(y_n | x_n)``, i.e., to capture how ``y_n`` depends on ``x_n``. The term "supervised" reflects the fact that the correct outputs ``y_n`` are provided in the training dataset ``D``. (The reasons for using probabilities will be discussed in the [Probability Theory lecture](https://bmlip.github.io/course/lectures/Probability%20Theory%20Review.html).)

Generally, we distinguish between **classification** and **regression** as two different supervised learning problems. 

"""

# ╔═╡ e1122eab-a25b-4441-a053-b0121b334731
md"""
#### Classification

In a classification problem, the target variable ``y`` is a *discrete-valued* vector representing class labels.  

The special case ``y \in \{\text{true},\text{false}\}`` is called **detection**. 


![](https://imgur.com/XSCNBN9.png)

"""

# ╔═╡ b00872c2-96d0-47e7-8495-a3d6e559ea63
NotebookCard("https://bmlip.github.io/course/lectures/Generative%20Classification.html"; link_text="Go to lecture")

# ╔═╡ 3ced29ae-d294-11ef-158b-09fcdaa47d1c
md"""
#### Regression

Regression, also called **curve fitting**, is the supervised learning task of estimating the conditional distribution ``p(y_n | x_n)``, where  ``x_n`` are input variables and ``y_n`` represent __continuous__ output variables. 

![](https://imgur.com/lKUUjWr.png)

"""

# ╔═╡ 672c35c0-c7ab-4e17-a280-867bf3cf2f27
NotebookCard("https://bmlip.github.io/course/lectures/Regression.html"; link_text="Go to lecture")

# ╔═╡ 3cec9250-d294-11ef-01ac-9d94676a65a3
md"""
## Unsupervised Learning

In the unsupervised learning setting, we are given a data set

```math
D=\{x_1,\ldots,x_N\}\,.
```

The task is to model the unconditional probability distribution ``p(x_n)``. The absence of target variables ``y_n`` in the dataset gives rise to the term unsupervised.

Because no targets are provided, unsupervised learning problems are generally considered more challenging than supervised ones. As in supervised learning, however, we can distinguish two main types of tasks: **clustering** (discovering structure in the data) and **compression** (learning efficient representations of the data).


"""

# ╔═╡ c5bfab8f-4985-420a-b46e-b4ff6d359d3d
md"""
#### Clustering

If the unobserved target variables take on discrete values, the task is referred to as clustering. In this sense, clustering can be regarded as "unsupervised classification".

![](https://imgur.com/yIvpvD6.png)
"""

# ╔═╡ bb485cc3-02e2-4cb4-9e4d-80574b8eb66c
NotebookCard("https://bmlip.github.io/course/lectures/Latent%20Variable%20Models%20and%20VB.html"; link_text="Go to lecture")

# ╔═╡ 3ced567c-d294-11ef-2657-df20e23a00fa
md"""
#### Compression 

In contrast, if the unobserved target variables are continuously valued, the task is referred to as **compression**. For example, compressing an image ``x`` produces an output ``y`` that is much smaller in size than the original. The objective is to learn a mapping ``y = f(x)`` (the encoder) such that the inverse mapping ``\hat{x} = g(y) \approx f^{-1}(y)`` reconstructs ``\hat{x}`` as close as possible (ideally identical) to the original input ``x``.

Compression can be interpreted as ''unsupervised regression''.

![](https://github.com/bmlip/course/blob/v2/assets/figures/fig-compression-example.png?raw=true)

In this lecture series, we unfortunately do not have enough time to discuss compression in detail in a separate lecture. [Chapter 12 in Bishop (2006)](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf#page=579) contains a nice introduction to compression. 

"""

# ╔═╡ 0d6029ef-87ba-4881-b7af-9de2dad1ed99
md"""
## Trial Design
"""

# ╔═╡ 3cecbc46-d294-11ef-24cb-2d9e41fb35d9
md"""

**Trial design** concerns learning which actions (trials) to perform in order to gain information about the environment and/or to achieve certain specific goals (such as crossing a street). In the broader literature, this idea is presented under various related labels, including **experimental design**, **active learning**, **decision-making under uncertainty**, **sequential decision-making**, **hypothesis testing**, **policy learning**, **planning**, and **control**. The sheer number of terms (often differing only in nuance) underscores the central importance of this task within the scientific inquiry process.

In trial design problems, the model is not only a description of the environment but also acts upon it, thereby influencing which data will be observed in the future. Such systems are called "agents". In addition to the labels mentioned above, the term [Agentic AI](https://en.wikipedia.org/wiki/Agentic_AI) has recently gained popularity.

In the machine learning and AI community, two prominent approaches to trial design are reinforcement learning and active inference:

 - **Reinforcement Learning**: Given an observed sequence of input signals and (occasionally observed) rewards for those inputs, *learn* to select actions that maximize *expected* future rewards.

 - **Active inference**: Given an observed sequence of input signals and a prior probability distribution about future observations, *learn* to select actions that minimize *expected* prediction errors (i.e., minimize actual minus predicted sensation).  


"""

# ╔═╡ 5971fb3c-1489-4e66-a77a-2c6e9714b8a2
Resource("https://github.com/bmlip/course/raw/refs/heads/main/assets/figures/minigrid%20loop.mp4", :autoplay => true, :loop=>true)

# ╔═╡ d2f07ece-5cea-4c00-8ed8-a70752e113b7
NotebookCard("https://bmlip.github.io/course/lectures/Intelligent%20Agents%20and%20Active%20Inference.html"; link_text="Go to lecture")

# ╔═╡ 3ced839a-d294-11ef-3dd0-1f8c5ef11b75
md"""
## $(HTML("<span id='some-ml-apps'>Some Machine Learning Applications</span>"))

- computer speech recognition, speaker recognition

- face recognition, iris identification

- printed and handwritten text parsing

- financial prediction, outlier detection (credit-card fraud)

- user preference modeling (amazon); modeling of human perception

- modeling of the web (google)

- machine translation

- medical expert systems for disease diagnosis (e.g., mammogram)

- strategic games (chess, go, backgammon), self-driving cars

In summary, **any 'knowledge-poor' but 'data-rich' problem**

"""

# ╔═╡ 438981cd-8450-4678-8b61-9cc5c0c0ebf1
md"""
# Summary
"""

# ╔═╡ f47f4370-4d6f-4e36-bbef-63f806130dbe
keyconceptsummary()

# ╔═╡ 86a2a956-fee6-4306-b8cb-f4b977ad3dbd
md"""
# Code
"""

# ╔═╡ fa1d5123-db02-4fda-93d2-3e5e2efed515
html"""
<style>
pluto-output img {
	background: white;
	border-radius: 3px;
}
</style>
"""

# ╔═╡ 3ced947a-d294-11ef-0403-512f2407a2d2
md"""

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BmlipTeachingTools = "656a7065-6f73-6c65-7465-6e646e617262"

[compat]
BmlipTeachingTools = "~1.3.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "3e0db0a10f1d7687b8c53fc91306ce22ead0cdba"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BmlipTeachingTools]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "PlutoTeachingTools", "PlutoUI", "Reexport"]
git-tree-sha1 = "806eadb642467b05f9d930f0d127f1e6fa5130f0"
uuid = "656a7065-6f73-6c65-7465-6e646e617262"
version = "1.3.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

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

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

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

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.5.20"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

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

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

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

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

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

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

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
# ╟─3ceb490e-d294-11ef-1883-a50aadd2d519
# ╟─d7d20de9-53c6-4e30-a1bd-874fca52f017
# ╟─3cebc804-d294-11ef-32bd-29507524ddb2
# ╟─3cebf2d4-d294-11ef-1fde-bf03ecfb9b99
# ╟─3cec06e6-d294-11ef-3359-5740f25965da
# ╟─3cec1032-d294-11ef-1b9d-237c491b2eb2
# ╟─3cec1832-d294-11ef-1317-07fe5c4e69c2
# ╟─3cec20f4-d294-11ef-1012-c19579a786e4
# ╟─3cec3062-d294-11ef-3dd6-bfc5588bdf1f
# ╟─a3ae8443-0100-41f7-a91a-c7d128064b88
# ╟─3cec43d4-d294-11ef-0a9f-43eb506527a6
# ╟─3cec5b96-d294-11ef-39e0-15e93768d2b1
# ╟─3cec86cc-d294-11ef-267d-7743fd241c64
# ╟─a66c3b31-0f1f-41ff-b19c-0926ea1c2ff5
# ╟─6f95adeb-d0a9-47fd-900a-0e55d497bab6
# ╟─3ced0d0c-d294-11ef-3000-7b63362a2351
# ╟─e1122eab-a25b-4441-a053-b0121b334731
# ╟─b00872c2-96d0-47e7-8495-a3d6e559ea63
# ╟─3ced29ae-d294-11ef-158b-09fcdaa47d1c
# ╟─672c35c0-c7ab-4e17-a280-867bf3cf2f27
# ╟─3cec9250-d294-11ef-01ac-9d94676a65a3
# ╟─c5bfab8f-4985-420a-b46e-b4ff6d359d3d
# ╟─bb485cc3-02e2-4cb4-9e4d-80574b8eb66c
# ╟─3ced567c-d294-11ef-2657-df20e23a00fa
# ╟─0d6029ef-87ba-4881-b7af-9de2dad1ed99
# ╟─3cecbc46-d294-11ef-24cb-2d9e41fb35d9
# ╟─5971fb3c-1489-4e66-a77a-2c6e9714b8a2
# ╟─d2f07ece-5cea-4c00-8ed8-a70752e113b7
# ╟─3ced839a-d294-11ef-3dd0-1f8c5ef11b75
# ╟─438981cd-8450-4678-8b61-9cc5c0c0ebf1
# ╟─f47f4370-4d6f-4e36-bbef-63f806130dbe
# ╟─86a2a956-fee6-4306-b8cb-f4b977ad3dbd
# ╟─a5d43e01-8f73-4c48-b565-f10eb807a9ab
# ╟─fa1d5123-db02-4fda-93d2-3e5e2efed515
# ╟─3ced947a-d294-11ef-0403-512f2407a2d2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
