### A Pluto.jl notebook ###
# v0.20.10

#> [frontmatter]
#> image = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/P%C3%A4ij%C3%A4nne_and_p%C3%A4ij%C3%A4tsalo.jpg/960px-P%C3%A4ij%C3%A4nne_and_p%C3%A4ij%C3%A4tsalo.jpg"
#> title = "Mini: How to use this knowledge?"
#> description = "What can Bayesian machine learning be used for after this course?"

using Markdown
using InteractiveUtils

# ╔═╡ f9b0981e-ea2b-4efd-aa71-cf9ac64e8100
md"""
*(note to self: let's format this a bit more nicely)*
"""

# ╔═╡ 66e17d86-e7a0-11ef-2a6c-c5394ba61921
md"""
# Where to apply our newfound knowledge?

Question from 2025:

> This course was a great introduction to the Bayesian methods and way of thinking, in my opinion, and I feel converted to the Bayesian cult (in a good way). What I’m only left wondering is if these skills can be applied ubiquitously. Is there a reason to not use the Bayesian approach when encountering a statistics or modelling problem? Or would you recommend everyone to try their hand at this in perhaps their next report/thesis? Additionally, is RxInfer the way to go in most cases, or is it recommended for those interested to also familiarise themselves with other options such as Turing.jl?
> 
> Thanks for the classes! I’ll likely never look at a problem without thinking about a Bayesian approach again.



"""

# ╔═╡ 3098e2c2-5b68-46a1-b87c-ccbb52139095
md"""
# Answer

Hi Borre –

I’m glad you liked the course, and thanks for the question.

Indeed, the Bayesian approach provides both a concrete method to solve information processing problems, and also (at a higher abstraction level) a quantitative philosophy for the entire scientific approach. I will answer your questions below:

**Should you use the Bayesian approach in your next assignment?**

Almost any information processing problem, for instance, in control, signal processing, video processing, digital communications, etc., can be phrased as an inference task on a generative model. So, if you are working in an information processing field, then, yes, in principle, the Bayesian approach, as discussed in the class, applies to your next assignments (e.g., traineeship, graduation project), and, in my opinion, it is the most principled approach to attack your problem. Therefore, at minimum, you should try to formulate the Bayesian approach for your next assignment.

Should you also religiously stick to working out your assignment through Bayesian methods? I think this issue depends on multiple criteria, and the decision may fall either way, depending on the context. For instance, are there good software tools available for your problem? Since almost all investments have gone to non-Bayesian AI, there are great software tools for non-Bayesian AI, while Bayesian software (probabilistic programming toolboxes) is relatively underdeveloped. There are other reasons that you may think about. Is data expensive in your problem? If it is costly, then you want good experimental design and proper (Bayesian) learning from your data. If data is cheap, then non-Bayesian training on a very large database may potentially solve your problem cheaper, so it would be a good engineering decision to go non-Bayesian. Another reason may be based on your environment. Are you working in a team, and are you the only person who understands the Bayesian approach? Then you may have to shift a bit to accommodate collaboration.

**Should you use RxInfer?**

I think reactive message passing is the “right” way to implement (variational) Bayesian inference, especially in a real-time processing situation. However, other toolboxes, such as Turing, can be used for inference in a more extensive range of models since they use a generic black-box approach to inference (like Monte Carlo sampling). Of course, Python also has a variety of PP toolboxes. Which tool is best depends on the demands of your application. Do you need real-time processing? Then consider RxInfer. Are you OK with a longer wait but a more straightforward programming effort? Then consider Turing or other alternatives. In short, the different tools have different advantages, and you should use the tool that best suits your application.

**General advice**

As a general trend, the Bayesian approach is increasing in popularity, and the toolboxes are improving. In my experience, formulating your problem as a Bayesian processing task is extremely useful in almost any assignment. I highly recommend that you continue challenging yourself to think the Bayesian way. Develop yourself as a Bayesian scientist and, simultaneously, as a practical engineer. Describe your assignments as a Bayesian, and then consider the practical context (tools, data, team, etc.) to decide how to solve your assignment concretely.

**Just a final note on the Bayesian approach in the literature**

I’ve read a lot of engineering papers on solving problems in control, communications etc. Most of the time, I get lost in the paper, even if the authors consider using probabilistic methods. Usually, the way for me to understand what’s going on is to collect the equations that describe the model and separate the model specification from inference issues. Many authors do not describe a complete model and start working on inference before they have specified all model assumptions. I recommend that you separate the model specification from inference issues when you read papers and also when you write your papers.

All the best, and thanks for the question,

–Bert
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.9"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

[deps]
"""

# ╔═╡ Cell order:
# ╟─f9b0981e-ea2b-4efd-aa71-cf9ac64e8100
# ╟─66e17d86-e7a0-11ef-2a6c-c5394ba61921
# ╟─3098e2c2-5b68-46a1-b87c-ccbb52139095
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
