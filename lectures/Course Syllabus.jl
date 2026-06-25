### A Pluto.jl notebook ###
# v0.20.28

#> [frontmatter]
#> description = "Course Syllabus"
#> 
#>     [[frontmatter.author]]
#>     name = "BMLIP"
#>     url = "https://github.com/bmlip"

using Markdown
using InteractiveUtils

# ╔═╡ f96d047f-9efa-4889-8b4e-a8d96677d072
using BmlipTeachingTools

# ╔═╡ 0cfd4bc0-d294-11ef-3537-630954a9dd27
title("5SSD0 Course Syllabus")

# ╔═╡ 467c8189-b5d3-4eaf-8886-6ae53136dd8f
PlutoUI.TableOfContents()

# ╔═╡ 0cffef7e-d294-11ef-3dd5-1fd862260b70
md"""
## Learning Goals

This course provides an introduction to Bayesian machine learning and information processing systems. The Bayesian approach affords a unified and consistent treatment of many useful information processing systems. 

Upon successful completion of the course, students should be able to:

  * understand the essence of the Bayesian approach to information processing.
  * specify a solution to an information processing problem as a Bayesian inference task on a probabilistic model.
  * design a probabilistic model by a specifying a likelihood function and prior distribution;
  * Code the solution in a probabilistic programming package.
  * execute the Bayesian inference task either analytically or approximately.
  * evaluate the resulting solution by examination of Bayesian evidence.
  * be aware of the properties of commonly used probability distribitions such as the Gaussian, Gamma and multinomial distribution; models such as hidden Markov models and Gaussian mixture models; and inference methods such as the Laplace approximation, variational Bayes and message passing in a factor graph.

"""

# ╔═╡ 0d013750-d294-11ef-333c-d9eb7578fab2
md"""
## Entrance Requirements (pre-knowledge)

Undergraduate courses in Linear Algebra and Probability Theory (or Statistics). 

Some scientific programming experience, eg in MATLAB or Python. In this class, we use the [Julia](https://julialang.org/) programming language, which has a similar syntax to MATLAB, but is (close to) as fast as C. 

"""

# ╔═╡ 0d0142b6-d294-11ef-0297-e5bb923ad942
md"""
## Important Links

Please bookmark the following three websites:

1. The course homepage [https://bmlip.nl](https://bmlip.nl) contains links to all materials, such as lecture notes and video lectures.
2. The [Piazza course site](https://piazza.com/tue.nl/winter2026/5ssd0/home) will be used for Q&A and communication.
3. The [Canvas course site](https://canvas.tue.nl/courses/33478) will be sparingly used for communication (mostly by ESA staff)

"""

# ╔═╡ 0d015ab4-d294-11ef-2e53-5339062c435c
md"""
## Materials

All materials can be accessed from the [course homepage](https://bmlip.nl). The materials consist of the following resources:

##### Mandatory materials for the exam

 * Lecture notes
 * Probabilistic Programming (PP) notes
 * The lecture notes and probabilistic programming notes contain the mandatory materials. Some lecture notes are extended by a reading assignment, see the first cell ("Preliminaries") in the lecture notes. These reading assignments are also part of the mandatory materials.
  
##### Optional materials to help understand the lectures and PP notes

 * video recordings of the Q2-2023/24 lecture series
 * Q&A at Piazza
 * practice exams
 * In the lecture notes, slides that are not required for the exam are moved to the end of the notes in the **Optional Slides** section.


Source materials are available at GitHub repo at [https://github.com/bmlip/course](https://github.com/bmlip/course). If you spot an error in the materials, please raise an issue at Piazza.  

"""

# ╔═╡ ab61d2fe-312c-4aca-9029-e446aaf2bfa2
keyconcept("",
md""" All study materials are accessible at the course homepage [`https://bmlip.nl`](https://bmlip.nl).""")

# ╔═╡ 0d016cf8-d294-11ef-0c84-336979a02dd7
md"""
## Study Guide

1. **Please study the lecture notes BEFORE you come to class!!**
   - Optionally, you can view the video recordings of the Q2-2023/24 lecture series for additional explanations. 

2. Then come to the class!
   - During the scheduled classroom meetings, I will not cover all of the material from the lecture notes in detail. Instead, I will begin with a summary of the notes and then be available to address any additional questions you may have.
   - Pose your questions in the classroom so others can also learn from the answers and/or discussion. 

3. If you still have questions after class, or later on when preparing for the exam, pose your question at the **Piazza site**!
   - Your questions will be answered at the Piazza site by fellow students and accorded (or corrected) by the teaching staff.

Each class is accompanied by a set of exercises (at bottom of lecture notes). These are often somewhat challenging and emphasize _quantitative skills beyond what will be required for the exam_. You may use this [**Formula Sheet**](https://github.com/bmlip/course/blob/main/assets/files/5SSD0_formula_sheet.pdf) when working on the exercises; the same sheet will also be provided during the written exam.
 

"""

# ╔═╡ 646b8c08-bcd8-4c20-973a-b03583a7d472
keyconcept("",
"Study the materials _before_ you come to the class.")

# ╔═╡ 0d017b82-d294-11ef-2d11-df36557202c9
md"""
## Piazza (Q&A)

We will be using [Piazza](https://piazza.com/) for Q&A and course announcements. Piazza is designed to get you help quickly and efficiently, both from classmates and the teaching staff.

👉 [Sign up for Piazza](http://piazza.com/tue.nl/winter2026/5ssd0) today if you haven’t already, and consider installing the Piazza app on your phone.

The sooner you start asking questions on Piazza (instead of sending emails), the sooner you’ll benefit from the collective knowledge of your classmates and instructors. Don’t hesitate to ask questions when something is unclear—you can even post anonymously if you prefer.

All **course-related announcements** will also be disseminated via Piazza. Unless it concerns a personal matter, please post your course-related questions there (in the appropriate folder).

We also encourage you to contribute by answering questions on Piazza:

  - You may answer anonymously if you wish.
  - Explaining material to others is an excellent way to deepen your own understanding.
  - Each question has one “student answer,” which the class can edit collaboratively, and one “instructor answer” for the teaching staff.

Piazza also supports **LaTeX**, please use it for math. And don’t forget to try the **search** function before posting a new question.

"""

# ╔═╡ 74dedaac-0a3e-4b83-a081-d76cdb301d56
keyconcept("",
md""" Piazza is a great resource for continuing discussion outside the classroom, where you can both ask and answer questions.""")

# ╔═╡ 8317254e-249d-49c8-acfb-1725c6349df8
md"""
## Pluto

All lectures were developed in executable [Julia](https://julialang.org/) code. Julia is an open-source programming language with a MATLAB-like syntax and performance comparable to C.

We created interactive course notebooks using [Pluto](https://plutojl.org/), which allows students to modify simulation conditions and immediately observe the results.

We look forward to receiving your feedback on your experiences with Pluto as an educational tool for playing with interactive learning materials.

"""

# ╔═╡ 0d018ee2-d294-11ef-3b3d-e34d0532a953
md"""
## Exam Guide

The course will be scored by two programming assignments and a final written exam. See the [course homepage](https://github.com/bmlip/course?tab=readme-ov-file#exams--assignments) for how the final score is computed.

**The written exam is in multiple-choice format.** 

You are not allowed to use books, smartphones, calculators, or bring printed or handwritten formula sheets to the exam. All difficult-to-remember formulas are included on this [Formula Sheet](https://github.com/bmlip/course/blob/main/assets/files/5SSD0_formula_sheet.pdf), which will be provided together with the exam.

The class homepage contains [two representative practice exams](https://github.com/bmlip/course?tab=readme-ov-file#exams--assignments) from previous terms. **It is highly recommended to practice with these previous exams when preparing for the exam.**. 


"""

# ╔═╡ 31f8669b-b547-4a11-acc6-64e02e6e9dc0
keyconcept("","The written exam will be in multiple-choice format. Two previous exams, along with their answers, are available on the course homepage.")

# ╔═╡ f46ccac8-e87c-4cbe-9d2c-fdb4723c639e
keyconcept("",md"""When working on exercises or preparing for the written exam, you may use this [Formula Sheet](https://github.com/bmlip/course/blob/main/assets/files/5SSD0_formula_sheet.pdf), which will also be provided during the exam.""" )

# ╔═╡ e56f489e-bd51-4499-aa80-1844c8651184
md"""
## Results of Previous Exams
![](https://github.com/bmlip/course/blob/main/assets/figures/exam-results.png?raw=true)
"""

# ╔═╡ 0d019cde-d294-11ef-0563-6b41bc2ca80f
TwoColumn(
md"""
## Preview
Check out [a recording from last year](https://youtu.be/k9DO26O6dIg?si=b8EiK12O_s76btPn) to understand what this class will be like. 
""",
md"""
![](https://github.com/bmlip/course/blob/main/assets/figures/Professor-Terguson.png?raw=true)
""")

# ╔═╡ f0a4b221-b4cd-425c-9a35-44c68c64e341
md"""
# Summary
"""

# ╔═╡ 9b1342f1-fcc9-469a-abfe-ed73a5b56d75
keyconceptsummary()

# ╔═╡ 46446ecd-006a-4f18-a61f-d50d5c198637
navigate_prev_next(
	nothing,
	"https://bmlip.github.io/course/lectures/Machine%20Learning%20Overview.html"
)

# ╔═╡ f3b97e01-f8d2-4865-ae81-2df412f7515a
md"""
# Code
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BmlipTeachingTools = "656a7065-6f73-6c65-7465-6e646e617262"

[compat]
BmlipTeachingTools = "~1.4.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "220ac410cc99f723de26dec023a8b8ff95db6976"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

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
git-tree-sha1 = "721865ca80c702e053b7d3958c5de5295ad84eca"
uuid = "656a7065-6f73-6c65-7465-6e646e617262"
version = "1.4.1"

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
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

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
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

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
version = "8.15.0+0"

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

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

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
version = "3.5.4+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

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

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

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

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

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
"""

# ╔═╡ Cell order:
# ╟─0cfd4bc0-d294-11ef-3537-630954a9dd27
# ╟─467c8189-b5d3-4eaf-8886-6ae53136dd8f
# ╟─0cffef7e-d294-11ef-3dd5-1fd862260b70
# ╟─0d013750-d294-11ef-333c-d9eb7578fab2
# ╟─0d0142b6-d294-11ef-0297-e5bb923ad942
# ╟─0d015ab4-d294-11ef-2e53-5339062c435c
# ╟─ab61d2fe-312c-4aca-9029-e446aaf2bfa2
# ╟─0d016cf8-d294-11ef-0c84-336979a02dd7
# ╟─646b8c08-bcd8-4c20-973a-b03583a7d472
# ╟─0d017b82-d294-11ef-2d11-df36557202c9
# ╟─74dedaac-0a3e-4b83-a081-d76cdb301d56
# ╟─8317254e-249d-49c8-acfb-1725c6349df8
# ╟─0d018ee2-d294-11ef-3b3d-e34d0532a953
# ╟─31f8669b-b547-4a11-acc6-64e02e6e9dc0
# ╟─f46ccac8-e87c-4cbe-9d2c-fdb4723c639e
# ╟─e56f489e-bd51-4499-aa80-1844c8651184
# ╟─0d019cde-d294-11ef-0563-6b41bc2ca80f
# ╟─f0a4b221-b4cd-425c-9a35-44c68c64e341
# ╟─9b1342f1-fcc9-469a-abfe-ed73a5b56d75
# ╠═46446ecd-006a-4f18-a61f-d50d5c198637
# ╟─f3b97e01-f8d2-4865-ae81-2df412f7515a
# ╠═f96d047f-9efa-4889-8b4e-a8d96677d072
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
