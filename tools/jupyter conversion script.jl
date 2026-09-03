### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 77cabc8b-29b8-4635-a5fd-d0c48240249e
using JupyterPlutoConverter

# ╔═╡ 08c15e27-7eac-4634-b3ad-33261d5095b3
using PlutoUI

# ╔═╡ 78a608ef-5546-4ee2-a74a-f77e3f6b9f0f
using Markdown: LaTeX

# ╔═╡ 1367cf9f-3997-4cca-9d76-3e90a6574bfb
lectures = [
	0 "Course-Syllabus"
	1 "Machine-Learning-Overview"
	2 "Probability-Theory-Review"
	3 "Bayesian-Machine-Learning"
	4 "Factor-Graphs"
	5 "The-Gaussian-Distribution"
	6 "The-Multinomial-Distribution"
	7 "Regression"
	8 "Generative-Classification"
	9 "Discriminative-Classification"
	10 "Latent-Variable-Models-and-VB"
	11 "Dynamic-Models"
	12 "Intelligent-Agents-and-Active-Inference"
]

# ╔═╡ 890c709f-aaa2-46ca-bf28-7dafc4cd9d3c
@bind lecture Slider(eachrow(lectures))

# ╔═╡ 887f764b-085f-43e3-a507-b1b6915462fc
lecture

# ╔═╡ 259172cb-f835-4408-9404-c9dda6bcae14
file = "/Users/fons/Documents/BMLIP/lessons/notebooks/$(lecture[2]).ipynb"

# ╔═╡ 5bdace51-e1a7-4eb8-a2fd-b044fe8cce9d
outfile = """/Users/fons/Documents/BMLIP-colorized/lectures/B$(lecture[1]) $(replace(lecture[2],"-" => " ")).jl"""

# ╔═╡ 020a0a14-9e97-4a31-a723-419e9d9944ac
Text(outfile)

# ╔═╡ 683b7af5-2882-40e2-8b51-8a18fff3571c
function transform_code(s::String)

	
	# remove complete lines containing these patterns
	s = replace(s, r".*IJulia.*" => "")
	s = replace(s, r".*Pkg\.activate.*" => "")

	s = strip(s)

	if isempty(s)
		String[]
	else
		@info "asdf" typeof(s)
		[s]
	end
end

# ╔═╡ 9359d000-6ff7-4161-b164-dc9b619f4947
s = raw"""### The Gaussian Distribution 

- Consider a random (vector) variable $x \in \mathbb{R}^M$ that is "normally" (i.e., Gaussian) distributed. The _moment_ parameterization of the Gaussian distribution is completely specified by its _mean_ $\mu$ and _variance_ $\Sigma$ and given by
$$
p(x | \mu, \Sigma) = \mathcal{N}(x|\mu,\Sigma) \triangleq \frac{1}{\sqrt{(2\pi)^M |\Sigma|}} \,\exp\left\{-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) \right\}\,.
$$
where $|\Sigma| \triangleq \mathrm{det}(\Sigma)$ is the determinant of $\Sigma$.  
  - For the scalar real variable $x \in \mathbb{R}$, this works out to 
$$
p(x | \mu, \sigma^2) =  \frac{1}{\sqrt{2\pi\sigma^2 }} \,\exp\left\{-\frac{(x-\mu)^2}{2 \sigma^2} \right\}\,.
$$"""

# ╔═╡ a7477c4d-34ef-4387-88b5-0fbca9761503
multiline_latex_regex = r"\$\$\n*(.*?)\n*\$\$"ms

# ╔═╡ 2522faf4-b084-4b97-b5a1-3142cdcc7400
multiline_latex_regex_replacer = s"\n\n```math\n\1\n```\n"

# ╔═╡ 5b546a5d-f090-469a-89e0-ee77f2e356e0
match(multiline_latex_regex, s)

# ╔═╡ f5087e8d-aa4d-4c8a-9aa8-379f620c78bd
replace(s, multiline_latex_regex => multiline_latex_regex_replacer)

# ╔═╡ 9a2cf635-c10c-4491-97ac-05eda9f5e380
basename(file)

# ╔═╡ 9d2c5770-b7a7-11ef-151b-c93f6b093d21
m = md"""
# asdf _s_

- Asdf ``adsf`` $\sqrt{a}$ yooo

```math
xoxoxo
```

```asdf
ff
```
"""

# ╔═╡ 0ff5c40f-96c0-47fd-bb2e-de410d04cc05

function Markdown.plain(io::IO, l::LaTeX)
    println(io, "```math")
    println(io, l.formula)
    println(io, "```")
end


# ╔═╡ 3cd1d102-e7d5-44fc-b47a-a955f18e85bc

function Markdown.plaininline(io::IO, l::LaTeX)
    print(io, "``", l.formula, "``")
end


# ╔═╡ 4847286f-c50d-463d-8929-ef6a86d4aed3
Text(m)

# ╔═╡ 13a0a367-862d-4311-aad8-f71536a30a0f
dump(m)

# ╔═╡ 3ae3d6a6-c6c0-4f9c-8302-98d226243ee9
md"""
# Removing the loplevel lists
"""

# ╔═╡ e0272eb8-86dc-458d-80e6-56f9940ead04
function remove_toplevel_lists(md::Markdown.MD)
	Iterators.flatmap(md.content) do c
		if c isa Markdown.List
			c.items
			### hmmmmmm and also i want to fix the headers
		elseif c isa Markdown.Header
			level = typeof(c).parameters[1]
			newlevel = max(1, level - 1)
			[Markdown.Header{newlevel}(c.text)]
		else
			[c]
		end


		
	end |> collect |> Markdown.MD

end

# ╔═╡ bbbff55c-eec8-4ff1-939f-08e22971c6de
m1 = md"""
- ``\Rightarrow`` Note that Bayesian inference is trivial in the [_canonical_ parameterization of the Gaussian](#natural-parameterization), where we would get
$$\begin{align*}
 \Lambda_c &= \Lambda_a + \Lambda_b  \quad &&\text{(precisions add)}\\
 \eta_c &= \eta_a + \eta_b \quad &&\text{(precision-weighted means add)}
\end{align*}$$
  - This property is an important reason why the canonical parameterization of the Gaussian distribution is useful in Bayesian data processing. 
""" 

# ╔═╡ de754bcf-062a-4522-ae08-52086f603ed0
remove_toplevel_lists(m1)

# ╔═╡ 02c0eb08-4b65-4435-8b6a-f81e67e8e6d1
dump(m1d)

# ╔═╡ aa76a7c6-4b7d-44a5-837d-d83ee8531df5


# ╔═╡ 1e16e207-14ab-48e2-8da6-007ea00afa4c
m2 = md"""
- $\Rightarrow$ Note that Bayesian inference is trivial in the [_canonical_ parameterization of the Gaussian](#natural-parameterization), where we would get
$$\begin{align*}
 \Lambda_c &= \Lambda_a + \Lambda_b  \quad &&\text{(precisions add)}\\
 \eta_c &= \eta_a + \eta_b \quad &&\text{(precision-weighted means add)}
\end{align*}$$
  - This property is an important reason why the canonical parameterization of the Gaussian distribution is useful in Bayesian data processing. 
""" 

# ╔═╡ 1de8add3-9a6b-4ea0-b514-ab6bf6f48fdc
m3 = md"""
- a
- b
  - c
  - d
e
- f
""" 

# ╔═╡ 41900791-87b5-4bc6-8e76-1c82a4fe1a6e
dump(m3)

# ╔═╡ 2da9f069-b6b4-4fdc-913f-f8b533e68081
md"""
## Images
"""

# ╔═╡ 66500acb-00ae-4538-9931-147ae28f03d4
sss = """

	hello

	
<p style="text-align:center;">
	<img src="./figures/fig-bishop12.png" width="400px"></p>



	
<p style="text-align:center;">
	<img src="./figures/fig-bishop12eee.png" width="400px"></p>


	
	sdf
"""

# ╔═╡ e7cdd08f-1423-4d26-8762-167b56996df4
replace(sss, r"<p[./s/S]*</p>"im => "aa") |> Text

# ╔═╡ 3be63442-0f64-4819-9473-37547c639c9e
p_img_regex = r"""<p(.|\n)*?src="(\S*)"(.|\n)*?<\/p>"""

# ╔═╡ 49e9561c-af98-480d-82ee-426eea9d4e57
p_img_regex_sub = s"""![](https://github.com/bertdv/BMLIP/blob/2024_pdfs/lessons/notebooks/\2?raw=true)"""

# ╔═╡ 72c060ae-714b-44ff-be14-c7ff8faa1d85
function transform_md(str::String)
	# removing the md""" prefix and suffix
	md_str = replace(str, r"(^\s*md\"\"?\"?)|(\"\"?\"?\s*$)" => "")

	# fix the $$ multiline latex (which is not fully supported by Julia Markdown)
	md_str = replace(md_str, multiline_latex_regex => multiline_latex_regex_replacer)

	md = Markdown.parse(md_str)

	md = remove_toplevel_lists(md)
	s = string(md)


	# image HTML
	s = replace(s, p_img_regex => p_img_regex_sub)

	# a to create links
	s = replace(s, r"""<a id=['"](.*?)['"]>(.*?)<\/a>""" => s"""$(HTML("<span id='\1'>\2</span>"))""")
	# remove <center>
	s = replace(s, r"""<center>(.*?)<\/center>""" => s"""\1""")

	# remove comments
	s = replace(s, r"""<![–-]{2}(.|\n)*?[–-]{2}>""" => "")
	

	@info "yay" md Text(md_str) Text(str) Text(s)

	"md\"\"\"\n$(s)\n\"\"\""
end

# ╔═╡ 453bfe18-088c-4683-a0aa-c8cc04236f09
jupyter2pluto(file, outfile; overwrite=true, transform_md, transform_code)

# ╔═╡ 31396180-970e-4da4-b578-ae751312cb16
replace(sss, p_img_regex => p_img_regex_sub) |> Text

# ╔═╡ d0153de3-7e63-4566-b961-ea92368760e8
for m in eachmatch(p_img_regex, sss)
	@info "yay" m
end

# ╔═╡ 150718c9-d873-4da6-bfc8-9a5a4c7177da
mm = match(p_img_regex, sss)

# ╔═╡ 41495a85-aef6-49ff-ae38-2df0d6e0a48b
mm.captures

# ╔═╡ ed7aa113-f938-462e-958d-571405e23159


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
JupyterPlutoConverter = "1eb89384-759d-4e70-b7ef-f01bc3213651"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
JupyterPlutoConverter = "~0.1.1"
PlutoUI = "~0.7.60"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.2"
manifest_format = "2.0"
project_hash = "770aa8befb8e39803124e644c77ac8c88eef1030"

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

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

[[deps.Configurations]]
deps = ["ExproniconLite", "OrderedCollections", "TOML"]
git-tree-sha1 = "4358750bb58a3caefd5f37a4a0c5bfdbbf075252"
uuid = "5218b696-f38b-4ac9-8b61-a12ec717816d"
version = "0.17.6"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.ExpressionExplorer]]
git-tree-sha1 = "7005f1493c18afb2fa3bdf06e02b16a9fde5d16d"
uuid = "21656369-7473-754a-2065-74616d696c43"
version = "1.1.0"

[[deps.ExproniconLite]]
git-tree-sha1 = "4c9ed87a6b3cd90acf24c556f2119533435ded38"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.13"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.FuzzyCompletions]]
deps = ["REPL"]
git-tree-sha1 = "be713866335f48cfb1285bff2d0cbb8304c1701c"
uuid = "fb4132e2-a121-4a70-b8a1-d5b831dcdcc2"
version = "0.5.5"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "6c22309e9a356ac1ebc5c8a217045f9bae6f8d9a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.13"

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

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JupyterPlutoConverter]]
deps = ["JSON", "Pluto", "UUIDs"]
git-tree-sha1 = "0372cb1707ed65b03ed398beebec74e1e344d461"
repo-rev = "flatmap"
repo-url = "https://github.com/fonsp/JupyterPlutoConverter.jl"
uuid = "1eb89384-759d-4e70-b7ef-f01bc3213651"
version = "0.1.2"

[[deps.LazilyInitializedFields]]
git-tree-sha1 = "0f2da712350b020bc3957f269c9caad516383ee0"
uuid = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
version = "1.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Malt]]
deps = ["Distributed", "Logging", "RelocatableFolders", "Serialization", "Sockets"]
git-tree-sha1 = "02a728ada9d6caae583a0f87c1dd3844f99ec3fd"
uuid = "36869731-bdee-424d-aa32-cab38c994e3b"
version = "1.1.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+1"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.Pluto]]
deps = ["Base64", "Configurations", "Dates", "Downloads", "ExpressionExplorer", "FileWatching", "FuzzyCompletions", "HTTP", "HypertextLiteral", "InteractiveUtils", "Logging", "LoggingExtras", "MIMEs", "Malt", "Markdown", "MsgPack", "Pkg", "PlutoDependencyExplorer", "PrecompileSignatures", "PrecompileTools", "REPL", "RegistryInstances", "RelocatableFolders", "Scratch", "Sockets", "TOML", "Tables", "URIs", "UUIDs"]
git-tree-sha1 = "7405b3be725d77858b835fbb129544f1acebf0a9"
uuid = "c3e4b0f8-55cb-11ea-2926-15256bba5781"
version = "0.19.47"

[[deps.PlutoDependencyExplorer]]
deps = ["ExpressionExplorer", "InteractiveUtils", "Markdown"]
git-tree-sha1 = "4bc5284f77d731196d3e97f23abb732ad6f2a6e4"
uuid = "72656b73-756c-7461-726b-72656b6b696b"
version = "1.0.4"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PrecompileSignatures]]
git-tree-sha1 = "18ef344185f25ee9d51d80e179f8dad33dc48eb1"
uuid = "91cefc8d-f054-46dc-8f8c-26e11d7c5411"
version = "3.0.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegistryInstances]]
deps = ["LazilyInitializedFields", "Pkg", "TOML", "Tar"]
git-tree-sha1 = "ffd19052caf598b8653b99404058fce14828be51"
uuid = "2792f1a3-b283-48e8-9a74-f99dce5104f3"
version = "0.1.0"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
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

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

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
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═77cabc8b-29b8-4635-a5fd-d0c48240249e
# ╠═1367cf9f-3997-4cca-9d76-3e90a6574bfb
# ╠═890c709f-aaa2-46ca-bf28-7dafc4cd9d3c
# ╠═887f764b-085f-43e3-a507-b1b6915462fc
# ╠═08c15e27-7eac-4634-b3ad-33261d5095b3
# ╠═259172cb-f835-4408-9404-c9dda6bcae14
# ╠═5bdace51-e1a7-4eb8-a2fd-b044fe8cce9d
# ╠═020a0a14-9e97-4a31-a723-419e9d9944ac
# ╠═72c060ae-714b-44ff-be14-c7ff8faa1d85
# ╠═683b7af5-2882-40e2-8b51-8a18fff3571c
# ╠═9359d000-6ff7-4161-b164-dc9b619f4947
# ╠═a7477c4d-34ef-4387-88b5-0fbca9761503
# ╠═2522faf4-b084-4b97-b5a1-3142cdcc7400
# ╠═5b546a5d-f090-469a-89e0-ee77f2e356e0
# ╠═f5087e8d-aa4d-4c8a-9aa8-379f620c78bd
# ╠═453bfe18-088c-4683-a0aa-c8cc04236f09
# ╠═9a2cf635-c10c-4491-97ac-05eda9f5e380
# ╠═9d2c5770-b7a7-11ef-151b-c93f6b093d21
# ╠═78a608ef-5546-4ee2-a74a-f77e3f6b9f0f
# ╠═0ff5c40f-96c0-47fd-bb2e-de410d04cc05
# ╠═3cd1d102-e7d5-44fc-b47a-a955f18e85bc
# ╠═4847286f-c50d-463d-8929-ef6a86d4aed3
# ╠═13a0a367-862d-4311-aad8-f71536a30a0f
# ╟─3ae3d6a6-c6c0-4f9c-8302-98d226243ee9
# ╠═e0272eb8-86dc-458d-80e6-56f9940ead04
# ╠═de754bcf-062a-4522-ae08-52086f603ed0
# ╠═bbbff55c-eec8-4ff1-939f-08e22971c6de
# ╠═02c0eb08-4b65-4435-8b6a-f81e67e8e6d1
# ╠═aa76a7c6-4b7d-44a5-837d-d83ee8531df5
# ╠═1e16e207-14ab-48e2-8da6-007ea00afa4c
# ╠═41900791-87b5-4bc6-8e76-1c82a4fe1a6e
# ╠═1de8add3-9a6b-4ea0-b514-ab6bf6f48fdc
# ╟─2da9f069-b6b4-4fdc-913f-f8b533e68081
# ╠═66500acb-00ae-4538-9931-147ae28f03d4
# ╠═e7cdd08f-1423-4d26-8762-167b56996df4
# ╠═3be63442-0f64-4819-9473-37547c639c9e
# ╠═49e9561c-af98-480d-82ee-426eea9d4e57
# ╠═31396180-970e-4da4-b578-ae751312cb16
# ╠═d0153de3-7e63-4566-b961-ea92368760e8
# ╠═150718c9-d873-4da6-bfc8-9a5a4c7177da
# ╠═41495a85-aef6-49ff-ae38-2df0d6e0a48b
# ╠═ed7aa113-f938-462e-958d-571405e23159
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
