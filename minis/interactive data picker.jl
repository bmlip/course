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

# ╔═╡ 62abfc12-dcab-11ef-3ee6-1991c0729d7e
using HypertextLiteral

# ╔═╡ fee14013-e70a-4ebf-b6e6-fa11f87b774f
using PlutoUI

# ╔═╡ edbcaefa-84ab-4658-98d0-3e2bed6c5488
function data_picker_1D(initial_data::Vector{<:Real}=Float64[]; xlim=(0,100), radius::Real=4.0, height::Real=160)
@htl """
    <script id="yolo">

	

    const Plot = this?.plotlib ?? await import("https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm");
	const div = this ?? document.createElement("div")

	const initial_data = $(initial_data)
	let data = [...initial_data]
	
	const reset_button = document.createElement("input")
	reset_button.type = "button"
	reset_button.value = "Reset"

	let plot = undefined


	const radius = $(radius)

	const render = () => {
		
		plot = Plot.plot({
		  height: $(height),
		  marks: [
		    Plot.dotX(data, Plot.dodgeY({x: (el=>el), r: radius, fill: "currentColor", padding:0})),
			data.length > initial_data.length ? null : Plot.text(["Click to add data"], {frameAnchor: "middle", fontSize: 20}),
		  ],
			x: {
				domain: $(xlim),
			}
		})

	    div.innerHTML = ""
		div.value = data
	    div.append(plot);
		div.append(reset_button)
	}

render()


div.addEventListener("click", (e) => {
	// if(e.target !== plot) return
	if(!plot.contains(e.target)) return

	const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(plot.getScreenCTM().inverse());

	const domain = plot.scale("x").domain

	const randn = (Math.random() + Math.random() + Math.random() - Math.random() - Math.random() - Math.random()) / 1.7
	
	const noise = 0 * randn * radius

	const clicked = plot.scale("x").invert(p.x + noise)

	data.push(clicked + noise)
	render()
	div.dispatchEvent(new CustomEvent("input"))
})


reset_button.addEventListener("click", (e) => {
console.log(123)
	data = [...initial_data]

	render()
	div.dispatchEvent(new CustomEvent("input"))
})


    div.plotlib = Plot
    return div

    </script>
"""
end

# ╔═╡ fcc155ed-6176-4223-843a-9dc905d339bc
@bind yollo data_picker_1D([60,61,62])

# ╔═╡ 1d766b00-22c2-4a12-a663-d37b3c8bdbb2
yollo

# ╔═╡ 5840dc91-ea47-4410-a8a6-19796c797167
function data_picker_2D(initial_data::Vector{<:Real}=Float64[]; xlim=(0,100), ylim=(0,100))
@htl """
    <script id="yolo">

	

    const Plot = this?.plotlib ?? await import("https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm");
	const div = this ?? document.createElement("div")

	const initial_data = $(initial_data)

	let data = [...initial_data]
	const reset_button = document.createElement("input")
	reset_button.type = "button"
	reset_button.value = "Reset"

	let plot = undefined


	const radius = 7.0

	const render = () => {
		
		plot = Plot.plot({
		  marks: [
		    Plot.dot(data, { r: radius, title: "name", fill: "currentColor", padding:0})
		  ],
			x: {
				domain: $(xlim),
			},
			y: {
				domain: $(ylim),
			},
		})

	    div.innerHTML = ""
		div.value = data
	    div.append(plot);
		div.append(reset_button)
	}

render()


div.addEventListener("click", (e) => {
	// if(e.target !== plot) return
	if(!plot.contains(e.target)) return

	const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(plot.getScreenCTM().inverse());


	data.push([plot.scale("x").invert(p.x), plot.scale("y").invert(p.y)])
	render()
	div.dispatchEvent(new CustomEvent("input"))
})


reset_button.addEventListener("click", (e) => {
console.log(123)
	data = [...initial_data]

	render()
	div.dispatchEvent(new CustomEvent("input"))
})


    div.plotlib = Plot
    return div

    </script>
"""
end

# ╔═╡ 541901aa-9ab5-4035-a817-ba82a04ea15e
@bind aaa data_picker_2D()

# ╔═╡ 1692b522-296f-4591-9985-357a8c6c893e
aaa

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
HypertextLiteral = "~1.0.0"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "62b830bf85efcb7f74b150ae6f77cafae5ae4846"

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

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

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

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

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
# ╠═62abfc12-dcab-11ef-3ee6-1991c0729d7e
# ╠═fee14013-e70a-4ebf-b6e6-fa11f87b774f
# ╠═1d766b00-22c2-4a12-a663-d37b3c8bdbb2
# ╠═fcc155ed-6176-4223-843a-9dc905d339bc
# ╠═edbcaefa-84ab-4658-98d0-3e2bed6c5488
# ╠═1692b522-296f-4591-9985-357a8c6c893e
# ╠═541901aa-9ab5-4035-a817-ba82a04ea15e
# ╠═5840dc91-ea47-4410-a8a6-19796c797167
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
