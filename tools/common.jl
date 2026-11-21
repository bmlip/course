if !isdir("pluto-deployment-environment")
    error("""
    Run me from the root of the repository directory, using:

    julia tools/scriptname.jl argument1 argument2 ...
    """)
end


begin
    import TOML
    manifest_version = TOML.parsefile("./pluto-deployment-environment/Manifest.toml")["julia_version"]
    
    @assert manifest_version == string(VERSION) "This repository uses Julia version $(manifest_version) (in pluto-deployment-environment), but this is Julia $(VERSION). Start a new Julia session with the correct version, or create a new Manifest.toml."
end

import Pkg
Pkg.activate("./pluto-deployment-environment")
Pkg.instantiate()
import Pluto
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0

flatmap(args...) = vcat(map(args...)...)


# Find all notebooks
lectures_dir = joinpath(@__DIR__, "..", "lectures")
prob_prog_dir = joinpath(@__DIR__, "..", "probprog")
minis_dir = joinpath(@__DIR__, "..", "minis")
ignore_dir = joinpath(@__DIR__, "..", "ignore_dir")

all_notebooks = mapreduce(union, [
    lectures_dir, 
    prob_prog_dir, 
    minis_dir,
    ignore_dir
]) do d
    isdir(d) ? 
        filter(Pluto.is_pluto_notebook, readdir(d; join=true)) : 
        String[]
end
