include("./common.jl")

if length(ARGS) != 2
    error("""
    Run me from the root of the repository directory, using:

    julia tools/update_notebook_packages.jl <level> <run_notebooks>
    
    Where <level> is one of: PATCH, MINOR, MAJOR
    And <run_notebooks> is true or false to run all notebooks with Pluto at the end. This will ensure that cells are stored in the correct order.
    """)
end

level = getfield(Pkg, Symbol("UPLEVEL_$(ARGS[1])"))
run_notebooks = parse(Bool, ARGS[2])


getfrom(dir) = flatmap(walkdir(dir)) do (root, _dirs, files)
    joinpath.((root,), files)
end

for n in all_notebooks
    @info "Updating" n
    ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0
    Pluto.update_notebook_environment(n; backup=false, level)
end

@info "All notebooks done!"

if run_notebooks
    @info "Running all notebooks with Pluto..."
    Pluto.run(notebook=all_notebooks)
end
