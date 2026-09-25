if !isdir("pluto-deployment-environment") || length(ARGS) != 1
    error("""
    Run me from the root of the repository directory, using:

    julia tools/combine_pkg_envs.jl <update (true or false)>

    Where <update> is true or false to update all packages as well.
    """)
end

update = parse(Bool, ARGS[1])

import Pkg
Pkg.activate("./pluto-deployment-environment")
Pkg.instantiate()
import Pluto
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0



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




"Extract embedded project and manifest contents from each notebook"
function extract_toml_contents(notebook_path::String)
    content = read(notebook_path, String)
    
    # Regex patterns to extract TOML contents
    project_pattern = r"PLUTO_PROJECT_TOML_CONTENTS\s*=\s*\"\"\"([\s\S]*?)\"\"\""
    manifest_pattern = r"PLUTO_MANIFEST_TOML_CONTENTS\s*=\s*\"\"\"([\s\S]*?)\"\"\""
    
    project_match = match(project_pattern, content)
    manifest_match = match(manifest_pattern, content)
    
    project_toml = project_match !== nothing ? project_match.captures[1] : nothing
    manifest_toml = manifest_match !== nothing ? manifest_match.captures[1] : nothing
    
    return (project_toml, manifest_toml)
end


"Create temp directories and write TOML files for each notebook"
function create_pkg_env(notebook_path::String)
    project_toml, manifest_toml = extract_toml_contents(notebook_path)
    
    if project_toml === nothing || manifest_toml === nothing
        @warn "Could not extract TOML contents from $(basename(notebook_path))"
        return nothing
    end
    
    temp_dir = mktempdir()
    write(joinpath(temp_dir, "Project.toml"), project_toml)
    write(joinpath(temp_dir, "Manifest.toml"), manifest_toml)
    
    return temp_dir
end

# Process all notebooks and collect temp directory paths
pkg_env_dirs = map(create_pkg_env, all_notebooks)

function get_package_names(env_dir)
    Pkg.activate(env_dir) do
        p = Pkg.project()
        Set(keys(p.dependencies))
    end
end
get_package_names(::Nothing) = Set{String}()



pkg_names = map(get_package_names, pkg_env_dirs)
all_pkgs_everywhere = reduce(union, pkg_names; init=Set{String}())
@info "Analysis done" all_notebooks pkg_env_dirs all_pkgs_everywhere



### Create the mega env
mega_env_dir = mktempdir()

# Start with the biggest env
@info "Finding best starting env for mega env"
best_start = argmax(length ∘ get_package_names, pkg_env_dirs)
cp(best_start, mega_env_dir; force=true)

# Add packages from other notebooks
@info "Creating mega env"
Pkg.activate(mega_env_dir) do
    Pkg.status()
    @info "Adding pkgs. This can take a couple of minutes." collect(all_pkgs_everywhere)
    Pkg.add(collect(all_pkgs_everywhere))
    
    if update
        @info "Updating packages"
        Pkg.update()
    end
end
@info "Mega env created" mega_env_dir


"""Write TOML contents from an environment directory back to a notebook file"""
function write_toml_to_notebook(notebook_path::String, env_dir::String)
    # Read the new Project.toml and Manifest.toml
    new_project_toml = read(joinpath(env_dir, "Project.toml"), String)
    new_manifest_toml = read(joinpath(env_dir, "Manifest.toml"), String)
    
    # Read the original notebook content
    notebook_content = read(notebook_path, String)
    
    # Replace the embedded TOML contents using regex
    project_pattern = r"PLUTO_PROJECT_TOML_CONTENTS\s*=\s*\"\"\"[\s\S]*?\"\"\""
    manifest_pattern = r"PLUTO_MANIFEST_TOML_CONTENTS\s*=\s*\"\"\"[\s\S]*?\"\"\""
    
    # Replace with new content, preserving the assignment structure
    notebook_content = replace(notebook_content, project_pattern => 
        "PLUTO_PROJECT_TOML_CONTENTS = \"\"\"\n$(new_project_toml)\"\"\"")
    
    notebook_content = replace(notebook_content, manifest_pattern => 
        "PLUTO_MANIFEST_TOML_CONTENTS = \"\"\"\n$(new_manifest_toml)\"\"\"")
    
    # Write the updated content back to the notebook
    write(notebook_path, notebook_content)
end


for (notebook_path, env_dir, pkgs) in zip(all_notebooks, pkg_env_dirs, pkg_names)
    env_dir === nothing && continue
    
    new_env = mktempdir()
    cp(mega_env_dir, new_env; force=true)
    Pkg.activate(new_env) do
        Pkg.rm(collect(setdiff(all_pkgs_everywhere, pkgs)))
    end
    # update the compat entries
    Pluto.PkgCompat.write_auto_compat_entries!(Pluto.PkgCompat.load_ctx(new_env))
    
    write_toml_to_notebook(notebook_path, new_env)
    @info "Updated ✅" notebook_path
end

@info "Done 🎉"


