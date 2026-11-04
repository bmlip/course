include("./common.jl")


for path in all_notebooks
    @info "🌸 Checking $path"
    notebook = Pluto.load_notebook_nobackup(path)
    
    
    old = notebook.topology
    new = notebook.topology = Pluto.updated_topology(old, notebook, notebook.cells)
    
    
    for c in notebook.cells
        c.code_folded && continue
        !Pluto.is_just_text(new, c) && continue
        if startswith(strip(c.code), "md") || startswith(strip(c.code), "html")
            @error "Markdown cell is not folded" path c.cell_id
        end
    end
    
end
