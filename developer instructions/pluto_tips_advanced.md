# Advanced tips
These are some things that might be useful, but they require more advanced Pluto knowledge.



## Open all notebooks at once
For example, this is how to start Pluto with all lecture notebooks running:

In the Julia REPL:
```julia
# first, cd in to the directory with the notebooks.

files = filter(x -> endswith(x, ".jl"), readdir())

Pluto.run(notebook=files)
```

_Note: this will open a browser tab with the first notebook, but this might not work. Just navigate to `localhost:1234` manually and you're good to go._






