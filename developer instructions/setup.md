# Setup Guide

This guide will help you set up your development environment for BMLIP Colorized.

## Using Git

Use git to clone this repository locally. We recommend using "GitHub Desktop" for an easy-to-use graphical interface.

When you open a notebook in Pluto and make changes, **Pluto will always auto-save, and the `.jl` notebook files get modified**. You can then use git normally to submit the changed files: you can make branches and commits.

Here is an example of a PR made with this method: https://github.com/bmlip/course/pull/42

## Setting up Pluto

### Step 0: Install Julia
Install the latest stable version of Julia from [here](https://julialang.org/install/). At the top of the page, it recommends `juliaup` for this, that's a good choice.

### Step 1: Install Pluto
Install Pluto by following the instructions at https://plutojl.org/#install

### Step 2: Run Pluto
Start Julia and type:

```julia
import Pluto
Pluto.run()
```

### Step 3: Open a Notebook
In the Pluto main menu, type the path to one of the notebooks in the file picker. For example:

![Pluto file picker example](https://github.com/user-attachments/assets/96579ab5-1732-44a6-9454-8d4a8a486845)

Click "OPEN" to load the notebook.

## Next Steps

- Learn how to [write content](content_writing.md)
- Understand the [publishing process](publishing.md)
- Check out [presentation tips](presentation.md) 
