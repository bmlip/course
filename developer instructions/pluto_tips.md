# Tips for awesome Pluto lectures

Pluto is designed for writing interactive lectures. But it does take a little while to get familiar with Pluto's interface, and with the ecosystem of packages that you can use.


## Tip: more cells
Pluto works best when you have **many small cells**, rather than a few large ones. In Jupyter, it makes sense to write a few large cells to avoid having to run many cells. In Pluto, reactivity means that you the small cells re-run automatically when needed.

Cells also allow you to **see results**. You can see one result per cell, so it helps to have many small cells to see intermediate results.

## Tip: cell order
At the end of a lecture, add a "Appendix" section. Here, you can place functions and calculations that you don't want to clutter the main content. Cells can be placed in any order, so place cells **in the order that fits your story**, which might not be the execution order.



## Tip: PlutoUI for interactivity
A great way to make lectures better is to add interactivity. 

Take a look at [the PlutoUI docs](https://featured.plutojl.org/basic/plutoui.jl) for the full list of inputs and widgets that you can use.


## Tip: PlutoTeachingTools for side content
The [PlutoTeachingTools.jl](https://github.com/JuliaPluto/PlutoTeachingTools.jl) package (not maintained by Fons but by other contributors) provides a variety of tools and widgets. Take a look at their example notebook.


## Tip: Other courses using Pluto
You can find some inspiration in other Pluto courses. Take a look at:
- https://featured.plutojl.org/
- https://computationalthinking.mit.edu/
- https://teaching.matmat.org/numerical-analysis/
- https://vchuravy.dev/rse-course/mod1_introduction/parallelism/





# Advanced tips

Take a look at [more advanced tips](pluto_tips_advanced.md).