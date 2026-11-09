# Content Writing Guide

This guide covers how to write and modify lectures in BMLIP Colorized.

## Markdown Basics

We use Markdown for prose content. Add a new cell, and write:

```julia
md"""
Some **content**.
"""
```

> [!TIP]
> You can use the keyboard shortcut **`Ctrl + M`** to quickly add/remove the `md"""` literal.

## Content Structure

### Blocks, Headers, and Quotes
There are many Markdown features to create callouts, code blocks, quotes, lists, etc. The `md"""` macro is from the Julia stdlib Markdown.jl. Read the full list of features here: https://docs.julialang.org/en/v1/stdlib/Markdown/

### Useful Widgets

**PlutoUI.jl** provides:
- `@bind` widgets like sliders
- `details` for collapsible sections ("click to read more")

**PlutoTeachingTools.jl** provides:
- `aside` for placing content in the side margin

## Package Management

Pluto has a built-in package manager that is automatically reproducible. With `using` or `import` you can import any package you want, Pluto will take care of the rest.

Read more: [https://plutojl.org/en/docs/packages/](https://plutojl.org/en/docs/packages/)

## LaTeX and Math

Check out https://plutojl.org/en/docs/latex/

## Adding Images

Check out https://plutojl.org/en/docs/images/

## HTML in Notebooks

Pluto has first-class support for HTML and JavaScript. You can use HTML cells:

```julia
html"""
<h1>My header</h1>
<p style='color: red;'>This is a paragraph in red.</p>
"""
```

Or use HTML inside Markdown with interpolation:

```julia
md"""
My header:

$(html"<h1>My header</h1>")
"""
```

## Linking

### Linking to a lecture
Go to the **course website**, find the lecture you want to link, and use that URL. For example:

```julia
md"""
Take a look at [the Bayesian Machine Learning lecture](https://bmlip.github.io/course/lectures/Bayesian%20Machine%20Learning.html).
"""
```


### Linking to a specific element in a lecture
Web browsers have a a special feature for linking to specific elements on a page. You can link to a specific element on a web page by adding a `#` followed by the element's ID. An "element" can be anything on a web page: a paragraph, a header, a pluto cell, etc. If it has an ID, you can link to it.

Because Pluto is a web application, this also works in Pluto. 


#### Linking from within a lecture
If you want to link to an element **inside the same notebook**, you can use `#id` as the URL. For example:

```julia
md"""
Take a look at [the function we used here](#remove_last_element).
"""
```

#### Linking from another lecture
If you want to link to an element **inside another notebook**, you can use the full URL of the lecture, and add a `#id` to the element you want to link to. For example:

```julia
md"""
Take a look at [the beta prior from the Bayesian Machine Learning lecture](https://bmlip.github.io/course/lectures/Bayesian%20Machine%20Learning.html#beta-prior).
"""
```

Here the URL consists of two parts, joined together:
```
# the URL
https://bmlip.github.io/course/lectures/Bayesian%20Machine%20Learning.html

# the ID
#beta-prior
```



### Getting the ID
Take a look at https://plutojl.org/en/docs/linking/ to learn how to get the ID of an element on a web page.


## Code in Pluto

Pluto has a stricter runtime than Jupyter to ensure reproducibility. Read more about the differences in [this article](https://featured.plutojl.org/basic/pluto%20for%20jupyter%20users).

## Next Steps

- Learn about the [publishing process](publishing.md)
- Check out [presentation tips](presentation.md) 
