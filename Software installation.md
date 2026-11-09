# Software installation _(for students)_

Welcome to Bayesian Machine Learning for Information Processing! This course combines mathematical theory and **software implementation**, so that you can _learn by doing_. 

A bit of **background information** about the software we are using:

## Julia and RxInfer
For the coding, we use the [Julia programming language](https://julialang.org/). Julia is a high-level programming language (like Python, Matlab, R, JavaScript, etc) which makes it easy to use for complex tasks, but it has the advantage that you can write very fast numerical code (like C++, Rust, Fortran, etc). Not convinced? _Learning Julia is also a [good way to get better at other languages like Python](https://plutojl.org/en/docs/education/#why-julia-for-education)!_

At [BIASlab](https://biaslab.github.io/), we have developed a specialized package for probabilistic programming and Bayesian inference in Julia: [RxInfer.jl](https://rxinfer.com/). We will use RxInfer in the later lectures, in the Probabilistic Programming lectures and in the assignments. You don't need to install RxInfer manually, it will be loaded automatically when you open a lecture/assignment that uses it.

## Pluto
We use the [Pluto programming environment](https://plutojl.org) for course materials and for assignments. Pluto is an educational notebook environment, which makes it ideal for interactive lectures (with buttons and sliders), and for your assignments (with easy setup, improved error messages, live documentation, and more). 

Pluto was developed at MIT to teach the course [Computational Thinking](https://computationalthinking.mit.edu/), the first interactive Julia course website. Pluto is now being developed at TU/e, specifically for this course! So if you have any feedback about Pluto, or you would like to contribute, contact [Fons](github.com/fonsp)!

# How to install

## 1️⃣ Julia
First, start by installing Julia. Follow the instructions on https://julialang.org/install/ . We recommend reading this page carefully. 

Now you have Julia installed! Check that you know **how to start Julia**, and run `1 + 1` (should give `2`). You need at least Julia 1.10 for this course.

## 2️⃣ Pluto
Next, you can install and run **Pluto**. This is easy, because Julia has an easy built-in package manager. 

👉 Follow the instructions on https://plutojl.org/#install

## 3️⃣ That's it
Because Pluto has a built-in package manager, this is all you need to do for now. 🌟 When you open a lecture or assignment, Pluto will automatically make sure that the right packages are installed. When a package has not been used for a while, Julia will automatically delete it.

# How to open an assignment
First, you need to **copy the URL** of the assignment file. This is a github URL, and it should end in `.jl`. 

Next, start Pluto, following the instructions above. You should now see the [Pluto main menu](https://plutojl.org/en/docs/launch-pluto/#main-menu). Now you can **paste the URL** in the textbox, and press Enter or click the "Open" button.



# ⚠️ Not working?
If you encounter **any problems**, please contact us in class or on Piazza.

# 🙋 Feedback
If you have feedback about Pluto, RxInfer or Julia (positive, negative, ideas), please write something in Piazza or tell us in class. Feedback is really valuable for me (Fons), to guide the development of Pluto! 




# 🗑️ Uninstall
To uninstall Julia and all the packages that you used, you can run the following commands in your terminal:

Linux/MacOS:

```
# Uninstall Julia and juliaup
juliaup self uninstall

# Remove the Julia installation caches
rm -rf ~/.julia
```

For Windows Powershell, replace the last command with:

```
Remove-Item -Recurse -Force "$env:USERPROFILE\.julia"
```

For Windows Command Prompt, replace the last command with:

```
rmdir /s /q %USERPROFILE%\.julia
```
