### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ d5a8ec92-a4fa-46e6-983c-44572a74f886
begin
	using BmlipTeachingTools
	using LinearAlgebra
	using DataFrames
	using ProgressMeter
	using Distributions
	using Plots
end

# ╔═╡ 660571c4-3d5e-4eba-b6d9-bcab17c2a2ee
title("Programming in Julia")

# ╔═╡ 77271b72-bafe-11f0-8fa2-33da6741b124
md"""

This notebook will teach you some of the basic programming routines in Julia. You will need these skills to complete the probabilistic programming assignments later on in the course. We will assume basic familiarity with programming, such as for-loops, if-else statements and function definitions.

Resources:
- **Julia Basics** from _"Julia Programming for Machine Learning"_ taught at TU Berlin:
  - [L1: Essentials](https://adrianhill.de/julia-ml-course/L1_Basics_1/) (syntax, types, operations, functions)
  - [L2: Arrays](https://adrianhill.de/julia-ml-course/L2_Basics_2/) (vector, matrix, linear algebra)
- [JuliaLang documentation](https://docs.julialang.org/en/v1/)
- [Differences to Python, Matlab, C and Java](https://docs.julialang.org/en/v1/manual/noteworthy-differences/)
- [Video on getting started](https://www.youtube.com/watch?v=4igzy3bGVkQ&list=PLP8iPy9hna6SCcFv3FvY_qjAmtTsNYHQE)
- [Pluto.jl documentation](https://plutojl.org/en/docs/)
"""

# ╔═╡ 705dedd6-755f-4a18-9d4d-cde8c9755e05
TableOfContents()

# ╔═╡ 02c03dad-d974-4584-9766-0e7ea79047e0
md"""
## Data types

- References: [Numbers](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/), [Integers and Float](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/), [Strings](https://docs.julialang.org/en/v1/base/strings/), [Symbols](https://docs.julialang.org/en/v1/manual/metaprogramming/).

Numbers in Julia have specific types, most notably `Integer`, `Real` and `Complex`. (See [this documentation page](https://docs.julialang.org/en/v1/base/numbers/) for the full type tree.) It is important to be aware of what type your numbers are because many functions operate differently on different number types. 
"""

# ╔═╡ 9d9799b0-2e0c-4180-ae80-44e5c268c8b4
a = 3

# ╔═╡ f5df9bc3-79df-4e0b-b199-9ade76f53894
typeof(a)

# ╔═╡ 29f6c7eb-69ba-4072-a77d-7097235c1dac
md"""
`Int64` is a 64-byte integer. Other options are 32-,16-, or 8-bit integers and they can be unsigned as well. The default real-valued numbers is a 64-bit floating-point number:
"""

# ╔═╡ 3e1ca65e-f4cc-4d28-8139-f04d70980ecb
b = 3.0

# ╔═╡ 6b4c2d62-b1d0-4f42-9ee6-eb2d9f8ba98a
typeof(b)

# ╔═╡ 5d627a4e-5dee-4c25-b99c-ae7e4221109f
md"""
Converting number types is easy:
"""

# ╔═╡ 14ab0808-3fdb-4113-8854-43255987a006
c = convert(Float64, 3)

# ╔═╡ e608d8a8-8faa-4302-bf7e-ae135cef47a4
typeof(c)

# ╔═╡ 0103d1f6-ea03-4497-a0de-e1b0c181d648


# ╔═╡ 7bc7fb04-9d74-4970-b06b-5bc95d7be757
md"""
Strings are constructed by enclosing symbols within double parentheses.
"""

# ╔═╡ 5013d835-99b7-4b08-bc97-8085017d47ab
d = "3"

# ╔═╡ f5d8574f-b115-4555-8c0e-41f95824740a
typeof(d)

# ╔═╡ 6bb76a54-2df8-4604-923a-ebb428eb0353
md"""
They can be concatenated by multiplication (which is an example of a function, namely `*`, which acts differently on different input argument types): 
"""

# ╔═╡ 83e2a97e-72d7-404e-b138-f2c5d30d6365
ab = "a"*"b"

# ╔═╡ 51123db8-3fdf-4497-b121-ce5b989bc063
md"""
You can incorporate numbers into strings through a `$` call:
"""

# ╔═╡ 6969d1e5-38d0-432d-842e-153139f86613
"e = $a"

# ╔═╡ ec16be17-cdbd-467a-be07-a6fed4e05dad
md"""
Julia also has a `Symbol` type, which is used for [meta-programming](https://docs.julialang.org/en/v1/manual/metaprogramming/) (not important for this course). A `Symbol` type is characterized by a `:` in front of a word. For example:
"""

# ╔═╡ 7a1e8d0a-cffc-497a-88a7-fd9757bc8eda
f = :foo

# ╔═╡ 8ebd00be-c361-4080-b1a6-8eca643439a5
typeof(f)

# ╔═╡ ecea873a-ab10-4539-ba66-b05b1ebdf1ec
md"""
You will not have to use `Symbol`s in the course, but you may see one every once in a while in the lecture notes. For example, the `plot` command (see Visualization section), may have a `markershape=:cross` argument.
"""

# ╔═╡ 77f5b5de-040f-4406-bb57-c1fe58e3b32b
md"""
## Array manipulation

- References: [Arrays](https://docs.julialang.org/en/v1/base/arrays/)

Arrays are indexed with square brackets, `A[i,j]`. You can construct a matrix by enclosing a set of numbers with square brackets. If you separate your numbers with comma's, then you will get a column vector:
"""

# ╔═╡ 3d89c37f-0876-4ba1-aa35-0a7273d4d269
v = [1,2,3,4]

# ╔═╡ 0d8059ae-c158-43f2-81a8-9cba2f667177
md"""
If you use spaces, then you will construct a row vector (i.e., a matrix of dimensions 1 by n):
"""

# ╔═╡ ccbf4589-8d25-434e-a7f4-3d972b4ee078
w = [1 2 3 4]

# ╔═╡ 895960bf-1ee2-4aa8-b38c-5857525b1ddc
md"""
Matrices can be constructed by separating elements with spaces and rows by semicolons:
"""

# ╔═╡ 96b831a8-09ba-4792-8268-14c368b804f8
M = [1 2; 3 4]

# ╔═╡ 519dbc40-2c83-4fda-b6fe-d3547b43b407
md"""
Common array constructors are:
"""

# ╔═╡ 881a67da-5f59-437d-91ce-8c8113985fb6
begin
	U = zeros(2,3)
	V = ones(2,3)
	W = randn(2,3)
end;

# ╔═╡ 58b280bc-914e-45e2-bae4-3c767ba09ec3
md"""
Matrix operations are intuitive and similar to the mathematical notation:
"""

# ╔═╡ be6f2317-24c7-470c-bde8-233ad3d6f3bd
begin
	A = [3 2 1; 
		 2 3 2; 
		 1 2 3]
	x = [0, 
		 1, 
		 2]
	A*x
end

# ╔═╡ c97780c0-c60f-443d-98c8-100e67576e23
md"""
A matrix can be transposed by a single parenthesis, `A'`. Note that this does not mutate the array in memory. It just tells functions defined for matrices that it should change how it indexes the matrix's elements.
"""

# ╔═╡ 174612aa-d8a0-4a21-be83-91648dac1f61
x'*A*x

# ╔═╡ 4588df60-9c5b-49cc-a3a1-44db790d00ed
md"""
## Pluto notebooks

The course is built in Pluto.jl, a reactive programming environment. Pluto notebooks are wonderfully intuitive and interactive, and are already revolutionizing mathematics and computer science education (e.g., [18.S191 Computational Thinking @MIT](https://computationalthinking.mit.edu)).

A few things to be aware of when working in Pluto notebooks:
- Code is not executed sequentially. In other words, you can call a variable in a cell even if that variable is defined in a later cell.
- You cannot re-use variable names, i.e., you cannot overwrite a variable with a different variable in a latter cell.
- A cell can only execute a single function/instruction; if you want to execute more instructions, you have to wrap them in a `begin .. end` environment.
- Each Pluto notebook manages its own software packages (more on that later).
- The `Live docs` button on the bottom-right corner can be used to look up documentation of functions.
- The `Status` button on the bottom-right corner displays what the kernel is doing in the background.
"""

# ╔═╡ e47bad96-4092-4d0a-8035-f05310359565
md"""
## Broadcasting

- Reference: [Broadcasting](https://docs.julialang.org/en/v1/manual/arrays/#Broadcasting)

You can apply functions to elements in an array by placing a dot in front:
"""

# ╔═╡ ab9f639a-abd0-44e0-bb89-ed27472a641d
 3 .*[1 2 3]

# ╔═╡ 767a5b82-b007-42aa-897c-6dfa4eadf28a
md"""
This also works for named functions:
"""

# ╔═╡ cf9d6290-0aeb-484a-8816-6972fd3c4891
sin.([1., 2., 3.])

# ╔═╡ 2e3b7a55-a85b-4498-8dc4-c14853ec06ab
md"""
## Iteration

- Reference: [Collections](https://docs.julialang.org/en/v1/base/collections/)

For-loops are one of the simplest forms of iteration and can be defined in a number of ways. First, the matlab way:
"""

# ╔═╡ cc21a0ab-d413-453b-94a0-15d33524ca4f
for n = 1:2:5
    @info(n)
end

# ╔═╡ ff063276-46ef-4768-a0b1-878e9eb1a13e
md"""
Next, we can use the `range` command to construct an array of numbers and then use the `in` command to loop over elements in the array:
"""

# ╔═╡ 2f5ff4dc-5267-4d64-aca5-a6a0be536f8f
for n in range(0, stop=4, length=2)
    @info(n)
end

# ╔═╡ 122a00c1-d3d4-43da-b5d1-6bc660f5d43e
md"""
If you need both the index and the value of the array element, you can use the `enumerate` command:
"""

# ╔═╡ 034f3378-6dc8-4fff-b8a7-afdc330b4f01
for (j,n) in enumerate(range(0,3))
    @info("$j, $n")
end

# ╔═╡ 408e3447-21bb-4617-bdf2-d60f7cfb8ece
md"""
You may be familiar with "list comprehension", which is a shortened form of iterating through a collection:
"""

# ╔═╡ a02297a2-8660-4fd1-95af-438f363b9f29
["n = $n" for n in range(2,4)]

# ╔═╡ 277c5574-910b-4f05-8661-ef81278e9319
md"""
## Control flow

- References: [Control flow](https://docs.julialang.org/en/v1/manual/control-flow/), [Logical Operators](https://docs.julialang.org/en/v1/manual/missing/#Logical-operators)

Control flow refers to redirecting how a compiler goes through a program. Instead of traversing it line-by-line, things like `if-else` statements can make a compiler skip steps. These require logical operations: you can use `==` to check if two variables have the same value, `===` to check if they are actually the same object (i.e., same memory reference) and `!=` to check if they're not equal.
"""

# ╔═╡ 61f5260a-dd5b-4ca1-9d55-30d7fd453f4e
begin
	r = 3.0
	
	if r < 0 
	    "Negative"
	elseif r == 0
	    "0.0"
	else 
	    "Positive"
	end
end

# ╔═╡ 66a29ea6-d467-4909-964d-c88d68da4d50
md"""
Simple `if-else` statements can often be replaced by `ternary` checks. Essentially, you ask a question (a logical operation followed by `?`) and then tell the program what to do when the answer is yes (written immediately after the question) or no (written after the yes-answer followed by a `:`).
"""

# ╔═╡ 826ef035-bf26-45ac-b33c-dc2dcaa23f7c
r > 0 ? "Positive" : "Not positive"

# ╔═╡ ce73c7c3-5025-4784-8657-b2b67097c702
md"""
## Functions

- References: [Functions](https://docs.julialang.org/en/v1/manual/functions/), [Mutation](https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention)

Function and expressions are a core component of the julia language. Its "multiple dispatch" feature means that you can define multiple functions with the same name but with behaviour that depends on the input argument types. When you're starting out, you may not notice this so much, but you will start to appreciate this feature tremendously when you begin to professionally develop software.
"""

# ╔═╡ 3889c64c-d018-4c04-80a4-246aaede7d4e
begin
	function foo(bar::Float64)
	    message = "Boo!"
	    return message
	end
	
	function foo(bar::Integer)
	    message = "Bah!"
	    return message
	end
	
	foo(1)
end

# ╔═╡ a5a577b6-4266-4698-a3a1-7147590795b5
md"""
Note that the `return` argument does not need to be at the end of a function ([the return keyword](https://docs.julialang.org/en/v1/manual/functions/#The-return-Keyword)).

You don't actually need the `function` keyword if you have simple enough functions:
"""

# ╔═╡ 721d9bd7-c04c-4874-897e-61fde49ba257
fn(x::Number) = 1/x

# ╔═╡ 3b0e1194-e6df-46e4-8c41-032f5beeb8fa
md"""
You can add keyword arguments to a function, which are input arguments with default values:
"""

# ╔═╡ 9dc43526-5604-4251-aa22-3aff5fa3bd0b
fn(; number::Number = 1) = 1/number

# ╔═╡ 90d21b8f-b29b-420c-894b-a94abb52f2bb
fn(4)

# ╔═╡ ebbf4498-4082-44a0-a904-c80ca6dab2c7
[fn() fn(number=3)]

# ╔═╡ 44add801-211a-423d-9279-54ab6a101420
md"""
Functions that modify their input arguments instead of creating new output variables are typically marked with an `!`. Below I have defined an unsorted vector and I call `sort` to sort it in increasing order. If I call the sort function and the original vector, then they will be different:
"""

# ╔═╡ 25c4102b-9ed9-44c7-9c5f-9d7e1810f229
begin
	e = [1, 3, 2]
	[sort(e) e]
end

# ╔═╡ a520e936-d80b-42fe-9b45-28d972383ff9
md"""
But if I call the `sort!` function and the original vector, you'll see that the original vector is now also sorted.
"""

# ╔═╡ f4753dd6-51f7-48db-b8e9-26b550af1bac
[sort!(x) x]

# ╔═╡ dab1a565-43a9-48ac-896f-cc0ee7856552
md"""
## Packages

The base form of Julia has many wonderful functions and methods. But not everything. To extend functionality, we have to import additional software packages. Below is a list of the few packages we cover in today's tutorial. But you'll see more later in the course.
"""

# ╔═╡ 9f761389-7125-4df0-877f-9925b315913f
md"""
Below are some examples of importing packages and using their added functionalities.
"""

# ╔═╡ 70a9cd5f-ec47-4c45-b822-64c3ff72322a
eigen([3. 2.;2. 0.4])

# ╔═╡ 8a606467-0434-418f-85e5-40a35c9df95f
begin
	D = Dict(
	    "a" => 1,
	    "b" => 2,
	    "c" => 3
	)
	
	df = DataFrame(D)
end

# ╔═╡ 5905847d-2358-4380-a5a0-c5ac8c716839
begin
	px = Normal(1.0, 0.5)
	pdf(px, 0.0)
end

# ╔═╡ f72045aa-fa31-40bf-b80c-5259583f1d30
md"""
You will see much more of the `Distributions.jl` package in this course. See also:
"""

# ╔═╡ eacafc14-d0f6-472f-be69-2ab0d81e41b1
NotebookCard("https://bmlip.github.io/course/minis/Distributions%20in%20Julia.html")

# ╔═╡ e3be7db2-a307-4083-bb35-db4e70dfe1bf
md"""
## Visualization

- Reference: [Plots.jl](https://github.com/JuliaPlots/Plots.jl)

Plots.jl provides a common interface to call various other visualization backends (GR, Matplotlib, PGFPlotsX, Plotly, etc.). In other words, commands like `plot(..)` will be translated into the appropriate command for the backend. So, it does not re-invent the wheel but rather gives you access to the already enormous field of visualization tools.

Below are a few examples of the most common visualization commands you'll see throughout the course. The core principle is to use a `plot` type as base command and then provide keyword arguments (in `String`, `Number` or `Symbol` type) to apply variations.
"""

# ╔═╡ d2d811d0-9e38-4fc6-b576-3156309b3715
xx = range(-3, stop=3, length=301)

# ╔═╡ 6048a5ad-2964-4aab-a708-32754e52eb1c
plot(xx, pdf.(px, xx), 
	 xlabel="x", 
	 ylabel="p(x)", 
	 label="pdf", 
	 color="red", 
	 linewidth=5, 
	 linestyle=:dash)

# ╔═╡ 755d9324-5447-4240-9a17-6e212dab4edd
md"""
`plot` and `scatter` are the most useful commands. A `scatter` command will ignore properties such as `linewidth` and `linestyle` (since there are no lines) and will listen to `markersize` and `markershape` commands (see [Supported Attributes](https://docs.juliaplots.org/stable/generated/supported/) in the API).
"""

# ╔═╡ bfbfe9cf-9a28-4b52-8c50-4ab9965a6288
X = randn(10,2)

# ╔═╡ 63530cf1-e20d-4250-a6e0-68443253a0e7
scatter(X[:,1], X[:,2], 
		xlabel="X_1", 
		ylabel="X_2", 
		markersize=10, 
		markershape=:circle)

# ╔═╡ f77783b5-b35a-4437-b436-323d7668d456
md"""
It is possible to add plots to the same figure by using a `!` added to your plot command.
"""

# ╔═╡ 909a83f1-507c-4d8f-855e-f1a45d3aa927
begin
	plot(xx, pdf.(Normal(0.,1.), xx), 
		 xlabel="x", 
		 ylabel="p(x)", 
		 label="N(0,1)", 
		 color="red", 
		 linewidth=5, 
		 linestyle=:dash)
	
	plot!(xx, pdf.(Normal(1.,0.5), xx), 
		  xlabel="x", 
		  ylabel="p(x)", 
		  label="N(1,0.5)", 
		  color="blue", 
		  linestyle=:solid)
end

# ╔═╡ de01ba69-cc44-4f3a-8dfa-98f4a194b28a
md"""
## Macros

- References: [Macros](https://docs.julialang.org/en/v1/manual/metaprogramming/#man-macros)

Words that start with the `@` symbol are _macros_ in Julia, for example `@time, @test, @model`. They represent a series of functions called on an input structure and are really handy when you have to use the same set of instructions often. 

For example, you could define a `ProgressMeter` bar, update it at every iteration of a for-loop and write a custom print statement every time. _Or_, you could call the `@showprogress` macro on the for-loop itself:
"""

# ╔═╡ b3e11050-b799-4d88-b5cc-bc9d057be7bc
@showprogress for n in 1:10
    sleep(0.2)
end

# ╔═╡ 73a811a3-8c64-4bf0-9656-1d8488819f45
md"""
Macros are a somewhat advanced form of metaprogramming. You will not need to define any new macros; this instruction is just here to explain what they are.
"""

# ╔═╡ 28f3894f-6135-4359-842e-ef5e783fc469
md"""
## Errors & Debugging

- References: [Stack traces](https://docs.julialang.org/en/v1/manual/stacktraces/)

When you call a function that Julia doesn't know, it will return a list of the steps it took to execute your command and point to where its progress was blocked. This is highly useful but it typically requires a bit of practice to parse (i.e., read and filter) stack traces.

### MethodError

Suppose we write a function that expects an `Integer` input and call that function with an `Float64` argument:
"""

# ╔═╡ 113fa658-4325-4ea6-a5ab-6337889006f4
function add1(a::Integer)
    return a + 1
end

# ╔═╡ 897d515d-bfda-4a8f-abbb-9276d69f0f4d
add1(3.1415)

# ╔═╡ 2bad907d-5859-4465-8ed6-4285642d9874
md"""
Note that we first of all get a `MethodError`. This points to the fact that Julia could not find the function (i.e, method) that you asked for; `add1(::Float64)` does not exist.

Furthermore, Julia provides a list of "Closest candidates" which are functions of the same name with different input arguments. It reports:
"""

# ╔═╡ 5e402035-3a17-4072-8288-cbc613bc4cba
@htl """
<pre><code>Closest candidates are:
  add1(<span style="color: red">::Integer</span>)</code></pre>
"""

# ╔═╡ 14e2c87f-557e-45fa-8036-464d5b4deac5
md"""

You should read this as "there exists a function called `add1` that expects an `Integer` input." Try that.
"""

# ╔═╡ 8d64ec48-639e-4fc2-9c6c-5cd207c1ba75
add1(3)

# ╔═╡ e5555433-b323-4af5-805c-db592461307b
md"""
#### Mathced types in MethodError

The red highlighting of $(html"<code><span style='color: red'>::Integer</span></code>") is also important but requires a slightly more complicated example. Suppose we have a function with two `Integer` inputs and we call with `Float64` and `Int64` arguments.
"""

# ╔═╡ 08b24136-c292-4360-a90e-a8fd82d45d6b
function add(a::Integer, b::Integer)
    return a + b
end

# ╔═╡ 0e341cb6-bbca-4581-858f-50b4e2ca858a
add(3.0, 4)

# ╔═╡ c66b8dec-aa58-4508-9f93-70d65ba2c051
md"""
We again get the `MethodError` that the function we asked for doesn't exist and a suggested alternative (i.e., "closest candidate"). But this time, only the first argument $(html"<code><span style='color: red'>::Integer</span></code>") is highlighted in red. 

Julia checks your input argument types against those in the closest candidates. This tells you _what_ to change: the solution would be to either:
- Change the call: Make the first argument of your function call to an `Integer` type. That would be `add(3, 4)`.
- Change the defintion: Make the function accept a more general `Real` number as the first argument.

"""

# ╔═╡ 8fe19ca9-6618-4c70-8eca-6ba002630ca9
add(3,4)

# ╔═╡ 4f785723-2f6a-4dcd-82a7-8d15daf191e8
md"""
### What to do with errors?
To summarize, when you get an error, it is important to **take your time**, and **read the message**. If you want to know _where_ something happened, look at the stack trace. Try to see an error message as **positive feedback**: you are on the right track, and Julia is trying to help you to write the correct code.

The type of error can also be informative. The `MethodError` is probably the error you will see most often. 
"""

# ╔═╡ 28c7a5ed-cbc7-4865-a6a9-b6395ea80a5c
md"""
### Stack trace

The stack trace will show you exactly _where_ an error happened.
"""

# ╔═╡ 54662c5d-7ba3-4e96-88ce-3c9ba1907f8e
md"""
For example, the cell below contains a mistake and evaluating it will throw an error. But which line has the mistake? Press the **Show stack trace...** button or **Show more...** to see the stack trace of the error below. Where did the error happen?
"""

# ╔═╡ e500c93a-9a4a-4141-819b-051c4e5d5bb0
md"""
Here is the function used in the last cell:
"""

# ╔═╡ b6399549-c754-498a-887c-401f252ba21f
function give_magic_number(x::Real)
	kk = x*2
	ll = 3*kk
	mm = kl*ll
	nn = mm*ll
end

# ╔═╡ 794970ef-f2d2-4dc8-ae43-2b1ecc306a74
give_magic_number(7)

# ╔═╡ 400ad142-8fad-42f6-a8cb-3a1b9c56f5db
md"""
## Closing

That's it for now. If you encounter mysterious errors, please let us know on [Piazza](https://piazza.com/tue.nl/winter2026/5ssd0). We can add them to this primer.

And if you have any feedback about improving this short primer on programming in Julia, let us know!
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BmlipTeachingTools = "656a7065-6f73-6c65-7465-6e646e617262"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
ProgressMeter = "92933f4c-e287-5a05-a399-4b506db050ca"

[compat]
BmlipTeachingTools = "~1.4.1"
DataFrames = "~1.8.2"
Distributions = "~0.25.132"
Plots = "~1.41.7"
ProgressMeter = "~1.13.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.3"
manifest_format = "2.1"
project_hash = "e5cb2d6b1582b874156e85b212ee7d704ff64ced"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "8c0bc12397dabf5d2dfc4b5763ec463854cc44d3"
registries = "General"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.46"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
registries = "General"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BmlipTeachingTools]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "PlutoTeachingTools", "PlutoUI", "Reexport"]
git-tree-sha1 = "721865ca80c702e053b7d3958c5de5295ad84eca"
registries = "General"
uuid = "656a7065-6f73-6c65-7465-6e646e617262"
version = "1.4.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
registries = "General"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "0cfb779e24fdced77a05df1dbe99200821a9f563"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.8+1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "a7cf827a79b378a381e31e5c7feca853ec19cb94"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.4"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a4e53eb7a1779254f92c1cc1302c2bc6acdba560"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.1"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "a633156e74096f90b3b5e833529292b651b5671a"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.3"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
registries = "General"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
registries = "General"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
registries = "General"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
registries = "General"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
registries = "General"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
registries = "General"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
registries = "General"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
registries = "General"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
registries = "General"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
registries = "General"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "2ddbc341251ed713d39e503b504593a5d6789551"
registries = "General"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.132"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "38ef52f18cbdfa783ec6c25acc23fb1e8525ec16"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.10.2"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
registries = "General"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d0f666deaf8099a7d7ac229a74402ada88898be1"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.9.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "7bd13840b4148949e290748071bbe6826b3eabcd"
registries = "General"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d9d3cd382f2c999f684e6bfa4039fcc16c153b6f"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "9.0.2+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "831f86fde78ab7f6470d6ba4d1161a11fb6b4ec0"
registries = "General"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.18.2"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "e1e9ba3a8f5d71e6ae698f138a1362aeabb42e9f"
registries = "General"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.18.3+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d95a7a3a7e0f2fffc6291bd33035830aab01310b"
registries = "General"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.15.0+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f3f50c8001bdfa84f312234fa86715b7c9243771"
registries = "General"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.18+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "64bbbb7d1499297751b536dd39c58b20750ab1db"
registries = "General"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.5.1+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "4d777f73c46b46b8b5276206059cf8a195499314"
registries = "General"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.27"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f8eb8f7ba13ea75083531647fc8faeda8d541f07"
registries = "General"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.27+0"

[[deps.Gamma]]
deps = ["LogExpFunctions"]
git-tree-sha1 = "becc397f7cfb06e343496ae6ffb04818a851da51"
registries = "General"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.2.0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
registries = "General"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
registries = "General"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "ea9f1fb37ff10597353b56f3ba0ebd525acef6e0"
registries = "General"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.90.1+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
registries = "General"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f700c32c1e9e11b576592043f270e6b33868a56f"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.15000.0+0"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "c28dceeadc1c9c699656a3ac9d63f79cfaa17472"
registries = "General"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.31"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
registries = "General"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
registries = "General"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
registries = "General"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InlineStrings]]
git-tree-sha1 = "06b65886c7577a3784d616e29f1302c2e36e389d"
registries = "General"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.6"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "5cc6ad3e2bfc75a11eed92632ee341d37f4a4536"
registries = "General"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.18"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
registries = "General"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "d22e8bdb3f03c300a103ff23c0dd5a654c6701de"
registries = "General"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.7"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
registries = "General"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
registries = "General"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "3a9b0da8237deffb70be5e8d2e9af7601e4a79b0"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.13.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]
    JSONSerializationExt = ["Serialization"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
registries = "General"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "39bca05343661c347aae0bca57a5994a0bf4f08d"
registries = "General"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.2.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
registries = "General"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.12"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.22.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.7+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.104+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
registries = "General"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
registries = "General"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
registries = "General"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b8a91037379e1fd661e9b5ff3d7a770424b1f9f2"
registries = "General"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.4+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
registries = "General"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9e887184f306b28dbfcff88cefbb0ec80f1aa9e0"
registries = "General"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.4+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "28d620bcdb37a3fb02de9f521928ee453690f414"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.3"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
registries = "General"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "d9d20d6eb2a3290ba5c70a1c013b45305cc39c9e"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.10.20"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "676c0f1c1c34224d6869675047a54b42fdc7b8c5"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
registries = "General"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.8+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
registries = "General"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
registries = "General"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "b079a6affebde167dad8ffe0d670b53917fbcb68"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.47.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "123266c25174ef6c8d4718920abc206452cf8de6"
registries = "General"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.41"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
registries = "General"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "d874044ef56b0c9dd19d5fe966ebb62a9d27a9e9"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
registries = "General"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
registries = "General"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Reexport", "Statistics"]
git-tree-sha1 = "ddcc36fd83fb4cd459a53c891981697bde0d3de6"
registries = "General"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.5.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "83bd514e8ff16b5858ac54c53fa0bcf6002a3b00"
registries = "General"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.7"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "90b41ced6bacd8c01bd05da8aed35c5458891749"
registries = "General"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
registries = "General"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "aef1987d90dcc2e2cbd919ad10b2d25bcce4f418"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "StyledStrings", "Tables"]
git-tree-sha1 = "99fd8f32ab6728ff205cddd24e2d9ea6cbb498d3"
registries = "General"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.5.0"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"
    PrettyTablesWriteDocxExt = "WriteDocx"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    WriteDocx = "d049ceea-54ee-41d7-a26f-ba29db3b6599"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Printf"]
git-tree-sha1 = "ce5ba3edfdeba107a721748d53b5f5887f415853"
registries = "General"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.13.0"

    [deps.ProgressMeter.extensions]
    ProgressMeterDistributedExt = "Distributed"
    ProgressMeterIJuliaExt = "IJulia"

    [deps.ProgressMeter.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
registries = "General"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
registries = "General"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
registries = "General"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
registries = "General"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
registries = "General"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
registries = "General"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
registries = "General"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0c0dcffc464cda2ff33fc8ffc1dd1bdd53cd98be"
registries = "General"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.4.0"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
registries = "General"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
registries = "General"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
registries = "General"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d40b2fe70437b01397d2a4d5b020008da4e7019"
registries = "General"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.2+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "c5e8a0a52c0d5d043214de04f11f4d9c2c6ccdb8"
registries = "General"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.13"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "4910c2c2f334d74735b7a8fd02f837ead352576a"
registries = "General"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.11"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates"]
git-tree-sha1 = "8238217340ad0aaabe11afe39c1098b5bc9f4c8e"
registries = "General"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.1.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
registries = "General"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.13.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "88ef0d98204515bc53dd4aaea5f7788882d1b66c"
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
registries = "General"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
registries = "General"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91a5737baed20ee31f3faea0e51f57461f6a689e"
registries = "General"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools", "StyledStrings"]
git-tree-sha1 = "9e3a02d73a8f0a9be04e677dbef758aab40f8bea"
registries = "General"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.6.1"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "96cf0c4f535157787dc4089eca1cbed62d124d5f"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.11.2"

    [deps.StructUtils.extensions]
    StructUtilsDataFramesExt = ["DataFrames"]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazilyInitializedFields = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.10.1+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
registries = "General"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
registries = "General"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
deps = ["Serialization"]
git-tree-sha1 = "50eda8011f098869a76da69a524ff59e0298711a"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
registries = "General"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
registries = "General"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
registries = "General"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
registries = "General"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e52eca002a11c30a858185efdfb15311e1c7a6bf"
registries = "General"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
registries = "General"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
registries = "General"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "ce91b54a5f90c4f27df6950706234b34e49cf79e"
registries = "General"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.14+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
registries = "General"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
registries = "General"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
registries = "General"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
registries = "General"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
registries = "General"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "704274ca960c42049fd9bda089fc4be32af5cdca"
registries = "General"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.5+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
registries = "General"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
registries = "General"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a54cf54677c9f6eac0d97e9d757c233b56bc21b9"
registries = "General"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.13+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
registries = "General"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
registries = "General"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
registries = "General"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
registries = "General"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
registries = "General"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
registries = "General"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
registries = "General"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
registries = "General"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
registries = "General"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "9a5e6e8f1a7c40941465c04840145e527f0d6b4b"
registries = "General"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.5.0+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "df77b22eb77051b3d0ef30aab979684f7482c66f"
registries = "General"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.48.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
registries = "General"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.2+0"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
registries = "General"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
registries = "General"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0343a4dc329ca4dd71bf5c5b3fc1a8c2f6e0692d"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.15.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
registries = "General"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
registries = "General"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
registries = "General"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
registries = "General"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
registries = "General"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
registries = "General"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "1fcd6cf05d23adab0b5ffc496d020537215751dc"
registries = "General"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.59+1"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "21cbc8fc2ac0bb182ed461e556d9c062ae6a3164"
registries = "General"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.24.1+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
registries = "General"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
registries = "General"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.70.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.3+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
registries = "General"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
registries = "General"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
registries = "General"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╟─660571c4-3d5e-4eba-b6d9-bcab17c2a2ee
# ╟─77271b72-bafe-11f0-8fa2-33da6741b124
# ╟─705dedd6-755f-4a18-9d4d-cde8c9755e05
# ╟─02c03dad-d974-4584-9766-0e7ea79047e0
# ╠═9d9799b0-2e0c-4180-ae80-44e5c268c8b4
# ╠═f5df9bc3-79df-4e0b-b199-9ade76f53894
# ╟─29f6c7eb-69ba-4072-a77d-7097235c1dac
# ╠═3e1ca65e-f4cc-4d28-8139-f04d70980ecb
# ╠═6b4c2d62-b1d0-4f42-9ee6-eb2d9f8ba98a
# ╟─5d627a4e-5dee-4c25-b99c-ae7e4221109f
# ╠═14ab0808-3fdb-4113-8854-43255987a006
# ╠═e608d8a8-8faa-4302-bf7e-ae135cef47a4
# ╟─0103d1f6-ea03-4497-a0de-e1b0c181d648
# ╟─7bc7fb04-9d74-4970-b06b-5bc95d7be757
# ╠═5013d835-99b7-4b08-bc97-8085017d47ab
# ╠═f5d8574f-b115-4555-8c0e-41f95824740a
# ╟─6bb76a54-2df8-4604-923a-ebb428eb0353
# ╠═83e2a97e-72d7-404e-b138-f2c5d30d6365
# ╟─51123db8-3fdf-4497-b121-ce5b989bc063
# ╠═6969d1e5-38d0-432d-842e-153139f86613
# ╟─ec16be17-cdbd-467a-be07-a6fed4e05dad
# ╠═7a1e8d0a-cffc-497a-88a7-fd9757bc8eda
# ╠═8ebd00be-c361-4080-b1a6-8eca643439a5
# ╟─ecea873a-ab10-4539-ba66-b05b1ebdf1ec
# ╟─77f5b5de-040f-4406-bb57-c1fe58e3b32b
# ╠═3d89c37f-0876-4ba1-aa35-0a7273d4d269
# ╟─0d8059ae-c158-43f2-81a8-9cba2f667177
# ╠═ccbf4589-8d25-434e-a7f4-3d972b4ee078
# ╟─895960bf-1ee2-4aa8-b38c-5857525b1ddc
# ╠═96b831a8-09ba-4792-8268-14c368b804f8
# ╟─519dbc40-2c83-4fda-b6fe-d3547b43b407
# ╠═881a67da-5f59-437d-91ce-8c8113985fb6
# ╟─58b280bc-914e-45e2-bae4-3c767ba09ec3
# ╠═be6f2317-24c7-470c-bde8-233ad3d6f3bd
# ╟─c97780c0-c60f-443d-98c8-100e67576e23
# ╠═174612aa-d8a0-4a21-be83-91648dac1f61
# ╟─4588df60-9c5b-49cc-a3a1-44db790d00ed
# ╟─e47bad96-4092-4d0a-8035-f05310359565
# ╠═ab9f639a-abd0-44e0-bb89-ed27472a641d
# ╟─767a5b82-b007-42aa-897c-6dfa4eadf28a
# ╠═cf9d6290-0aeb-484a-8816-6972fd3c4891
# ╟─2e3b7a55-a85b-4498-8dc4-c14853ec06ab
# ╠═cc21a0ab-d413-453b-94a0-15d33524ca4f
# ╟─ff063276-46ef-4768-a0b1-878e9eb1a13e
# ╠═2f5ff4dc-5267-4d64-aca5-a6a0be536f8f
# ╟─122a00c1-d3d4-43da-b5d1-6bc660f5d43e
# ╠═034f3378-6dc8-4fff-b8a7-afdc330b4f01
# ╟─408e3447-21bb-4617-bdf2-d60f7cfb8ece
# ╠═a02297a2-8660-4fd1-95af-438f363b9f29
# ╟─277c5574-910b-4f05-8661-ef81278e9319
# ╠═61f5260a-dd5b-4ca1-9d55-30d7fd453f4e
# ╟─66a29ea6-d467-4909-964d-c88d68da4d50
# ╠═826ef035-bf26-45ac-b33c-dc2dcaa23f7c
# ╟─ce73c7c3-5025-4784-8657-b2b67097c702
# ╠═3889c64c-d018-4c04-80a4-246aaede7d4e
# ╟─a5a577b6-4266-4698-a3a1-7147590795b5
# ╠═721d9bd7-c04c-4874-897e-61fde49ba257
# ╠═90d21b8f-b29b-420c-894b-a94abb52f2bb
# ╟─3b0e1194-e6df-46e4-8c41-032f5beeb8fa
# ╠═9dc43526-5604-4251-aa22-3aff5fa3bd0b
# ╠═ebbf4498-4082-44a0-a904-c80ca6dab2c7
# ╟─44add801-211a-423d-9279-54ab6a101420
# ╠═25c4102b-9ed9-44c7-9c5f-9d7e1810f229
# ╟─a520e936-d80b-42fe-9b45-28d972383ff9
# ╠═f4753dd6-51f7-48db-b8e9-26b550af1bac
# ╟─dab1a565-43a9-48ac-896f-cc0ee7856552
# ╠═d5a8ec92-a4fa-46e6-983c-44572a74f886
# ╟─9f761389-7125-4df0-877f-9925b315913f
# ╠═70a9cd5f-ec47-4c45-b822-64c3ff72322a
# ╠═8a606467-0434-418f-85e5-40a35c9df95f
# ╠═5905847d-2358-4380-a5a0-c5ac8c716839
# ╟─f72045aa-fa31-40bf-b80c-5259583f1d30
# ╟─eacafc14-d0f6-472f-be69-2ab0d81e41b1
# ╟─e3be7db2-a307-4083-bb35-db4e70dfe1bf
# ╠═d2d811d0-9e38-4fc6-b576-3156309b3715
# ╠═6048a5ad-2964-4aab-a708-32754e52eb1c
# ╟─755d9324-5447-4240-9a17-6e212dab4edd
# ╠═bfbfe9cf-9a28-4b52-8c50-4ab9965a6288
# ╠═63530cf1-e20d-4250-a6e0-68443253a0e7
# ╟─f77783b5-b35a-4437-b436-323d7668d456
# ╠═909a83f1-507c-4d8f-855e-f1a45d3aa927
# ╟─de01ba69-cc44-4f3a-8dfa-98f4a194b28a
# ╠═b3e11050-b799-4d88-b5cc-bc9d057be7bc
# ╟─73a811a3-8c64-4bf0-9656-1d8488819f45
# ╟─28f3894f-6135-4359-842e-ef5e783fc469
# ╠═113fa658-4325-4ea6-a5ab-6337889006f4
# ╠═897d515d-bfda-4a8f-abbb-9276d69f0f4d
# ╟─2bad907d-5859-4465-8ed6-4285642d9874
# ╟─5e402035-3a17-4072-8288-cbc613bc4cba
# ╟─14e2c87f-557e-45fa-8036-464d5b4deac5
# ╠═8d64ec48-639e-4fc2-9c6c-5cd207c1ba75
# ╟─e5555433-b323-4af5-805c-db592461307b
# ╠═08b24136-c292-4360-a90e-a8fd82d45d6b
# ╠═0e341cb6-bbca-4581-858f-50b4e2ca858a
# ╟─c66b8dec-aa58-4508-9f93-70d65ba2c051
# ╠═8fe19ca9-6618-4c70-8eca-6ba002630ca9
# ╟─4f785723-2f6a-4dcd-82a7-8d15daf191e8
# ╟─28c7a5ed-cbc7-4865-a6a9-b6395ea80a5c
# ╟─54662c5d-7ba3-4e96-88ce-3c9ba1907f8e
# ╠═794970ef-f2d2-4dc8-ae43-2b1ecc306a74
# ╟─e500c93a-9a4a-4141-819b-051c4e5d5bb0
# ╠═b6399549-c754-498a-887c-401f252ba21f
# ╟─400ad142-8fad-42f6-a8cb-3a1b9c56f5db
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
