### A Pluto.jl notebook ###
# v0.20.13

#> [frontmatter]
#> image = "https://github.com/bmlip/course/blob/v2/assets/figures/scientific-inquiry-loop-w-BML-eqs.png?raw=true"
#> description = "Introduction to Bayesian modeling, parameter estimation, and model evaluation."
#> 
#>     [[frontmatter.author]]
#>     name = "BMLIP"
#>     url = "https://github.com/bmlip"

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ df312e6a-503f-486f-b7ec-15404070960c
using Distributions, StatsPlots, SpecialFunctions

# ╔═╡ 3987d441-b9c8-4bb1-8b2d-0cc78d78819e
using Plots, LaTeXStrings, Plots.PlotMeasures

# ╔═╡ caba8eee-dfea-45bc-a8a7-1dd20a1fa994
using PlutoUI, PlutoTeachingTools

# ╔═╡ 6a23b828-d294-11ef-371a-05d061144a43
md"""
# Bayesian Machine Learning

"""

# ╔═╡ 6be2e966-4048-44d0-a37e-95060e3fe30b
PlutoUI.TableOfContents()

# ╔═╡ 6a23df9e-d294-11ef-3ddf-a51d4cea00fc
md"""
## Preliminaries

##### Goals

  * Introduction to Bayesian (i.e., probabilistic) modeling

##### Materials

  * Mandatory

      * These lecture notes

  * Optional

      * Bishop pp. 68-74 (on the coin toss example)
      * [Ariel Caticha - 2012 - Entropic Inference and the Foundations of Physics](https://github.com/bmlip/course/blob/main/assets/files/Caticha-2012-Entropic-Inference-and-the-Foundations-of-Physics.pdf), pp.35-44 (section 2.9, on deriving Bayes rule for updating probabilities)

    

"""

# ╔═╡ 6a24376c-d294-11ef-348a-e9027bd0ec29
md"""
$(section_outline("Challenge:", "Predicting a Coin Toss"))

##### Problem 

  * We observe the following sequence of heads (outcome ``=1``) and tails (outcome ``=0``) when tossing the same coin repeatedly: 

```math
D=\{1011001\}\,.
```

  * What is the probability that heads comes up next?

##### Solution

  * Later in this lecture. 
"""

# ╔═╡ 6a24b9e4-d294-11ef-3ead-9d272fbf89be
md"""
# The Bayesian Modeling Approach

"""

# ╔═╡ 6a24c3e6-d294-11ef-3581-2755a9ba15ba
md"""

Suppose that your application is to predict a future observation ``x``, based on ``N`` past observations ``D=\{x_1,\dotsc,x_N\}``.

The **Bayesian modeling** approach to solving this task involves four stages: 

"""

# ╔═╡ e2de9415-7bd8-4e95-abeb-53fc068ee950
md"""
	REPEAT
		1. Model Specification
		2. Parameter Estimation
		3. Model Evaluation
	UNTIL model performance is satisfactory
		4. Apply Model
"""

# ╔═╡ 6a24c9f4-d294-11ef-20cc-172ea50da901
md"""
In principle, based on the model evaluation results, you may want to re-specify your model and *repeat* the design process (a few times), until model performance is acceptable. 

"""

# ╔═╡ 6a24cee0-d294-11ef-35cb-71ab9ef935e5
md"""
Next, we discuss these four stages in a bit more detail.

"""

# ╔═╡ 6a24d478-d294-11ef-2a75-9d03a5ba7ff8
md"""
## 1. Model Specification

Your first task is to propose a probabilistic model for generating the observations ``x``.

"""

# ╔═╡ 6a24fde8-d294-11ef-29bf-ad3e20a53c29
md"""
A probabilistic model ``m`` consists of a joint distribution ``p(x,\theta|m)`` that relates observations ``x`` to model parameters ``\theta``. Usually, the model is proposed in the form of a data-generating  distribution ``p(x|\theta,m)`` and a prior ``p(\theta|m)``,

"""

# ╔═╡ a75c75ed-c67b-4be2-adbf-8984f27fc05d
md"""


```math
\underbrace{p(x,\theta|m)}_{\text{model}} = \underbrace{p(x|\theta,m)}_{\substack{ \text{data}\\ \text{generation}}} \,\underbrace{p(\theta|m)}_{\text{prior}} \,.
```

"""

# ╔═╡ 6a251a08-d294-11ef-171a-27b9d0f818bc
md"""
*You* are responsible to choose the data generating distribution ``p(x|\theta)`` based on your physical understanding of the data generating process. (For brevity, if we are working on one given model ``m`` with no alternative models, we usually drop the given dependency on ``m`` from the notation).

"""

# ╔═╡ 6a252250-d294-11ef-33cd-89b18066817d
md"""
*You* must also choose the prior ``p(\theta)`` to reflect what you know about the parameter values before you see the data ``D``.

"""

# ╔═╡ 6a25307e-d294-11ef-0662-3db678b32e99
md"""
## 2. Parameter Estimation

You must now specify a likelihood function for the parameters from the data-generating distribution. Note that, for a given (i.e., *observed*) data set ``D=\{x_1,x_2,\dots,x_N\}`` with *independent* observations ``x_n``, the likelihood factorizes as 

```math
 p(D|\theta) = \prod_{n=1}^N p(x_n|\theta)\,.
```

So, usually you select the data-generating distribution for one observation ``x_n`` and then use (in-)dependence assumptions to combine these models into a likelihood function for the model parameters.

"""

# ╔═╡ 6a25379a-d294-11ef-3e07-87819f6d75cb
md"""
The likelihood and prior both contain information about the model parameters. Next, you use Bayes rule to fuse these two information sources into a posterior distribution for the parameters:

```math
\begin{align*}
\underbrace{p(\theta|D) }_{\text{posterior}}  =\frac{\overbrace{p(D|\theta)}^{\text{likelihood}} \,\overbrace{p(\theta)}^{\text{prior}}}{\underbrace{\int p(D|\theta) p(\theta) \mathrm{d}\theta}_{p(D)\text{ (evidence)}}}
\end{align*}
```

"""

# ╔═╡ 6a254460-d294-11ef-1890-230b75b6b9ee
md"""
Note that there's **no need for you to design some clever parameter estimation algorithm**. Bayes rule *is* the parameter estimation algorithm, which can be entirely expressed in terms of the likelihood and prior. The only complexity lies in the computational issues (in particular, the computational load of computing the evidence)! 

"""

# ╔═╡ 6a2552ac-d294-11ef-08d6-179e068bc297
md"""
This parameter estimation "recipe" works if the right-hand side (RHS) factors can be evaluated; the computational details can be quite challenging and this is what machine learning is about.     


"""

# ╔═╡ ce75e785-868f-4361-93f8-c582ac1b891b
keyconcept(" ", 
	md"""
	
	Bayesian Machine learning is EASY, apart from computational details :)
	
	"""
)

# ╔═╡ 6a2561c0-d294-11ef-124d-373846e3120c
md"""
## 3. Model Evaluation

In the framework above, parameter estimation was executed by "perfect" Bayesian reasoning. So is everything settled now? 

"""

# ╔═╡ 6a257020-d294-11ef-0490-e151934b2f42
md"""
No, there appears to be one remaining problem: how good really were our assumptions ``p(x|\theta)`` and ``p(\theta)`` in the model specification phase? We want to "score" the model performance.

"""

# ╔═╡ 6a257f34-d294-11ef-2928-fbb800e81124
md"""
Note that this question is only interesting in practice if we have alternative models to choose from. After all, if you don't have an alternative model, any value for the model evidence would still not lead you to switch to another model.  

"""

# ╔═╡ 6a25a11e-d294-11ef-1c51-09482dad86f2
md"""
Let's assume that we have more candidate models, say ``\mathcal{M} = \{m_1,\ldots,m_K\}`` where each model relates to a specific prior ``p(\theta|m_k)`` and likelihood ``p(D|\theta,m_k)``? Can we evaluate the relative performance of a model against another model from the set?

"""

# ╔═╡ 6a25edfc-d294-11ef-3411-6f74c376461e
md"""
Start again with **model specification**. *You* must now specify a *model* prior ``p(m_k)`` (next to the likelihood ``p(D|\theta,m_k)`` and *parameter* prior ``p(\theta|m_k)``) for each of the models to get a new model specification that includes the model ``m_k`` as a parameter:
"""

# ╔═╡ 53de7edd-6c28-49a7-9f54-cf7b8ca42aeb
md"""
```math
p(D,\theta,m_k) = p(D|\theta,m_k) p(\theta|m_k) p(m_k)
```
"""

# ╔═╡ 288fbee6-0783-4447-b5d0-f5c2b29b39c7
md"""

Then, solve the desired inference problem for the posterior over the model ``m_k``:      

```math
\begin{align} 
\underbrace{p(m_k|D)}_{\substack{\text{model}\\\text{posterior}}} 
  = \underbrace{p(m_k)}_{\substack{\text{model}\\\text{prior}}}\, \underbrace{\int_\theta \underbrace{p(D|\theta,m_k)}_{\text{likelihood}} \,\underbrace{p(\theta|m_k)}_{\substack{\text{parameter} \\ \text{prior}}}\, \mathrm{d}\theta }_{\substack{\text{evidence }p(D|m_k)\\\text{= model likelihood}}}\\
\end{align}
```

"""

# ╔═╡ 74fa1925-0d9f-47f6-a6bd-b822948a4fbc
details("Proof this yourself, and click for solution",
md"""
```math
\begin{align} 
p(m_k|D)&= \frac{p(m_k,D) }{p(D)} \\
  &\propto p(m_k,D)\\
 &= \int_\theta p(D,\theta,m_k) \,\mathrm{d}\theta\\
  &= p(m_k)\int_\theta p(D|\theta,m_k)\,p(\theta|m_k)\, \mathrm{d}\theta 
\end{align}
```	   
""")

# ╔═╡ 6a261278-d294-11ef-25a0-5572de58ad06
md"""
You *can* evaluate the RHS of this equation since *you* selected the model priors ``p(m_k)``, the parameter priors ``p(\theta|m_k)``, and the likelihoods ``p(D|\theta,m_k)``.

"""

# ╔═╡ 6a26549a-d294-11ef-1f10-15c4d14ae41f
md"""
Note that, to evaluate the model posterior, you must calculate the **model evidence** ``p(D|m_k)``, which can be interpreted as a likelihood function for model ``m_k``. 

"""

# ╔═╡ 6a262182-d294-11ef-23e9-ed45e1da9f46
md"""
You can now compare posterior distributions ``p(m_k|D)`` for a set of models ``\{m_k\}`` and decide on the merits of each model relative to alternative models. This procedure is called **Bayesian model comparison**.

"""

# ╔═╡ 6a2672d6-d294-11ef-1886-3195c9c7cfa9
md"""
Again, **no need to invent a special algorithm for estimating the performance of your model**. Straightforward application of probability theory takes care of all that. 

"""

# ╔═╡ 6aa2399d-a949-40f9-8ee6-b0c2be1dc478
keyconcept(" ", 
	md"""
	
	In a Bayesian modeling framework, **model evaluation** follows the same recipe as parameter estimation; it just works at one higher hierarchical level.
	
	"""
)


# ╔═╡ 6a2664c6-d294-11ef-0a49-5192e17fb9ea
md"""

Compare the calculations between parameter estimation and model evaluation
```math
\begin{align*}
p(\theta|D) &\propto p(D|\theta) p(\theta) \; &&\text{(parameter estimation)} \\
p(m_k|D) &\propto p(D|m_k) p(m_k) \; &&\text{(model evaluation)}
\end{align*}
```

"""

# ╔═╡ 6a26a31e-d294-11ef-2c2f-b349d0859a27
md"""
With the (relative) performance evaluation scores of your model in hand, you could now re-specify your model (hopefully an improved model) and *repeat* the design process until the model performance score is acceptable (see the 4-step [Bayesian modeling process](#Bayesian-modeling-recipe) above). 

"""

# ╔═╡ 6a269568-d294-11ef-02e3-13402d296391
md"""
In principle, you could proceed with asking how good your choice for the candidate model set ``\mathcal{M}`` was. You would have to provide a set of alternative model sets ``\{\mathcal{M}_1,\mathcal{M}_2,\ldots,\mathcal{M}_M\}`` with priors ``p(\mathcal{M}_m)`` for each set and compute posteriors ``p(\mathcal{M}_m|D)``. And so forth ...  

"""

# ╔═╡ 6a26b7bc-d294-11ef-03e7-2715b6f8dcc7
md"""
### Bayes Factors

"""

# ╔═╡ 6a26f244-d294-11ef-0488-c1e4ec6e739d
md"""
As an aside, in the (statistics and machine learning) literature, performance comparison between two models is often reported by the [Bayes Factor](https://en.wikipedia.org/wiki/Bayes_factor), which is defined as the ratio of model evidences: 

```math
\begin{align*}
\mathrm{BF_{12}} \triangleq  \frac{p(D|m_1)}{p(D|m_2)}  
= \underbrace{\frac{p(m_1|D)}{p(m_2|D)}}_{\substack{\text{posterior} \\ \text{ratio}}} \cdot \underbrace{\frac{p(m_2)}{p(m_1)}}_{\substack{\text{prior} \\ \text{ratio}}}
\end{align*}
```
"""

# ╔═╡ 99db44c9-185c-4f39-ae5e-1a4cd751d980
details("Proof this yourself, and click for solution",
md"""
```math
\begin{align*}
\mathrm{BF_{12}} &= \frac{p(D|m_1)}{p(D|m_2)}  \\
&= \frac{p(D,m_1)}{p(m_1)} \bigg/ \frac{p(D,m_2)}{p(m_2)} \\
&= \frac{p(m_1|D) p(D)}{p(m_1)} \cdot \frac{p(m_2)}{p(m_2|D) p(D)} \\
&= \frac{p(m_1|D)}{p(m_2|D)} \cdot \frac{p(m_2)}{p(m_1)} 
\end{align*}
```
""")		

# ╔═╡ d22f58ac-9f68-41cb-8e61-cf74d3692c44
md"""
Hence, for equal model priors (``p(m_1)=p(m_2)=0.5``), the Bayes Factor reports the posterior probability ratio for the two models. 

In principle, any hard decision on which is the better model has to accept some *ad hoc* arguments.  [Jeffreys (1961)](https://www.amazon.com/Theory-Probability-Classic-Physical-Sciences/dp/0198503687/ref=sr_1_1?qid=1663516628&refinements=p_27%3Athe+late+Harold+Jeffreys&s=books&sr=1-1&text=the+late+Harold+Jeffreys) advises to use the **log-Bayes factor**,  

```math
\mathrm{logBF}_{12} := ^{10}\log\frac{p(D|m_1)}{p(D|m_2)} \,,
```

to quantify evidence for preferring model ``m_1`` over ``m_2`` by the following interpretation:

| ``\mathrm{logBF}_{12}`` | Evidence for ``m_1``    |
|:---------------------|:----------------------------|
| 0 to 0.5             | not worth mentioning        |
| 0.5 to 1             | substantial                 |
| 1 to 2               | strong                      |
| >2                   | decisive                    |

"""

# ╔═╡ 6a2707e6-d294-11ef-02ad-31bf84662c70
md"""
## 4. Apply Model (Prediction)

Once we are satisfied with the evidence for a (trained) model, we can apply the model to our prediction/classification/etc task.

"""

# ╔═╡ 6a271a56-d294-11ef-0046-add807cc0b4f
md"""
Given the data ``D``, our knowledge about a yet unobserved datum ``x`` is captured by the following inference problem (where everything is conditioned on the selected model):

```math
p(x|D) = \int \underbrace{p(x|\theta)}_{\substack{\text{data } \\ \text{generating}}} \, \underbrace{p(\theta|D)}_{\text{posterior}} \,\mathrm{d}\theta
```

"""

# ╔═╡ f6ee5570-9b92-42b6-baf3-3eed5352a060
details("Proof this yourself, and click for solution",
md"""
```math
\begin{align*}
p(x|D) &\stackrel{s}{=} \int p(x,\theta|D) \,\mathrm{d}\theta\\
 &\stackrel{p}{=} \int p(x|\theta,D) p(\theta|D) \,\mathrm{d}\theta\\
 &\stackrel{m}{=} \int p(x|\theta) \, p(\theta|D) \,\mathrm{d}\theta
\end{align*}
```		

In the last equation, the simplification ``p(x|\theta,D) = p(x|\theta)`` follows from our model specification. In particular, we assumed a *parametric* data generating distribution ``p(x|\theta)`` with no explicit dependency on the data set ``D``. Technically, in our model specification, we assumed that ``x`` is conditionally independent from ``D``, given the parameters ``\theta``, i.e., we assumed ``p(x|\theta, D) = p(x|\theta)``. The information from the data set ``D`` has been absorbed in the posterior ``p(\theta|D)``, so all information from ``D`` is passed to a new observation ``x`` through the (posterior distribution over the) parameters ``\theta``. 
		
""")

# ╔═╡ 6a273ae0-d294-11ef-2c00-9b3eaed93f6d
md"""
Again, **no need to invent a special prediction algorithm**. Probability theory takes care of all that. The complexity of prediction is just computational, namely, how to carry out the marginalization over ``\theta``.

"""

# ╔═╡ 6a274948-d294-11ef-0563-1796b8883306
md"""
Note that the application of the learned posterior ``p(\theta|D)`` not necessarily has to be a prediction task. We use it here as an example, but other applications (e.g., classification, regression etc.) are of course also possible. 

"""

# ╔═╡ 6a275a52-d294-11ef-1323-9d83972f611a
md"""
### Prediction with multiple models

When you have a posterior ``p(m_k|D)`` for the models, you don't *need* to choose one model for the prediction task. You can do prediction by **Bayesian model averaging**, which combines the predictive power from all models:

```math
\begin{align*}
p(x|D) &= \sum_k \int p(x,\theta,m_k|D)\,\mathrm{d}\theta \\
 &= \sum_k \int  p(x|\theta,m_k) \,p(\theta|m_k,D)\, p(m_k|D) \,\mathrm{d}\theta \\
  &= \sum_k \underbrace{p(m_k|D)}_{\substack{\text{model}\\\text{posterior}}} \cdot \int \underbrace{p(\theta|m_k,D)}_{\substack{\text{parameter}\\\text{posterior}}} \, \underbrace{p(x|\theta,m_k)}_{\substack{\text{data generating}\\\text{distribution}}} \,\mathrm{d}\theta
\end{align*}
```

"""

# ╔═╡ 6a27684e-d294-11ef-040e-c302cdad714a
md"""
Alternatively, if you do need to work with one model (e.g. due to computational resource constraints), you can for instance select the model with largest posterior ``p(m_k|D)`` and use that model for prediction. This is called **Bayesian model selection**.

"""

# ╔═╡ 6a2777d0-d294-11ef-1ac3-add102c097d6
md"""
Bayesian model averaging is the principal way to apply PT to machine learning. You don't throw away information by discarding lesser performant models, but rather use PT (marginalization of models) to compute 

```math
p(\text{what-I-am-interested-in} \,|\, \text{all available information})\,.
```

"""

# ╔═╡ 6a278784-d294-11ef-11ae-65bd398910d5
md"""
## We're Done!

In principle, you now have the recipe in your hands to solve all your prediction/classification/regression (etc.) problems by the same Bayesian modeling method:

"""

# ╔═╡ c03229ef-3e0f-4612-909b-97f488a1e4c9
md"""
	REPEAT
		1. Model Specification
		2. Parameter Estimation
		3. Model Evaluation
	UNTIL model performance is satisfactory
		4. Apply Model
"""

# ╔═╡ 6a27951c-d294-11ef-2e1a-b5a4ce84aceb
md"""
Crucially, there is no need to invent clever machine learning algorithms, and there is no need to invent a clever prediction algorithm nor a need to invent a model performance criterion. Instead, you propose a model and, from there on, you let PT reason about everything that you care about. 

"""

# ╔═╡ 6a27a28a-d294-11ef-1f33-41b444761429
md"""
Your problems are only of computational nature. Perhaps the integral to compute the evidence may not be analytically tractable, etc.

"""

# ╔═╡ 6a27b114-d294-11ef-099d-1d55968934a6
md"""
## Bayesian Evidence as a Model Performance Criterion

I'd like to convince you that $(HTML("<span id='Bayesian-model-evidence'>Bayesian model evidence</span>")) ``p(D|m)`` is an excellent criterion for assessing your model's performance. To do so, let us consider a decomposition that relates model evidence to other highly-valued criteria such as **accuracy** and **model complexity**.

"""

# ╔═╡ 6a27beca-d294-11ef-1895-d57b11b827c1
md"""
Consider a model ``p(x,\theta|m)`` and a data set ``D = \{x_1,x_2, \ldots,x_N\}``.

"""

# ╔═╡ cc8af69e-6d00-4327-aaa2-0b1023052b8a
md"""
Given the data set ``D``, the log-evidence for model ``m`` decomposes as 
"""

# ╔═╡ c454be00-05e7-42f6-a243-bf559ed6eff7
md"""
```math
\begin{flalign}
\underbrace{\log p(D|m)}_{\text{log-evidence}} = \underbrace{\int p(\theta|D,m) \log p(D|\theta,m) \mathrm{d}\theta}_{\text{accuracy (a.k.a. data fit)}} - \underbrace{\int p(\theta|D,m) \log  \frac{p(\theta|D,m)}{p(\theta|m)} \mathrm{d}\theta}_{\text{complexity}} \,.
\end{flalign}
```

""" 

# ╔═╡ 6a9ad1c4-dfb2-4987-9ddc-da6131605083
details("Click for proof", 
md"""
```math
\begin{flalign}
\log p(D|m)&= \log p(D|m) \cdot   \underbrace{\int p(\theta|D,m)\mathrm{d}\theta}_{\text{evaluates to }1} \\
 &= \int p(\theta|D,m) \log p(D|m) \mathrm{d}\theta  \qquad \text{(move $\log p(D|m)$ into the integral)} \\
 &= \int p(\theta|D,m) \log \underbrace{\frac{p(D|\theta,m) p(\theta|m)}{p(\theta|D,m)}}_{\text{by Bayes rule}} \mathrm{d}\theta \\
  &= \underbrace{\int p(\theta|D,m) \log p(D|\theta,m) \mathrm{d}\theta}_{\text{accuracy (a.k.a. data fit)}} - \underbrace{\int p(\theta|D,m) \log  \frac{p(\theta|D,m)}{p(\theta|m)} \mathrm{d}\theta}_{\text{complexity}}
\end{flalign}
```
""")

# ╔═╡ 6a27efc6-d294-11ef-2dc2-3b2ef95e72f5
md"""
The "accuracy" term (also known as data fit) measures how well the model predicts the data set ``D``. We want this term to be high because good models should predict the data ``D`` well. Indeed, higher accuracy leads to higher model evidence. To achieve high accuracy, applying Bayes' rule will shift the posterior ``p(\theta|D)`` away from the prior towards the likelihood function ``p(D|\theta)``.

"""

# ╔═╡ 6a280132-d294-11ef-10ac-f3890cb3f78b
md"""
The second term ("complexity", also known as "information gain") is technically a [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) (KLD) between the posterior and prior distributions, see [OPTIONAL SLIDE](#KLD) below. The KLD is an information-theoretic quantity that can be interpreted as a "distance" measure between two distributions. In other words, the complexity term measures how much the beliefs about ``\theta`` changed, due to learning from the data ``D``. Generally, we like the complexity term to be low, because moving away means forgetting previously acquired information represented by the prior. Indeed, lower complexity leads to higher model evidence.

"""

# ╔═╡ 80edf8a4-e738-4bdb-bea3-0967926da645
TODO("Fons, can we move teh OPTIONAL Slide KL divergence to a mini?")

# ╔═╡ 6a2814b0-d294-11ef-3a76-9b93c1fcd4d5
md"""
Models with high evidence ``p(D|m)`` prefer both high accuracy and low complexity. Therefore, models with high evidence tend to predict the training data ``D`` well (high accuracy), yet also try to preserve the information encoded by the prior (low complexity). These types of models are said to *generalize* well, since they can be applied to different data sets without specific adaptations for each data set.  

"""

# ╔═╡ 6a282892-d294-11ef-2c12-4b1c7374617c
md"""
Focussing only on accuracy maximization could lead to *overfitting* of the data set ``D``. Focussing only on complexity minimization could lead to *underfitting* of the data. Bayesian ML attends to both terms and avoids both underfitting and overfitting.  

"""

# ╔═╡ 6a286b04-d294-11ef-1b34-8b7a85c0048c
keyconcept(" ", 
	md"""
	
	Bayesian learning automatically leads to models that generalize well. There is **no need for early stopping or validation data sets**. There is also **no need for tuning parameters** in the learning process. Just learn on the full data set and all behaves well. 	
	"""
)

# ╔═╡ 6a2879e6-d294-11ef-37db-df7babe24d25
md"""
Put provocatively, this highlights that the common machine learning practice of splitting a dataset into training, validation, and test sets is, in essence, an ad hoc workaround, a substitute for formulating the learning task properly as a Bayesian inference problem.

"""

# ╔═╡ 6a2889ae-d294-11ef-2439-e1a541a5ccd7
md"""
## Bayesian Modeling and the Scientific Method Revisited

The Bayesian modeling approach provides a unified framework for the Scientific Inquiry method. We can now add equations to the design loop. (Trial design to be discussed in [Intelligent Agent lesson](https://bmlip.github.io/course/lectures/Intelligent%20Agents%20and%20Active%20Inference.html).) 

![](https://github.com/bmlip/course/blob/v2/assets/figures/scientific-inquiry-loop-w-BML-eqs.png?raw=true)

"""

# ╔═╡ c050f468-7eec-403f-9304-552bd0d9b222
html"""
<style>
pluto-output img {
	background: white;
	border-radius: 3px;
}
</style>
"""

# ╔═╡ 6a2898ea-d294-11ef-39ec-31e4bac1e048
md"""
# Revisiting the Challenge: Predicting a Coin Toss

At the beginning of this lesson, we posed the following challenge:

We observe a the following sequence of heads (outcome = ``1``) and tails (outcome = ``0``) when tossing the same coin repeatedly 

```math
D=\{1011001\}\,.
```

What is the probability that heads comes up next? We solve this in the next slides ...

"""

# ╔═╡ 6a28a704-d294-11ef-1bf2-efbdb0cb4cbc
md"""
## 1. Model Specification for Coin Toss

We observe a sequence of ``N`` coin tosses ``D=\{x_1,\ldots,x_N\}`` with ``n`` heads. 

"""

# ╔═╡ 6a28b44c-d294-11ef-15da-81be8753d311
md"""
Let us denote outcomes by 

```math
x_k = \begin{cases} 1 & \text{if heads comes up} \\
  0 & \text{otherwise (tails)} \end{cases}
  
```

"""

# ╔═╡ 6a28c9b4-d294-11ef-222b-97bf0912efe7
md"""
### Likelihood

Assume a [**Bernoulli** distributed](https://en.wikipedia.org/wiki/Bernoulli_distribution) variable ``p(x_k=1|\mu)=\mu`` for a single coin toss, leading to 

```math
p(x_k|\mu)=\mu^{x_k} (1-\mu)^{1-x_k} \,.
```

Assume ``n`` times heads were thrown out of a total of ``N`` throws. The likelihood function then follows a [**binomial** distribution](https://en.wikipedia.org/wiki/Binomial_distribution) :

```math
   
p(D|\mu) = \prod_{k=1}^N p(x_k|\mu) = \mu^n (1-\mu)^{N-n}
```

"""

# ╔═╡ 6a28d81e-d294-11ef-2a9f-d32daa5556ae
md"""
### $(HTML("<span id='beta-prior'>Prior</span>"))

Assume the prior beliefs for ``\mu`` are governed by a [**beta distribution**](https://en.wikipedia.org/wiki/Beta_distribution)

```math
p(\mu) = \mathrm{Beta}(\mu|\alpha,\beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \mu^{\alpha-1}(1-\mu)^{\beta-1}
```

where the [Gamma function](https://en.wikipedia.org/wiki/Gamma_function) is sort-of a generalized factorial function. In particular, if ``\alpha,\beta`` are integers, then 

```math
\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} = \frac{(\alpha+\beta-1)!}{(\alpha-1)!\,(\beta-1)!}
```

"""

# ╔═╡ 6a28e674-d294-11ef-391b-0d33fd609fb8
md"""
A *what* distribution? Yes, the **beta distribution** is a [**conjugate prior**](https://en.wikipedia.org/wiki/Conjugate_prior) for the binomial distribution, which means that 

```math
\underbrace{\text{beta}}_{\text{posterior}} \propto \underbrace{\text{binomial}}_{\text{likelihood}} \times \underbrace{\text{beta}}_{\text{prior}}
```

so we get a closed-form posterior.

"""

# ╔═╡ 6a28f466-d294-11ef-3af9-e34de9736c71
md"""

``\alpha`` and ``\beta`` are called **hyperparameters**, since they parameterize the distribution for another parameter (``\mu``). E.g., ``\alpha=\beta=1`` leads to a uniform prior for ``\mu``. We use Julia below to visualize some priors ``\mathrm{Beta}(\mu|\alpha,\beta)`` for different values of ``\alpha, \beta``.

"""

# ╔═╡ 51bed1cc-c960-46fe-bc09-2b684df3b0cc
# maintain a vector of log evidences to plot later
params = [
    (α=0.1, β=0.1)
    (α=1.0, β=1.0)
    (α=2.0, β=3.0)
    (α=8.0, β=4.0)
]

# ╔═╡ 513414c7-0a54-4767-a583-7d779f8fbc55
let
	x = 0:0.01:1
	
	plots = map(enumerate(params)) do (i, (α, β))
	    y = pdf.(Beta(α, β), x)
	    plot(x, y; 
			label="α=$α, β=$β", 
			xlabel=i in [3, 4] ? "μ" : nothing, 
			ylabel=i in [1, 3] ? "Density" : nothing,
		)
	end
	
	plot(plots...;
		layout=(2, 2),
		suptitle="PDFs of some Beta distributions",
		legend=:topleft,
		link=:both, 
		padding=10,
	)
end

# ╔═╡ 6a294790-d294-11ef-270b-5b2152431426
md"""
Before observing any data, you can express your state-of-knowledge about the coin by choosing values for ``\alpha`` and ``\beta`` that reflect your beliefs. Stronger yet, you *must* choose values for ``\alpha`` and ``\beta``, because the Bayesian framework does not allow you to walk away from your responsibility to explicitly state your beliefs before the experiment.  

"""

# ╔═╡ b872cd69-d534-4b04-bb76-d85bb7ef0ea9
md"""
## 2. Parameter Estimation for Coin Toss

Next, infer the posterior PDF over ``\mu`` (and evidence) through Bayes rule,
"""

# ╔═╡ 1ba1939d-9986-4b97-9273-4f2434f1d385
md"""
```math
\begin{flalign*}
p&(D|\mu)\cdot p(\mu)   \\
  &= \underbrace{\biggl(\frac{B(n+\alpha,N-n+\beta)}{B(\alpha,\beta)}\biggr)}_{\text{evidence }p(D)} \cdot \underbrace{\biggl( \frac{1}{B(n+\alpha,N-n+\beta)} \mu^{n+\alpha-1} (1-\mu)^{N-n+\beta-1}\biggr)}_{\text{posterior }p(\mu|D)=\mathrm{Beta}(\mu|n+\alpha, N-n+\beta)}
\end{flalign*}
```
where ``B(\alpha,\beta) \triangleq \frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha + \beta)}``. 
"""

# ╔═╡ b426df32-5629-4773-b862-101cfbd82d42
details("Proof this, and click for solution",
md"""
```math
\begin{flalign*}
p&(D|\mu)\cdot p(\mu)  \\
  &=  \underbrace{\biggl( \mu^n (1-\mu)^{N-n}\biggr)}_{\text{likelihood}} \cdot \underbrace{\biggl( \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \mu^{\alpha-1}(1-\mu)^{\beta-1} \biggr)}_{\text{prior}} \\
  &= \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \mu^{n+\alpha-1} (1-\mu)^{N-n+\beta-1} \\
  &= \frac{1}{B(\alpha,\beta)} \mu^{n+\alpha-1} (1-\mu)^{N-n+\beta-1} \\
  &= \underbrace{\biggl(\frac{B(n+\alpha,N-n+\beta)}{B(\alpha,\beta)}\biggr)}_{\text{evidence }p(D)} \cdot \underbrace{\biggl( \frac{1}{B(n+\alpha,N-n+\beta)} \mu^{n+\alpha-1} (1-\mu)^{N-n+\beta-1}\biggr)}_{\text{posterior }p(\mu|D)=\mathrm{Beta}(\mu|n+\alpha, N-n+\beta)}
\end{flalign*}
```
In the final equation, we included the term ``\frac{1}{B(n+\alpha,\,N-n+\beta)}`` to normalize the posterior ``p(\mu | D)``, and we compensated for this normalization in the evidence factor.	
		""")

# ╔═╡ 181ade96-8e1e-4186-9227-c1561352529d
md"""
Hence, the posterior is also beta-distributed as

```math
p(\mu|D) = \mathrm{Beta}(\mu|\,n+\alpha, N-n+\beta)
```

"""

# ╔═╡ 6a29d548-d294-11ef-1361-ad2230cad02b
md"""
## 3. Model Evaluation for Coin Toss

It follow from the above calculation that the evidence for model ``m`` can be analytically expressed as

```math
\begin{align}
p(D|m) &= \frac{B(n+\alpha,N-n+\beta)}{B(\alpha,\beta)} \\
\Big( &=  \frac{\Gamma(n+\alpha) \Gamma(N-n+\beta)}{\Gamma(N+\alpha+\beta)} \Bigg/ \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}\,. \Big)
\end{align}
```

The model evidence is a scalar. The absolute value is not important. However, you may want to compare the model evidence of this model to the evidence for another model on the same data set.  

"""

# ╔═╡ 6a29e25e-d294-11ef-15ce-5bf3d8cdb64c
md"""
## 4. Prediction (Application) for Coin Toss

Once we have accepted a model, let's apply it to the application, in this case, predicting future observations. 

"""

# ╔═╡ 6a29f1c2-d294-11ef-147f-877f99e5b57c
md"""
Marginalize over the parameter posterior to get the predictive PDF for a new coin toss ``x_\bullet``, given the data ``D``,

```math
\begin{align*}
p(x_\bullet=1|D)  &= \int_0^1 p(x_\bullet=1|\mu)\,p(\mu|D) \,\mathrm{d}\mu \\
  &= \int_0^1 \mu \times  \mathrm{Beta}(\mu|\,n+\alpha, N-n+\beta) \,\mathrm{d}\mu  \\
  &= \frac{n+\alpha}{N+\alpha+\beta}
\end{align*}
```

This result is known as [**Laplace's rule of succession**](https://en.wikipedia.org/wiki/Rule_of_succession).

"""

# ╔═╡ 6a2a000e-d294-11ef-17d6-bdcddeedc65d
md"""
The above integral computes the mean of a beta distribution, which is given by ``\mathbb{E}[x] = \frac{a}{a+b}`` for ``x \sim \mathrm{Beta}(a,b)``, see [wikipedia](https://en.wikipedia.org/wiki/Beta_distribution).

"""

# ╔═╡ 6a2a0f18-d294-11ef-02c2-ef117377ca66
md"""
Finally, we're ready to solve our challenge: for ``D=\{1011001\}`` and uniform prior (``\alpha=\beta=1``), we get

```math
 p(x_\bullet=1|D)=\frac{n+1}{N+2} = \frac{4+1}{7+2} = \frac{5}{9}
```

In other words, given the model assumptions (the Bernoulli data-generating distribution and Beta prior as specified above), and the observations ``D=\{1011001\}``, the probability for observing heads (outcome=``1``) on the next toss is ``\frac{5}{9}``.

"""

# ╔═╡ 6a2a1daa-d294-11ef-2a67-9f2ac60a14c5
md"""
Be aware that there is no such thing as an "objective" or "correct" prediction. Every prediction is conditional on the selected model and the used data set. 

"""

# ╔═╡ 6a2a2af2-d294-11ef-0072-bdc3c6f95bb3
md"""
## What did we learn from the data?

What did we learn from the data? Before seeing any data, we think that the probability for throwing heads is 

```math
\left. p(x_\bullet=1|D) \right|_{n=N=0} = \left.\frac{n+\alpha}{N+\alpha+\beta}\right|_{n=N=0} = \frac{\alpha}{\alpha + \beta}\,.
```

"""

# ╔═╡ 6a2a389e-d294-11ef-1b8c-b55de794b65c
md"""
Hence, ``\alpha`` and ``\beta`` can be interpreted as prior pseudo-counts for heads and tails, respectively. 

"""

# ╔═╡ 6a2a465e-d294-11ef-2aa0-43c954a6439e
md"""
If we were to assume zero pseudo-counts, i.e. ``\alpha=\beta \rightarrow 0``, then our prediction for throwing heads after ``N`` coin tosses is completely based on the data, given by

```math
\left. p(x_\bullet=1|D) \right|_{\alpha=\beta \rightarrow 0} = \left.\frac{n+\alpha}{N+\alpha+\beta}\right|_{\alpha=\beta \rightarrow 0} = \frac{n}{N}\,.
```

"""

# ╔═╡ 48fd2dff-796d-48bc-b5a8-bee270d119fd
md"""
Note the following decomposition
"""

# ╔═╡ e3f9e571-2248-403c-8ab8-f6b99597f595
md"""
```math
\begin{flalign*}
    p(x_\bullet=1|\,D) &= \frac{n+\alpha}{N+\alpha+\beta} \\
        &= \underbrace{\frac{\alpha}{\alpha+\beta}}_{\substack{\text{prior}\\\text{prediction}}} + \underbrace{\underbrace{\frac{N}{N+\alpha+\beta}}_{\text{gain}}\cdot \underbrace{\biggl( \underbrace{\frac{n}{N}}_{\substack{\text{data-based}\\\text{prediction}}} - \underbrace{\frac{\alpha}{\alpha+\beta}}_{\substack{\text{prior}\\\text{prediction}}} \biggr)}_{\text{prediction error}}}_{\text{correction}}
\end{flalign*}
```

""" 

# ╔═╡ 90f691ad-046c-4595-99b0-19a1d6cb599e
details("Proof this yourself, and click for solution",
md"""
```math
\begin{align*}
    p(x_\bullet=1|\,D) &= \frac{n+\alpha}{N+\alpha+\beta} \\
    &= \frac{\alpha}{N+\alpha+\beta} + \frac{n}{N+\alpha+\beta}  \\
    &= \frac{\alpha}{N+\alpha+\beta}\cdot \frac{\alpha+\beta}{\alpha+\beta} + \frac{n}{N+\alpha+\beta}\cdot \frac{N}{N}  \\
    &= \frac{\alpha}{\alpha+\beta}\cdot \frac{\alpha+\beta}{N+\alpha+\beta} + \frac{N}{N+\alpha+\beta}\cdot \frac{n}{N}  \\
    &= \frac{\alpha}{\alpha+\beta}\cdot \biggl(1-\frac{N}{N+\alpha+\beta} \biggr) + \frac{N}{N+\alpha+\beta}\cdot \frac{n}{N}  \\
        &= \underbrace{\frac{\alpha}{\alpha+\beta}}_{\substack{\text{prior}\\\text{prediction}}} + \underbrace{\underbrace{\frac{N}{N+\alpha+\beta}}_{\text{gain}}\cdot \underbrace{\biggl( \underbrace{\frac{n}{N}}_{\substack{\text{data-based}\\\text{prediction}}} - \underbrace{\frac{\alpha}{\alpha+\beta}}_{\substack{\text{prior}\\\text{prediction}}} \biggr)}_{\text{prediction error}}}_{\text{correction}}
\end{align*}
```
		""")

# ╔═╡ 6a2a9faa-d294-11ef-1284-cfccb1da444e
md"""
Let's interpret this decomposition of the posterior prediction. Before the data ``D`` was observed, our model generated a *prior prediction* ``p(x_\bullet=1) = \frac{\alpha}{\alpha+\beta}``. Next, the degree to which the actually observed data matches this prediction is represented by the *prediction error* ``\frac{n}{N} - \frac{\alpha}{\alpha-\beta}``. The prior prediction is then updated to a *posterior prediction* ``p(x_\bullet=1|D)`` by adding a fraction of the prediction error to the prior prediction. Hence, the data plays the role of "correcting" the prior prediction. 

Note that, since ``0\leq \underbrace{\frac{N}{N+\alpha+\beta}}_{\text{gain}} \lt 1``, the Bayesian prediction lies between (fuses) the prior and data-based predictions.

"""

# ╔═╡ 6a2aad42-d294-11ef-3129-3be5be8c82d6
md"""
For large ``N``, the gain goes to ``1`` and ``\left. p(x_\bullet=1|D)\right|_{N\rightarrow \infty} \rightarrow \frac{n}{N}`` goes to the data-based prediction (the observed relative frequency).

"""

# ╔═╡ 6a2abb16-d294-11ef-0243-d376e8a39bb0
section_outline("Code Example:", "Bayesian Evolution for the Coin Toss")

# ╔═╡ 6a2acb7e-d294-11ef-185c-9d49ce79c31b
md"""
Let's code an example for a sequence of coin tosses, where we assume that the true coin generates data ``x_n \in \{0,1\}`` by a Bernoulli distribution:

```math
p(x_n|\mu=0.4)=0.4^{x_n} \cdot 0.6^{1-x_n}
```

So, this coin is biased!

To predict the outcomes of future coin tosses, we'll compare **two models**. Both models have the same data-generating distribution (also Bernoulli):

```math
p(x_n|\mu,m_k) = \mu^{x_n} (1-\mu)^{1-x_n} \quad \text{for }k=1,2 \,,
```

but they have different priors:

```math
\begin{aligned}
p(\mu|m_1) &= \mathrm{Beta}(\mu|\alpha=100,\beta=500) \\
p(\mu|m_2) &= \mathrm{Beta}(\mu|\alpha=8,\beta=13). \\
\end{aligned}
```

You can verify that model ``m_2`` has the best prior, since

```math
\begin{align*}
p(x_n=1|m_1) &= \left.\frac{\alpha}{\alpha+\beta}\right|_{m_1} = 100/600 \approx 0.17 \\
p(x_n=1|m_2) &= \left.\frac{\alpha}{\alpha+\beta}\right|_{m_2} = 8/21 \approx 0.38 \,,
\end{align*}
```

(but you are not supposed to know that the real coin has a probability ``0.4`` for heads.) 

Let's run ``500`` tosses:

"""

# ╔═╡ 8bfc4f37-4bf8-42a3-bd55-f046c8d2624a
TODO("What code do we want to show, and what do we want to hide? We might want to move cells with hidden code to the end of this section.")

# ╔═╡ 51829800-1781-49ae-8ee7-ac15c0bfcb88
# computes log10 of Gamma function
function log10gamma(num)
    num = convert(BigInt, num)
    return log10(gamma(num))
end

# ╔═╡ de7a1b82-f1c4-4eff-b372-ac76cf11c015
μ  = 0.4;                        # specify model parameter

# ╔═╡ d1d2bb84-7083-435a-9c19-4c02074143e3
n_tosses = 500                   # specify number of coin tosses

# ╔═╡ 9c751f8e-f7ed-464f-b63c-41e318bbff2d
samples = rand(n_tosses) .<= μ   # Flip 500 coins

# ╔═╡ e99e7650-bb72-4576-8f2a-c3994533b644
function handle_coin_toss(prior::Beta, observation::Bool)
    posterior = Beta(prior.α + observation, prior.β + (1 - observation))
	return posterior
end

# ╔═╡ 7a624d2f-812a-47a0-a609-9fe299de94f5
function log_evidence_prior(prior::Beta, N::Int64, n::Int64)
    log10gamma(prior.α + prior.β) - 
	log10gamma(prior.α) - 
	log10gamma(prior.β) + 
	log10gamma(n+prior.α) + 
	log10gamma((N-n)+prior.β) - 
	log10gamma(N+prior.α+prior.β)
end

# ╔═╡ 3a903a4d-1fb0-4566-8151-9c86dfc40ceb
begin
	priors = [Beta(100., 500.), Beta(8., 13.)]  # specify prior distributions 
	n_models = length(priors)
	
	# save a sequence of posterior distributions for every prior, starting with the prior itself
	posterior_distributions = [[d] for d in priors] 
	log_evidences = [[] for _ in priors] 

	# for every sample we want to update our posterior
	for (N, sample) in enumerate(samples)
		# at every sample we want to update all distributions
	    for (i, prior) in enumerate(priors)

			# do bayesian updating
	        posterior = handle_coin_toss(prior, sample)
			
			# add posterior to vector of posterior distributions
	        push!(posterior_distributions[i], posterior)
	        
	        # compute log evidence and add to vector
	        log_evidence = log_evidence_prior(posterior_distributions[i][N], N, sum(samples[1:N]))
	        push!(log_evidences[i], log_evidence)
	
	        # the prior for the next sample is the posterior from the current sample
	        priors[i] = posterior
	    end
	end
end

# ╔═╡ 6a2af90a-d294-11ef-07bd-018326577791
md"""
For each model, as a function of the number of coin tosses, we plot the evolution of the parameter posteriors 

```math
p(\mu|D_n,m_\bullet)
```

"""

# ╔═╡ d484c41d-9834-4528-bf47-93ab4e35ebaa
md"""
Select iteration: $(@bind toss_index_1 Slider(1:n_tosses; show_value=true))
"""

# ╔═╡ 6a2b1106-d294-11ef-0d64-dbc26ba3eb44
# Animate posterior distributions over time in a gif

let i = toss_index_1
    p = plot(title=string("n = ", i))
    for j in 1:n_models
        plot!(posterior_distributions[j][i+1], xlims = (0, 1), fill=(0, .2,), label=string("Posterior m", j), linewidth=2, ylims=(0,28), xlabel="μ")
    end
	p
end

# ╔═╡ 6a2b2d44-d294-11ef-33ba-15db357708b1
md"""
Note that both posteriors move toward the "correct" value (``\mu=0.4``). However, the posterior for ``m_1`` (blue) moves much slower because we assumed far more pseudo-observations for ``m_1`` than for ``m_2``. 

As we get more observations, the influence of the prior diminishes. 

"""

# ╔═╡ 6a2b3ba4-d294-11ef-3c28-176be260cb15
md"""
We have an intuition that ``m_2`` is superior over ``m_1``. Let's check this by plotting over time the relative Bayesian evidences for each model:

```math
\frac{p(D_n|m_i)}{\sum_{i=1}^2 p(D_n|m_i)}
```

"""

# ╔═╡ c69c591f-1947-4b07-badb-3882fd097785
evidences = map(model -> exp.(model), log_evidences)

# ╔═╡ ebcfcd1b-7fc8-42b7-a35e-4530f798cfdf
md"""
Select iteration: $(@bind toss_index_2 Slider(1:n_tosses; show_value=true))
"""

# ╔═╡ 188b5bea-6765-4dcf-9369-3b1fdbe94494
let i = toss_index_2
	p = plot(title=string(L"\frac{p_i(\mathbf{x}_{1:n})}{\sum_i p_i(\mathbf{x}_{1:n})}","   n = ", i), ylims=(0, 1), legend=:topleft)
    total = sum([evidences[j][i] for j in 1:n_models])
    bar!([(evidences[j][i] / total) for j in 1:n_models], group=["Model $i" for i in 1:n_models])
end

# ╔═╡ 84e7ff22-e232-4ab7-a206-ccdd943043dd


# ╔═╡ 6a2b9676-d294-11ef-241a-89ff7aa676f9
md"""
Over time, the relative evidence of model ``m_1`` converges to ``0``. Can you explain this behavior?

"""

# ╔═╡ 9c5d7c89-f65c-4f52-9e49-14692bed2452
md"""
# Maximum Likelihood Estimation
"""

# ╔═╡ 6a2bb18a-d294-11ef-23bb-99082caf6e01
md"""
## From Posterior to Point-Estimate

In the example above, Bayesian parameter estimation and prediction were tractable in closed form. This is often not the case. In that case, we will need to approximate some of the computations. 

"""

# ╔═╡ 6a2bd3ac-d294-11ef-0543-6fe202ca35b6
md"""
Recall Bayesian prediction

```math
p(x|D) = \int p(x|\theta)p(\theta|D)\,\mathrm{d}{\theta}
```

"""

# ╔═╡ 6a2bf332-d294-11ef-1ff1-cdbfb7732cf1
md"""
If we approximate the posterior by a delta function, i.e., ``p(\theta|D) = \delta(\theta-\hat\theta)`` for one "best" value ``\hat\theta``, then the predictive distribution collapses to

```math
p(x|D)= \int p(x|\theta)\,\delta(\theta-\hat\theta)\,\mathrm{d}{\theta} = p(x|\hat\theta)
```

"""

# ╔═╡ 6a2c008e-d294-11ef-2f07-11cdfb2bddca
md"""
This is just the data-generating distribution ``p(x|\theta)`` evaluated at ``\theta=\hat\theta``, which is easy to evaluate.

"""

# ╔═╡ 6a2c11e6-d294-11ef-173b-23fc6dbfefca
md"""
The next question is how to get the parameter estimate ``\hat{\theta}``? (See next slide).

"""

# ╔═╡ 6a2c229e-d294-11ef-2f24-ebe43cbfbfa4
md"""
## Some Well-known Point-Estimates

- **Bayes estimate** (the mean of the posterior)

```math
\hat \theta_{\text{Bayes}}  = \int \theta \, p\left( \theta |D \right)
\,\mathrm{d}{\theta}
```

"""

# ╔═╡ 6a2c3036-d294-11ef-23cb-c3b36c475e8f
md"""
- **Maximum A Posteriori** (MAP) estimate 

```math
\hat \theta_{\text{map}}=  \arg\max _{\theta} p\left( \theta |D \right) =
\arg \max_{\theta}  p\left(D |\theta \right) \, p\left(\theta \right)
```

"""

# ╔═╡ 6a2c4058-d294-11ef-2312-d9c672d49701
md"""
- **Maximum Likelihood** (ML) estimate

```math
\hat \theta_{ml}  = \arg \max_{\theta}  p\left(D |\theta\right)
```

Note that Maximum Likelihood (ML) is MAP with a uniform prior. MAP is sometimes called a 'penalized' ML procedure:

```math
\hat \theta_{map}  = \arg \max _\theta  \{ \underbrace{\log
p\left( D|\theta  \right)}_{\text{log-likelihood}} + \underbrace{\log
p\left( \theta \right)}_{\text{penalty}} \}
```

ML is the most common approximation to the full Bayesian posterior.

"""

# ╔═╡ 6a2c505c-d294-11ef-1c92-c1b0e9d50da5
md"""
## Bayesian vs Maximum Likelihood Learning

Consider the task: predict a future observation ``x`` from an observed data set ``D``. Let us compare full Bayesian modeling with the maximum likelihood approach. 

"""

# ╔═╡ 7c8b1add-085a-41ba-9d6c-b26d3eef22e4
md"""

|        | **Bayesian**             | **Maximum Likelihood**             |
|:----|:---------|:-----|
| 1. **Model Specification** | Choose a model ``m`` with data-generating distribution ``p(x\|\theta, m)`` and parameter prior ``p(\theta\|m)``.       | Choose a model ``m`` with same data generating distribution ``p(x\|\theta, m)``. No need for priors. |
| 2. **Learning**             | Use Bayes rule to find the parameter posterior: $(HTML("<br>"))``p(\theta\|D) \propto p(D\|\theta) p(\theta)``                   | By Maximum Likelihood (ML) optimization: $(HTML("<br>")) ``\hat \theta = \arg \max_{\theta} p(D\|\theta)``         |
| 3. **Prediction**           | ``p(x\|D) = \int p(x\|\theta) p(\theta\|D) \,\mathrm{d}\theta``                                                         | ``p(x\|D) = p(x\|\hat\theta)``                                                                   |


"""

# ╔═╡ 6a2c5e08-d294-11ef-213d-97bcfa16eb5a
md"""
## Report Card on Maximum Likelihood Estimation



"""

# ╔═╡ 6a2c7230-d294-11ef-05a2-3ff2f65d10e0
md"""
(good!). ML works rather well if we have a lot of data because the influence of the prior diminishes with more data.

"""

# ╔═╡ 6a2c7f5a-d294-11ef-2e17-9108a39df280
md"""
(good!). Computationally often do-able. Useful fact that makes the optimization easier (since ``\log`` is monotonously increasing):

```math
\arg\max_\theta \log p(D|\theta) =  \arg\max_\theta p(D|\theta)
```

"""

# ╔═╡ 6a2c8f4a-d294-11ef-213c-dfa929a403bc
md"""
(bad). ML cannot be used for model comparison! In ML estimation, the Bayesian model evidence is undefined because no prior distribution is specified. Even if we attempt to simulate ML as a special case of Bayesian inference by using a uniform prior, the evidence still collapses: a uniform prior over the entire real line is not a proper probability distribution, since its integral does not evaluate to 1. Consequently, when performing ML estimation, Bayesian model evidence cannot be used to evaluate model performance:

```math
\begin{align*}
\underbrace{p(D|m)}_{\substack{\text{Bayesian}\\ \text{evidence}}} &= \int p(D|\theta) \cdot p(\theta|m)\,\mathrm{d}\theta \\
  &= \lim_{(b-a)\rightarrow \infty} \int p(D|\theta)\cdot \underbrace{\text{Uniform}(\theta|a,b)}_{\text{"ML prior"}}\,\mathrm{d}\theta \\
  &= \lim_{(b-a)\rightarrow \infty} \frac{1}{b-a}\underbrace{\int_a^b p(D|\theta)\,\mathrm{d}\theta}_{<\infty}  \\
    &= 0
\end{align*}
```

In fact, this is a serious disadvantage because Bayesian evidence is a principled performance assessment criterion that follows from straightforward PT. In practice, when estimating parameters by maximum likelihood, we often evaluate model performance by an *ad hoc* performance measure such as mean-squared-error on a testing data set.

"""

# ╔═╡ 6a2ca496-d294-11ef-0043-1f350b36773e
keyconcept(" ", 
	md"""
	
	**Maximum Likelihood estimation is at best an approximation to Bayesian learning**, but for good reason, a very popular learning method when faced with lots of available data.
	"""
)


# ╔═╡ f2969d91-4a5b-4665-9fa5-521db750302f
md"""
$(section_outline("Excercises","",header_level=1))

#####  Bayes estimate (**)

(##) The Bayes estimate is a summary of a posterior distribution by a delta distribution on its mean, i.e.,

```math
\hat \theta_{bayes}  = \int \theta \, p\left( \theta |D \right)
\,\mathrm{d}{\theta}
```

Proof that the Bayes estimate minimizes the mean-squared error, i.e., proof that

```math
\hat \theta_{bayes} = \arg\min_{\hat \theta} \int_\theta (\hat \theta -\theta)^2 p \left( \theta |D \right) \,\mathrm{d}{\theta}
```
"""

# ╔═╡ 7dd9a456-9dca-47c8-98c5-51f87f28e6a4
details("Click for solution",
md"""
To minimize the expected mean-squared error we will look for ``\hat{\theta}`` that makes the gradient of the integral with respect to ``\hat{\theta}`` vanish.

```math
\begin{align*}
  \nabla_{\hat{\theta}}  \int_\theta (\hat \theta -\theta)^2 p \left( \theta |D \right) \,\mathrm{d}{\theta} &= 0 \\
  \int_\theta \nabla_{\hat{\theta}}  (\hat \theta -\theta)^2 p \left( \theta |D \right) \,\mathrm{d}{\theta} &= 0 \\
  \int_\theta  2(\hat \theta -\theta) p \left( \theta |D \right) \,\mathrm{d}{\theta} &= 0 \\
  \int_\theta  \hat \theta p \left( \theta |D \right) \,\mathrm{d}{\theta} &= \int_\theta  \theta p \left( \theta |D \right) \,\mathrm{d}{\theta} \\
  \hat \theta \underbrace{\int_\theta p \left( \theta |D \right) \,\mathrm{d}{\theta}}_{1} &= \int_\theta  \theta p \left( \theta |D \right) \,\mathrm{d}{\theta} \\
  \Rightarrow \hat \theta &= \int_\theta  \theta p \left( \theta |D \right) \,\mathrm{d}{\theta}
\end{align*}
```
"""
)

# ╔═╡ b2820dfd-b3ca-477b-8cb7-c430e0fe18dd
md"""

##### Coin Toss MAP and ML (**)

Consider the coin toss example with model
```math
\begin{align}
p(x_k|\mu) &= \mu^{x_k} (1-\mu)^{1-x_k} \\ 
p(\mu) &= \mathrm{Beta}(\mu|\alpha,\beta) \,.
\end{align}
```
and a given data set ``D=\{x_1, x_2,\ldots,x_N\}``.

- (a) Derive the Maximum Likelihood estimate for ``\mu``.
- (b) Derive the MAP estimate for ``\mu``.           
- (c) Do these two estimates ever coincide (if so, under what circumstances)?


"""

# ╔═╡ 664d4183-edb6-4818-a44b-bf4c0a22a33c
details("Click for solution",
md"""
- (a) The likelihood is given by ``p(D|\mu) = \mu^n\cdot (1-\mu)^{(N-n)}``. It follows that


```math
\begin{align*}
    \nabla \log p(D|\mu) &= 0 \\
    \nabla \left( n\log \mu + (N-n)\log(1-\mu)\right) &= 0\\
    \frac{n}{\mu} - \frac{N-n}{1-\mu} &= 0 \\
    \rightarrow \hat{\mu}_{\text{ML}} &= \frac{n}{N}
  \end{align*}
```

- (b) We can write the posterior as as


```math
\begin{align*}
   p(\mu|D) &\propto p(D|\mu)p(\mu) \\
      &\propto \mu^n (1-\mu)^{N-n} \mu^{\alpha-1} (1-\mu)^{\beta-1} \\
      &\propto \mathcal{B}(\mu|n+\alpha,N-n+\beta)
   \end{align*}
```

The MAP estimate for a beta distribution ``\mathcal{B}(a,b)`` is located at ``\frac{a - 1}{a+b-2}``, see [wikipedia](https://en.wikipedia.org/wiki/Beta_distribution). Hence,


```math
\begin{align*}
\hat{\mu}_{\text{MAP}} &= \frac{(n+\alpha)-1}{(n+\alpha) + (N-n+\beta) -2} \\
  &= \frac{n+\alpha-1}{N + \alpha +\beta -2}
\end{align*}
```

- (c) As ``N`` gets larger, the MAP estimate approaches the ML estimate. In the limit the MAP solution converges to the ML solution.


"""
)

# ╔═╡ ecb036da-a0a2-4919-b1aa-bc33b6ba7e73
md"""

##### Model Comparison (**)

A model ``m_1`` is described by a single parameter ``\theta``, with ``0 \leq \theta \leq 1``. The system can produce data ``x \in \{0,1\}``. The sampling distribution and prior are given by

```math
\begin{align*}
p(x|\theta,m_1) &=  \theta^x (1-\theta)^{(1-x)} \\
p(\theta|m_1) &= 6\theta(1-\theta)
\end{align*}
```

- (a) Work out the probability ``p(x=1|m_1)``.    

- (b) Determine the posterior ``p(\theta|x=1,m_1)``.     

Now consider a second model ``m_2`` with the following sampling distribution and prior on ``0 \leq \theta \leq 1``:

```math
\begin{align*}
p(x|\theta,m_2) &= (1-\theta)^x \theta^{(1-x)} \\
p(\theta|m_2) &= 2\theta
\end{align*}
```

- (c) Determine the probability ``p(x=1|m_2)``.    

Now assume that the model priors are given by

```math
\begin{align*}
    p(m_1) &= 1/3  \\
    p(m_2) &= 2/3
    \end{align*}
```

- (d) Compute the probability ``p(x=1)`` by "Bayesian model averaging", i.e., by weighing the predictions of both models appropriately.  


- (e) Compute the fraction of posterior model probabilities ``\frac{p(m_1|x=1)}{p(m_2|x=1)}``.     


- (f) Which model do you prefer after observation ``x=1``?


"""

# ╔═╡ de08c2a1-c5e3-4add-8b22-2c633247da48
details("Click for solutions",
md"""
- (a) Work out the probability ``p(x=1|m_1)``.    

```math
\begin{align*}
  p(x=1|m_1) &= \int_0^1 p(x=1|\theta,m_1) p(\theta|m_1) \mathrm{d}\theta \\
  &= \int \theta \cdot 6\theta (1-\theta) \mathrm{d}\theta \\
  &= 6 \cdot \left(\frac{1}{3}\theta^3 - \frac{1}{4}\theta^4\right) \bigg|_0^1 \\
  &= 6 \cdot (\frac{1}{3} - \frac{1}{4}) = \frac{1}{2}
\end{align*}
```

- (b) Determine the posterior ``p(\theta|x=1,m_1)``.     

```math
\begin{align*}
  p(\theta|x=1,m_1) &= \frac{p(x=1|\theta) p(\theta|m_1)}{p(x=1|m_1)} \\
  &= 2\cdot \theta \cdot 6\theta (1-\theta) \\
  &= \begin{cases} 12 \theta^2 (1-\theta) & \text{if }0 \leq \theta \leq 1 \\
  0 & \text{otherwise} \end{cases}
  \end{align*}
```

Now consider a second model ``m_2`` with the following sampling distribution and prior on ``0 \leq \theta \leq 1``:

```math
\begin{align*}
p(x|\theta,m_2) &= (1-\theta)^x \theta^{(1-x)} \\
p(\theta|m_2) &= 2\theta
\end{align*}
```

- (c) Determine the probability ``p(x=1|m_2)``.    

```math
\begin{align*}
  p(x=1|m_2) &= \int_0^1 p(x=1|\theta,m_2) p(\theta|m_2) \mathrm{d}\theta \\
  &= \int (1-\theta) \cdot 2\theta \mathrm{d}\theta \\
  &= 2 \cdot \left( \frac{1}{2}\theta^2 - \frac{1}{3}\theta^3 \right) \bigg|_0^1 \\
  &= 2 \cdot (\frac{1}{2} - \frac{1}{3}) = \frac{1}{3}
  \end{align*}
```

Now assume that the model priors are given by

```math
\begin{align*}
    p(m_1) &= 1/3  \\
    p(m_2) &= 2/3
    \end{align*}
```

- (d) Compute the probability ``p(x=1)`` by "Bayesian model averaging", i.e., by weighing the predictions of both models appropriately.  

```math
\begin{align*}
    p(x=1) &= \sum_{k=1}^2 p(x=1|m_k) p(m_k)  \\
    &= \frac{1}{2} \cdot \frac{1}{3} + \frac{1}{3} \cdot \frac{2}{3} = \frac{7}{18} 
    \end{align*}
```

- (e) Compute the fraction of posterior model probabilities ``\frac{p(m_1|x=1)}{p(m_2|x=1)}``.     

```math
\frac{p(m_1|x=1)}{p(m_2|x=1)} = \frac{p(x=1|m_1) p(m_1)}{p(x=1|m_2) p(m_2)} = \frac{\frac{1}{2} \cdot \frac{1}{3}}{\frac{1}{3} \cdot \frac{2}{3}} =\frac{3}{4}
```

- (f) Which model do you prefer after observation ``x=1``?

In principle, the observation ``x=1`` favors model ``m_2``, since ``p(m_2|x=1) = \frac{4}{3} \times p(m_1|x=1)``. However, note that ``\log_{10} \frac{3}{4} \approx -0.125``, so the extra evidence for ``m_2`` relative to ``m_1`` is very low. At this point, after 1 observation, we have no preference for a model yet.

""")

# ╔═╡ 6a2cb25e-d294-11ef-1d88-1fc784b33df0
md"""
# Optional Slides

"""

# ╔═╡ 1edae118-dcc7-4169-95cf-f36025f2c336
md"""
## Working with Distributions in code

Take a look at this mini lecture to see some simple examples of using distributions in Julia:
"""

# ╔═╡ 275a9a69-3135-4cbd-8a35-b1abee4af83f
NotebookCard("https://bmlip.github.io/course/minis/Distributions%20in%20Julia.html")

# ╔═╡ 6a2ccd16-d294-11ef-22ee-a5cff62ccd9c
md"""
## The Kullback-Leibler Divergence

The $(HTML("<span id='KLD'>Kullback-Leibler Divergence</span>")) (a.k.a. relative entropy) between two distributions ``q`` and ``p`` is defined as

```math
D_{\text{KL}}[q,p] \equiv \sum_z q(z) \log \frac{q(z)}{p(z)}
```

The following [Gibbs Inequality](https://en.wikipedia.org/wiki/Gibbs%27_inequality) holds (see [wikipedia](https://en.wikipedia.org/wiki/Gibbs%27_inequality) for proof): 

```math
D_{\text{KL}}[q,p] \geq 0 \quad \text{with equality only if } p=q 
```

The KL divergence can be interpreted as a distance between two probability distributions.

As an aside, note that ``D_{\text{KL}}[q,p] \neq D_{\text{KL}}[p,q]``. Both divergences are relevant. 

"""

# ╔═╡ 6a2cd9be-d294-11ef-33cf-4b23b92e1cbf
md"""
Here is an animation that shows the KL divergence between two Gaussian distributions:

"""

# ╔═╡ 10306ea6-6092-463c-9315-7a216c83606e
function kullback_leibler(q::Normal, p::Normal)                                 
    # Calculates the KL Divergence between two gaussians 
    # (see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence for calculations)
    return log((p.σ / q.σ)) + ((q.σ)^2 + (p.μ - q.μ)^2) / (2*p.σ^2) - (1/2)
end

# statistics of distributions we'll keep constant (we'll vary the mean of q)
# feel free to change these and see what happens

# ╔═╡ ff20449d-1489-430a-aeee-a3a66bece706
μ_p = 0; σ_p = 1;

# ╔═╡ b47a2e71-48bb-4cc7-9a14-c0e654c5d2f8
σ_q = 1;

# ╔═╡ 9a58bf5d-f072-4572-bb90-9b860133dce8
p = Normal(μ_p, σ_p)

# ╔═╡ 635e4f4c-5274-4bd4-a940-b2f3819426ec
@bind KL_animation_step Slider(1:100)

# ╔═╡ 7bf0fde7-b201-4646-9934-ec93e661cf22
let i = KL_animation_step
	# sequence of means tested so far (to compute sequence of KL divergences)
    μ_seq = [(j / 10.) - 5. + μ_p for j in 1:i]
	# KL divergence data
    kl = [kullback_leibler(Normal(μ, σ_q), p) for μ in μ_seq]

	
    viz = plot(; 
			   right_margin=8mm, 
			   title=string(L"D_{KL}(Q || P) = ", round(100 * kl[i]) / 100.), 
			   legend=:topleft
	)

	# extract mean of current frame from mean sequence
    μ_q = μ_seq[i]
    q = Normal(μ_q, σ_q)

    plot!(p;
		  xlims = (μ_p - 8, μ_p + 8), 
		  fill=(0, .2,), 
		  label="P", 
		  linewidth=2, 
		  ylims=(0,0.5),
	)
    plot!(q;
		  fill=(0, .2,), 
		  label="Q",
		  linewidth=2, 
		  ylims=(0,0.5),
	)
	# plot KL divergence data with different y-axis scale and different legend 
    plot!(twinx(), μ_seq, kl;
		  xlims = (μ_p - 8, μ_p + 8),
		  ylims=(0, maximum(kl) + 3), 
		  xticks=:none, 
		  linewidth = 3,                                                             
		  color="green",
		  legend=:topright,
		  label=L"D_{KL}(Q || P)",
	)
end

# ╔═╡ 1f92c406-6792-4af6-9132-35efd8223bc5
md"""
# Appendix
"""

# ╔═╡ 7a764a14-a5df-4f76-8836-f0a571fc3519
wideq(x) = PlutoUI.ExperimentalLayout.Div([x]; style="min-width: max-content;") |> WideCell

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.119"
LaTeXStrings = "~1.4.0"
Plots = "~1.40.13"
PlutoTeachingTools = "~0.4.5"
PlutoUI = "~0.7.69"
SpecialFunctions = "~2.5.1"
StatsPlots = "~0.15.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "abd58c9f6700767f652d81b5f75ddf0c01e33a11"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "a656525c8b46aa6a1c76891552ed5381bb32ae7b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.30.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "0037835448781bb46feb39866934e243886d756a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3e6d038b77f22791b8e3472b7c633acea1ecac06"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.120"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "83dc665d0312b41367b7263e8a4d172eac1897f4"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.4"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3a948313e7a41eb1db7a1e733e6335f17b4ab3c4"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "7.1.1+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "797762812ed063b9b94f6cc7742bc8883bb5e69e"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.9.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "1828eb7275491981fa5f1752a5e126e8f26f8741"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.17"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "27299071cc29e409488ada41ec7643e0ab19091f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.17+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "35fbd0cefb04a516104b8e183ce0df11b70a3f1a"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.84.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "ed5e9c58612c4e081aecdb6e1a479e18462e041e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "ba51324b894edaf1df3ab16e2cc6bc3280a2f1a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.10"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "52e1296ebbde0db845b356abbbe67fb82a0a116c"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.9"

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

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ca7e18198a166a1f3eb92a3650d53d94ed8ca8a1"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.22"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+4"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2ae7d4ddec2e13ad3bddf5c0796f7547cf682391"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.2+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "275a9a6d85dc86c24d03d1837a0010226a96f540"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.3+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "9a9216c0cf706cb2cc58fd194878180e3e51e8c0"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.18"

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
git-tree-sha1 = "85778cdf2bed372008e6646c64340460764a5b85"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "2d7662f95eafd3b6c346acdbfc11a762a2256375"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.69"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "eb38d376097f47316fe089fc62cb7c6d85383a52"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "e1d5e16d0f65762396f9ca4644a5f4ddab8d452b"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "cbea8a6bd7bed51b1619658dec70035e07b8502f"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.14"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2c962245732371acd51700dbb268af311bddd719"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.6"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

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
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "6258d453843c466d84c17a58732dda5deeb8d3af"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.24.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "af305cc62419f9bd61b6644d19170a4d258c7967"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.7.0"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "e9aeb174f95385de31e70bd15fa066a505ea82b9"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.7"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "9caba99d38404b285db8801d5c45ef4f4f425a6d"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.1+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "c5bf2dad6a03dfef57ea0a170a1fe493601603f2"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.5+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4bba74fa59ab0755167ad24f98800fe5d727175b"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.12.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
"""

# ╔═╡ Cell order:
# ╟─6a23b828-d294-11ef-371a-05d061144a43
# ╟─6be2e966-4048-44d0-a37e-95060e3fe30b
# ╟─6a23df9e-d294-11ef-3ddf-a51d4cea00fc
# ╟─6a24376c-d294-11ef-348a-e9027bd0ec29
# ╟─6a24b9e4-d294-11ef-3ead-9d272fbf89be
# ╟─6a24c3e6-d294-11ef-3581-2755a9ba15ba
# ╟─e2de9415-7bd8-4e95-abeb-53fc068ee950
# ╟─6a24c9f4-d294-11ef-20cc-172ea50da901
# ╟─6a24cee0-d294-11ef-35cb-71ab9ef935e5
# ╟─6a24d478-d294-11ef-2a75-9d03a5ba7ff8
# ╟─6a24fde8-d294-11ef-29bf-ad3e20a53c29
# ╟─a75c75ed-c67b-4be2-adbf-8984f27fc05d
# ╟─6a251a08-d294-11ef-171a-27b9d0f818bc
# ╟─6a252250-d294-11ef-33cd-89b18066817d
# ╟─6a25307e-d294-11ef-0662-3db678b32e99
# ╟─6a25379a-d294-11ef-3e07-87819f6d75cb
# ╟─6a254460-d294-11ef-1890-230b75b6b9ee
# ╟─6a2552ac-d294-11ef-08d6-179e068bc297
# ╟─ce75e785-868f-4361-93f8-c582ac1b891b
# ╟─6a2561c0-d294-11ef-124d-373846e3120c
# ╟─6a257020-d294-11ef-0490-e151934b2f42
# ╟─6a257f34-d294-11ef-2928-fbb800e81124
# ╟─6a25a11e-d294-11ef-1c51-09482dad86f2
# ╟─6a25edfc-d294-11ef-3411-6f74c376461e
# ╟─53de7edd-6c28-49a7-9f54-cf7b8ca42aeb
# ╟─288fbee6-0783-4447-b5d0-f5c2b29b39c7
# ╟─74fa1925-0d9f-47f6-a6bd-b822948a4fbc
# ╟─6a261278-d294-11ef-25a0-5572de58ad06
# ╟─6a26549a-d294-11ef-1f10-15c4d14ae41f
# ╟─6a262182-d294-11ef-23e9-ed45e1da9f46
# ╟─6a2672d6-d294-11ef-1886-3195c9c7cfa9
# ╟─6aa2399d-a949-40f9-8ee6-b0c2be1dc478
# ╟─6a2664c6-d294-11ef-0a49-5192e17fb9ea
# ╟─6a26a31e-d294-11ef-2c2f-b349d0859a27
# ╟─6a269568-d294-11ef-02e3-13402d296391
# ╟─6a26b7bc-d294-11ef-03e7-2715b6f8dcc7
# ╟─6a26f244-d294-11ef-0488-c1e4ec6e739d
# ╟─99db44c9-185c-4f39-ae5e-1a4cd751d980
# ╟─d22f58ac-9f68-41cb-8e61-cf74d3692c44
# ╟─6a2707e6-d294-11ef-02ad-31bf84662c70
# ╟─6a271a56-d294-11ef-0046-add807cc0b4f
# ╟─f6ee5570-9b92-42b6-baf3-3eed5352a060
# ╟─6a273ae0-d294-11ef-2c00-9b3eaed93f6d
# ╟─6a274948-d294-11ef-0563-1796b8883306
# ╟─6a275a52-d294-11ef-1323-9d83972f611a
# ╟─6a27684e-d294-11ef-040e-c302cdad714a
# ╟─6a2777d0-d294-11ef-1ac3-add102c097d6
# ╟─6a278784-d294-11ef-11ae-65bd398910d5
# ╟─c03229ef-3e0f-4612-909b-97f488a1e4c9
# ╟─6a27951c-d294-11ef-2e1a-b5a4ce84aceb
# ╟─6a27a28a-d294-11ef-1f33-41b444761429
# ╟─6a27b114-d294-11ef-099d-1d55968934a6
# ╟─6a27beca-d294-11ef-1895-d57b11b827c1
# ╟─cc8af69e-6d00-4327-aaa2-0b1023052b8a
# ╟─c454be00-05e7-42f6-a243-bf559ed6eff7
# ╟─6a9ad1c4-dfb2-4987-9ddc-da6131605083
# ╟─6a27efc6-d294-11ef-2dc2-3b2ef95e72f5
# ╟─6a280132-d294-11ef-10ac-f3890cb3f78b
# ╠═80edf8a4-e738-4bdb-bea3-0967926da645
# ╟─6a2814b0-d294-11ef-3a76-9b93c1fcd4d5
# ╟─6a282892-d294-11ef-2c12-4b1c7374617c
# ╟─6a286b04-d294-11ef-1b34-8b7a85c0048c
# ╟─6a2879e6-d294-11ef-37db-df7babe24d25
# ╟─6a2889ae-d294-11ef-2439-e1a541a5ccd7
# ╟─c050f468-7eec-403f-9304-552bd0d9b222
# ╟─6a2898ea-d294-11ef-39ec-31e4bac1e048
# ╟─6a28a704-d294-11ef-1bf2-efbdb0cb4cbc
# ╟─6a28b44c-d294-11ef-15da-81be8753d311
# ╟─6a28c9b4-d294-11ef-222b-97bf0912efe7
# ╟─6a28d81e-d294-11ef-2a9f-d32daa5556ae
# ╟─6a28e674-d294-11ef-391b-0d33fd609fb8
# ╟─6a28f466-d294-11ef-3af9-e34de9736c71
# ╠═df312e6a-503f-486f-b7ec-15404070960c
# ╟─3987d441-b9c8-4bb1-8b2d-0cc78d78819e
# ╟─51bed1cc-c960-46fe-bc09-2b684df3b0cc
# ╟─513414c7-0a54-4767-a583-7d779f8fbc55
# ╟─6a294790-d294-11ef-270b-5b2152431426
# ╟─b872cd69-d534-4b04-bb76-d85bb7ef0ea9
# ╟─1ba1939d-9986-4b97-9273-4f2434f1d385
# ╟─b426df32-5629-4773-b862-101cfbd82d42
# ╟─181ade96-8e1e-4186-9227-c1561352529d
# ╟─6a29d548-d294-11ef-1361-ad2230cad02b
# ╟─6a29e25e-d294-11ef-15ce-5bf3d8cdb64c
# ╟─6a29f1c2-d294-11ef-147f-877f99e5b57c
# ╟─6a2a000e-d294-11ef-17d6-bdcddeedc65d
# ╟─6a2a0f18-d294-11ef-02c2-ef117377ca66
# ╟─6a2a1daa-d294-11ef-2a67-9f2ac60a14c5
# ╟─6a2a2af2-d294-11ef-0072-bdc3c6f95bb3
# ╟─6a2a389e-d294-11ef-1b8c-b55de794b65c
# ╟─6a2a465e-d294-11ef-2aa0-43c954a6439e
# ╟─48fd2dff-796d-48bc-b5a8-bee270d119fd
# ╟─e3f9e571-2248-403c-8ab8-f6b99597f595
# ╟─90f691ad-046c-4595-99b0-19a1d6cb599e
# ╟─6a2a9faa-d294-11ef-1284-cfccb1da444e
# ╟─6a2aad42-d294-11ef-3129-3be5be8c82d6
# ╟─6a2abb16-d294-11ef-0243-d376e8a39bb0
# ╟─6a2acb7e-d294-11ef-185c-9d49ce79c31b
# ╟─8bfc4f37-4bf8-42a3-bd55-f046c8d2624a
# ╟─51829800-1781-49ae-8ee7-ac15c0bfcb88
# ╟─de7a1b82-f1c4-4eff-b372-ac76cf11c015
# ╟─d1d2bb84-7083-435a-9c19-4c02074143e3
# ╟─9c751f8e-f7ed-464f-b63c-41e318bbff2d
# ╟─e99e7650-bb72-4576-8f2a-c3994533b644
# ╟─7a624d2f-812a-47a0-a609-9fe299de94f5
# ╟─3a903a4d-1fb0-4566-8151-9c86dfc40ceb
# ╟─6a2af90a-d294-11ef-07bd-018326577791
# ╟─6a2b1106-d294-11ef-0d64-dbc26ba3eb44
# ╟─d484c41d-9834-4528-bf47-93ab4e35ebaa
# ╟─6a2b2d44-d294-11ef-33ba-15db357708b1
# ╟─6a2b3ba4-d294-11ef-3c28-176be260cb15
# ╠═c69c591f-1947-4b07-badb-3882fd097785
# ╟─188b5bea-6765-4dcf-9369-3b1fdbe94494
# ╟─ebcfcd1b-7fc8-42b7-a35e-4530f798cfdf
# ╟─84e7ff22-e232-4ab7-a206-ccdd943043dd
# ╟─6a2b9676-d294-11ef-241a-89ff7aa676f9
# ╟─9c5d7c89-f65c-4f52-9e49-14692bed2452
# ╟─6a2bb18a-d294-11ef-23bb-99082caf6e01
# ╟─6a2bd3ac-d294-11ef-0543-6fe202ca35b6
# ╟─6a2bf332-d294-11ef-1ff1-cdbfb7732cf1
# ╟─6a2c008e-d294-11ef-2f07-11cdfb2bddca
# ╟─6a2c11e6-d294-11ef-173b-23fc6dbfefca
# ╟─6a2c229e-d294-11ef-2f24-ebe43cbfbfa4
# ╟─6a2c3036-d294-11ef-23cb-c3b36c475e8f
# ╟─6a2c4058-d294-11ef-2312-d9c672d49701
# ╟─6a2c505c-d294-11ef-1c92-c1b0e9d50da5
# ╟─7c8b1add-085a-41ba-9d6c-b26d3eef22e4
# ╟─6a2c5e08-d294-11ef-213d-97bcfa16eb5a
# ╟─6a2c7230-d294-11ef-05a2-3ff2f65d10e0
# ╟─6a2c7f5a-d294-11ef-2e17-9108a39df280
# ╟─6a2c8f4a-d294-11ef-213c-dfa929a403bc
# ╟─6a2ca496-d294-11ef-0043-1f350b36773e
# ╟─f2969d91-4a5b-4665-9fa5-521db750302f
# ╟─7dd9a456-9dca-47c8-98c5-51f87f28e6a4
# ╟─b2820dfd-b3ca-477b-8cb7-c430e0fe18dd
# ╟─664d4183-edb6-4818-a44b-bf4c0a22a33c
# ╟─ecb036da-a0a2-4919-b1aa-bc33b6ba7e73
# ╟─de08c2a1-c5e3-4add-8b22-2c633247da48
# ╟─6a2cb25e-d294-11ef-1d88-1fc784b33df0
# ╟─1edae118-dcc7-4169-95cf-f36025f2c336
# ╟─275a9a69-3135-4cbd-8a35-b1abee4af83f
# ╟─6a2ccd16-d294-11ef-22ee-a5cff62ccd9c
# ╟─6a2cd9be-d294-11ef-33cf-4b23b92e1cbf
# ╠═10306ea6-6092-463c-9315-7a216c83606e
# ╠═ff20449d-1489-430a-aeee-a3a66bece706
# ╠═b47a2e71-48bb-4cc7-9a14-c0e654c5d2f8
# ╠═9a58bf5d-f072-4572-bb90-9b860133dce8
# ╟─7bf0fde7-b201-4646-9934-ec93e661cf22
# ╟─635e4f4c-5274-4bd4-a940-b2f3819426ec
# ╟─1f92c406-6792-4af6-9132-35efd8223bc5
# ╠═caba8eee-dfea-45bc-a8a7-1dd20a1fa994
# ╟─7a764a14-a5df-4f76-8836-f0a571fc3519
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
