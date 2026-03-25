---
title: 'Linear Mappings'
description: 'A primer on Linear Algebra for Machine Learning'
pubDate: 'Feb 08 2026'
heroImage: '/Linear_algebra_coverimage.png'
series: 'Linear Algebra'
order: 8
---

A $\Large \textbf{mapping}$ joins two vector spaces: If the input to the mapping is from a vector space $\boldsymbol{V}$ then the output of the mapping is from a vector space $\boldsymbol{W}$. Such mappings are notated as $\boldsymbol{\Phi}: \boldsymbol{V} \rightarrow \boldsymbol{W}$

A mapping is a $\Large \textbf{linear mapping}$ if it satisfies the following properties:

1. $\boldsymbol{\Phi}(\boldsymbol{u} + \boldsymbol{v}) = \boldsymbol{\Phi}(\boldsymbol{u}) + \boldsymbol{\Phi}(\boldsymbol{v})$ for all $\boldsymbol{u}, \boldsymbol{v} \in \boldsymbol{V}$
2. $\boldsymbol{\Phi}(\alpha \boldsymbol{u}) = \alpha \boldsymbol{\Phi}(\boldsymbol{u})$ for all $\alpha \in \mathbb{R}$ and $\boldsymbol{u} \in \boldsymbol{V}$

Another way to represent a linear mapping is:

$$
\forall \boldsymbol{x},  \boldsymbol{y} \in \boldsymbol{V}, \quad \forall \lambda, \psi \in \mathbb{R}, \quad \boldsymbol{\Phi}(\lambda \boldsymbol{x} + \psi \boldsymbol{y}) = \lambda \boldsymbol{\Phi}(\boldsymbol{x}) + \psi \boldsymbol{\Phi}(\boldsymbol{y})
$$

A mapping is $\Large \textbf{injective}$ if $\forall \boldsymbol{u}, \boldsymbol{v} \in \boldsymbol{V}, \quad \boldsymbol{\Phi}(\boldsymbol{u}) = \boldsymbol{\Phi}(\boldsymbol{v}) \implies \boldsymbol{u} = \boldsymbol{v}$

A mapping is $\Large \textbf{surjective}$ if $\forall \boldsymbol{w} \in \boldsymbol{W}, \quad \exists \boldsymbol{u} \in \boldsymbol{V}, \quad \boldsymbol{\Phi}(\boldsymbol{u}) = \boldsymbol{w}$

A mapping is $\Large \textbf{bijective}$ if it is both injective and surjective.


<!-- <img src="/lin_alg_blog_fig5.png" alt="Linear System" style="width: 200%; max-width: 900px; margin: 20px auto; display: block;">
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Different types of linear mappings.</div> -->

<video controls loop autoplay muted playsinline style="width: 100%; max-width: 900px; margin: 20px auto; display: block;">
  <source src="/MappingExamples.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

If $ \Phi$ is surjective, then every element in $W$ can be “reached” from $V$
using $ \Phi$.


A bijective $ \Phi$ can be “undone”, i.e., there exists a unique mapping $ \Psi$ :
$W$ → $V$ so that $ \Psi \circ \Phi(x)$ = $x$. This mapping $ \Psi$ is then called the inverse
of $ \Phi$ and normally denoted by $ \Phi^{-1}$.

$\Large \textbf{Isomorphism}$: $ \Phi : V \rightarrow W$ linear and bijective is called an isomorphism.

<video controls loop autoplay muted playsinline style="width: 100%; max-width: 900px; margin: 20px auto; display: block;">
  <source src="/IsomorphismConcept.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

$\Large \textbf{Endomorphism}$: $ \Phi : V \rightarrow V$ linear is called an endomorphism.

<video controls loop autoplay muted playsinline style="width: 100%; max-width: 900px; margin: 20px auto; display: block;">
  <source src="/EndomorphismConcept.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

$\Large \textbf{Automorphism}$: $ \Phi : V \rightarrow V$ linear and bijective

<video controls loop autoplay muted playsinline style="width: 100%; max-width: 900px; margin: 20px auto; display: block;">
  <source src="/AutomorphismConcept.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

$ \Large \textbf{Homomorphism}$: An example is representing the 2D vector space $\mathbb{R}^2$ by complex numbers $\mathbb{C}$

<video controls loop autoplay muted playsinline style="width: 100%; max-width: 900px; margin: 20px auto; display: block;">
  <source src="/HomomorphismExample.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

If mapping $\Phi : V \rightarrow W$ is linear and mapping $\Psi : W \rightarrow X$ is also linear, then $\Phi \circ \Psi : V \rightarrow X$ is also linear.


Consider a vector space $V$ and an ordered
basis $B = (b_1, \ldots, b_n)$ of $V$. For any $x \in V$ we obtain a unique representation (linear combination)
<div style="position: relative; text-align: center;">

$$x = \alpha_1 b_1 + \alpha_2 b_2 + \ldots + \alpha_n b_n$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(1)</div>
</div>

of $x$ with respect to $B$. Then $\alpha_1, \ldots, \alpha_n$ are the $\Large \textbf{coordinates}$ of $x$ with respect to $B$, and the vector
$$
\alpha =
\begin{bmatrix}
\alpha_1 \\
\vdots \\
\alpha_n
\end{bmatrix}
$$
is the $\Large \textbf{coordinate vector/coordinate representation}$ of $x$ with respect to the ordered basis $B, \quad \alpha \in \mathbb{R}^n$


Consider vector spaces $V, W$ with corresponding (ordered) bases $B = (b_1, \ldots, b_n)$ and $C = (c_1, \ldots, c_m)$.
Moreover, we consider a linear mapping $\Phi : V \rightarrow W$. For $j \in \{1, \ldots, n\}$

<div style="position: relative; text-align: center;">

$$\Phi(b_j) = \alpha_{1j} c_1 + \alpha_{2j} c_2 + \ldots + \alpha_{mj} = \displaystyle \sum_{i=1}^m \alpha_{ij} c_i$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(2)</div>
</div>


then the matrix $\mathbf{A_\phi}$ containing the coefficients 

<div style="position: relative; text-align: center;">

$\mathbf{A_\phi} (i, j) = \alpha_{ij}$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(3)</div>
</div>

is called the $\Large \textbf{transformation matrix}$ of $\Phi$ with respect to the ordered bases $B$ and $C$.

<video controls loop autoplay muted playsinline style="width: 100%; max-width: 900px; margin: 20px auto; display: block;">
  <source src="/LinearTransformations.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


Consider two ordered bases $B = (b_1, . . . , b_n),$ and $\tilde{B} = (\tilde{b}_1, . . . , \tilde{b}_n)$ of $V$ 

and two ordered bases
$C = (c_1, . . . , c_m),$ and $\tilde{C} = (\tilde{c}_1, . . . , \tilde{c}_m)$
of $W$ . 

Moreover, $A_\Phi \in \mathbb{R}^{m \times n}$ is the transformation matrix of the linear
mapping $\Phi : V \rightarrow W$ with respect to the bases $B$ and $C$

and $\tilde{A}_\Phi \in \mathbb{R}^{m \times n}$
is the corresponding transformation mapping with respect to $\tilde{B}$ and $\tilde{C}$

The corresponding transformation matrix $\tilde{A}_\Phi$ with respect to the bases $\tilde{B}$ and $\tilde{C}$
is given as
<div style="position: relative; text-align: center;">

$$
\tilde{A}_\Phi = T^{-1}A_\Phi S
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(4)</div>
</div>

where $S \in \mathbb{R}^{n \times n}$ and $T \in \mathbb{R}^{m \times m}$ are the transformation matrices of the bases $B \rightarrow \tilde{B}$ and $C \rightarrow \tilde{C}$ respectively.


Two matrices $A, \tilde{A} \in \mathbb{R}^{m \times n}$ are called $\Large \textbf{equivalent}$ if there exists an invertible matrix $S \in \mathbb{R}^{n \times n}$ and $T \in \mathbb{R}^{m \times m}$ such that $\tilde{A} = T^{-1}AS$.

Two matrices $A, \tilde{A} \in \mathbb{R}^{n \times n}$ are called $\Large \textbf{similar}$ if there exists an invertible matrix $S \in \mathbb{R}^{n \times n}$ such that $\tilde{A} = S^{-1}AS$.


For $\Phi : V \rightarrow W$ , we define the $\Large \textbf{kernel/null space}$ as

<div style="position: relative; text-align: center;">

$$
\text{ker}(\Phi) := \Phi^{-1}(0_W) = \{v \in V : \Phi(v) = 0_W\}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(5)</div>
</div>

Which means, that the kernel of $\Phi$ is the set of all vectors in $V$ that are mapped to the zero vector in $W$.

$\Large \textbf{Image/Range}$ is defined as   
<div style="position: relative; text-align: center;">

$$
\text{Im}(\Phi) := \Phi(V) = \{w \in W | \exists v \in V : \Phi(v) = w\}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(6)</div>
</div>

Which means, that the image/range of $\Phi$ is the set of all vectors in $W$ that are mapped to by some vector in $V$.

W is also called the $\Large \textbf{codomain}$ of $\Phi$.