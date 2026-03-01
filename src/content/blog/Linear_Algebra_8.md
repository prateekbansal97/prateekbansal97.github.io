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

