---
title: 'Linear Algebra for Machine Learning'
description: 'A primer on Linear Algebra for Machine Learning'
pubDate: 'Feb 08 2026'
heroImage: '/Linear_algebra_coverimage.png'
series: 'Linear Algebra'
order: 1
---

In Linear Algebra, a common approach is to consider a set of objects and a set of rules to manipulate them. Particularly, linear algebra deals with vectors, vector spaces, and linear transformations between them. 

We know from school what a vector is. From an abstract point of view, any object that satisfies the following properties is a vector:

1. Adding two vectors results in a vector in the same set.
2. Multiplying a vector by a scalar results in a vector in the same set.

If this is the case, then even polynomials of degree $n$ form a vector space, since we can add two polynomials of degree $n$ and the resulting polynomial will also be of degree $n$. Similarly, we can multiply a polynomial of degree $n$ by a scalar $\lambda \in \mathbf{\mathbb{R}}$ and the resulting polynomial will also be of degree $n$.

<div style="position: relative; text-align: center;">

$$
\begin{aligned}
\large p_1 &= x^3 + 2x^2 + 3x + 4 \\
\large p_2 &= 2x^3 + 3x^2 + 4x + 5 \\
\large p_1 + p_2 &= 3x^3 + 5x^2 + 7x + 9 \\
\large 2p_1 &= 2x^3 + 4x^2 + 6x + 8
\end{aligned}
$$

<!-- <div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(1)</div> -->
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Addition and scalar multiplication of polynomials gives polynomials of the same degree.</div>


This is in addition to the geometric vectors that we are introduced to in high school, which also satisfy the above two properties. 

Geometric vectors are directed segments with atleast a dimension of 2. Two geometric vectors $\vec{x}$ and $\vec{y}$ can be added, such that $\vec{x} + \vec{y} = \vec{z}$ is another geometric vector. 

Furthemore, multiplication by a scalar $\lambda \vec{x}, \lambda \in \mathbf{\mathbb{R}}$, is also a geometric vector. 

<div style="position: relative; text-align: center;">

$$
\begin{aligned}
\large \vec{x} &= \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad \vec{y} = \begin{bmatrix} 3 \\ 4 \end{bmatrix} \\
\large \vec{x} + \vec{y} &= \begin{bmatrix} 4 \\ 6 \end{bmatrix} \\
\large 2\vec{x} &= \begin{bmatrix} 2 \\ 4 \end{bmatrix}
\end{aligned}
$$

<!-- <div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(1)</div> -->
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Addition and scalar multiplication of geometric vectors gives geometric vectors.</div>

<!-- Can you add the image for the geometric vectors? -->

The figure below shows the addition and scalar multiplication of (a) geometric vectors and (b) polynomials.

<img src="/lin_alg_blog_fig1.png" alt="Geometric Vectors" style="width: 100%; max-width: 600px; margin: 20px auto; display: block;">
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Figure 1: Addition and scalar multiplication of (a) geometric vectors and (b) polynomials.</div>

Most linear algebra on machine learning focuses on vectors in $\mathbb{R}^n$. 

One of the most important concepts in linear algebra is the concept of a vector space. It comes from the idea of "closure", which essentially asks the question, "What is the set of vectors that can result from the proposed operations on a given set of objects?" In the case of vectors, the question becomes, "What is the set of vectors that can result from adding vectors or scaling them?"  

<img src="/lin_alg_blog_fig2.png" alt="Vector Space" style="width: 100%; max-width: 600px; margin: 20px auto; display: block;">
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Figure 2: Vector Space</div>

Next, we cover Systems of Linear Equations.



