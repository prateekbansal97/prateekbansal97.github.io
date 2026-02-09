---
title: 'Basis and Rank'
description: 'A primer on Linear Algebra for Machine Learning'
pubDate: 'Feb 08 2026'
heroImage: '/Linear_algebra_coverimage.png'
series: 'Linear Algebra'
order: 7
---

## Generating Set and Basis

Consider a vector space $V = (\mathcal{V}, +, \cdot)$. A set of vectors $\mathcal{A} = \{\boldsymbol{v}_1, \boldsymbol{v}_2, \dots, \boldsymbol{v}_n\} \subseteq \mathcal{V}$ is called a $\Large \textbf{generating set}$ of $V$ if every vector in $V$ can be written as a linear combination of these vectors.

The set of all linear combinations of vectors in $\mathcal{A}$ is called the $\Large \textbf{span}$ of $\mathcal{A}$, denoted by $\text{span}\left[\mathcal{A}\right]$ or $\text{span}\left[x_1, x_2, \dots, x_n\right]$.

Vectors $\boldsymbol{v}_1, \boldsymbol{v}_2, \dots, \boldsymbol{v}_n \in \mathcal{V}$ are called a $\Large \textbf{basis}$ of $V$ if they are linearly independent and span $V$.

$\Large \textbf{canonical basis}$ of $\mathbb{R}^n$
$$\mathcal{E} = \left[\boldsymbol{e}_1, \boldsymbol{e}_2, \dots, \boldsymbol{e}_n\right]$$

where $\boldsymbol{e}_i$ is the vector with a 1 in the $i$-th position and 0s elsewhere.

## Dimension

The $\Large \textbf{dimension}$ of a vector space $V$ is the number of vectors in any basis of $V$. It is denoted by $\dim(V)$.

## Rank

The $\Large \textbf{rank}$ of a matrix $A$ is the dimension of its column space (or row space). It is the maximum number of linearly independent columns (or rows) in the matrix.