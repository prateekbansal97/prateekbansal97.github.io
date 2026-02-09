---
title: 'Vector Spaces'
description: 'A primer on Linear Algebra for Machine Learning'
pubDate: 'Feb 08 2026'
heroImage: '/Linear_algebra_coverimage.png'
series: 'Linear Algebra'
order: 5
---

## Groups

Consider a set $\mathcal{G}$ together with a binary operation $\otimes$: $\mathcal{G} \times \mathcal{G} \to \mathcal{G}$  defined on $\mathcal{G}$. Then $G := \left(\mathcal{G}, \otimes\right)$ is a $\Large \textbf{group}$ if it satisfies the following four properties:

1. $\textbf{Closure}$ of $\mathcal{G}$ under $\otimes$: For all $a, b \in \mathcal{G}$, $a \otimes b \in \mathcal{G}$.
2. $\textbf{Associativity}$: For all $a, b, c \in \mathcal{G}$, $(a \otimes b) \otimes c = a \otimes (b \otimes c)$.
3. $\textbf{Identity element}$: There exists an element $e \in \mathcal{G}$ such that for all $a \in \mathcal{G}$, $a \otimes e = e \otimes a = a$.
4. $\textbf{Inverse element}$: For each $a \in \mathcal{G}$, there exists an element $a^{-1} \in \mathcal{G}$ such that $a \otimes a^{-1} = a^{-1} \otimes a = e$.

If additionally, for all $a, b \in \mathcal{G}$, $a \otimes b = b \otimes a$, then $G = (\mathcal{G}, \otimes)$ is called an $\Large \textbf{Abelian group}$.

1. $(\mathbb{Z}, +)$ is a group.
2. $(\mathbb{N}_0, +)$ is not a group, since it does not have an inverse.
3. $(\mathbb{Z}, \times)$ is not a group, since it does not have an inverse for any $z \in \mathbb{Z} \setminus \{-1, 1\}$.
4. $(\mathbb{R}, \times)$ is not a group, since it does not have an inverse for $r \in \mathbb{R} = 0$.
5. $(\mathbb{R} \setminus \{0\}, \times)$ is an abelian group.
6. $(\mathbb{R}^n, +), (\mathbb{Z}^n, +)$, $n \in \mathbb{N}$ are Abelian if $+$ is defined component-wise., i.e.
$$(x_1, x_2, \dots, x_n) + (y_1, y_2, \dots, y_n) = (x_1 + y_1, x_2 + y_2, \dots, x_n + y_n)$$
7. $(\mathbb{R}^{m \times n}, +)$ the set of all $m \times n$ matrices is an Abelian group.

Let us have a closer look at $(\mathbb{R}^{n \times n}, \cdot)$, the set of all $n \times n$ matrices with matrix multiplication as the binary operation.

1. Closure and associativity are satisfied from the definition of matrix multiplication itself.
2. The identity element is the identity matrix $I \in \mathbb{R}^{n \times n}$.
3. Inverse element: For a matrix $A \in \mathbb{R}^{n \times n}$, the inverse exists if and only if $\det(A) \neq 0$. In this case, the inverse is given by $A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$, where $\text{adj}(A)$ is the adjugate of $A$, and $(\mathbb{R}^{n \times n}, \cdot)$ is called the $\textbf{general linear group}$.

## Vector Space

Groups only need an inner operation. However, we can define an outer operation on a group, e.g. scalar multiplication of a vector by a scalar. 

A real-valued $\Large \textbf{vector space}$ $V = (\mathcal{V}, +, \cdot)$ is a set $\mathcal{V}$ with two operations:
1. $+: \mathcal{V} \times \mathcal{V} \to \mathcal{V}$ (inner operation)
2. $\cdot: \mathbb{R} \times \mathcal{V} \to \mathcal{V}$ (outer operation)

where
1. $(\mathcal{V}, +)$ is an Abelian group
2. Distributivity: For all $\lambda \in \mathbb{R}$ and $u, v \in \mathcal{V}$, $\lambda \cdot (u + v) = (\lambda \cdot u) + (\lambda \cdot v)$
3. Associativity (outer operation): For all $\lambda, \mu \in \mathbb{R}$ and $u \in \mathcal{V}$, $(\lambda \mu) \cdot u = \lambda \cdot (\mu \cdot u)$
4. Neutral element wrt outer operation: For all $u \in \mathcal{V}$, $1 \cdot u = u$

The elements $\boldsymbol{x} \in \mathbb{R}^n$ are called vectors, and the elements in $\mathbb{R}$ are called scalars.

## Outer product and Inner product of vectors

Let $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^n$. Then the $\Large \textbf{outer product}$ of $\boldsymbol{x}$ and $\boldsymbol{y}$ is given by
$$\boldsymbol{x} \cdot \boldsymbol{y} = \boldsymbol{x} \boldsymbol{y}^\top; \quad \boldsymbol{xy}^\top \in \mathbb{R}^{n \times n}$$


and the $\Large \textbf{inner product}$ of $\boldsymbol{x}$ and $\boldsymbol{y}$ is given by
$$\boldsymbol{x} \cdot \boldsymbol{y} = \boldsymbol{x}^\top \boldsymbol{y}; \quad \boldsymbol{x}^\top \boldsymbol{y} \in \mathbb{R}$$

## Vector Subspaces

Let $V = (\mathcal{V}, +, \cdot)$ be a vector space. A subset $\mathcal{U} \subseteq \mathcal{V}, \mathcal{U} \neq \emptyset$ is called a $\Large \textbf{vector subspace}$ if $(\mathcal{U}, +, \cdot)$ is also a vector space if $+$ and $\cdot$ are restricted to $\mathcal{U} \times \mathcal{U}$ and $\mathbb{R} \times \mathcal{U}$ respectively. 


