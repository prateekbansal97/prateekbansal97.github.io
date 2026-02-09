---
title: 'Matrices'
description: 'A primer on Linear Algebra for Machine Learning'
pubDate: 'Feb 08 2026'
heroImage: '/Linear_algebra_coverimage.png'
series: 'Linear Algebra'
order: 3
---

With $m, n \in \mathbb{N}$, a $\Large \textbf{matrix}$ $A$ is a rectangular array of numbers $a_{ij}$, where $i \in \{1, \ldots, m\}$ and $j \in \{1, \ldots, n\}$.

<div style="position: relative; text-align: center;">

$$
\mathbf{A} = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(1)</div>
</div>

where $a_{ij} \in \mathbb{R}$.

#### Matrix-Stacking into Column Vectors

Matrices can be $\Large \textbf{stacked}$ into column vectors as follows:

<img src="/lin_alg_blog_fig4.png" alt="Linear System" style="width: 30%; max-width: 600px; margin: 20px auto; display: block;">
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Stacking of matrices into column vectors.</div>

## Matrix Addition and Multiplication

Given two matrices $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{m \times n}$, $\Large \textbf{matrix addition}$ is defined with adding them element-wise, i.e., $\mathbf{A} + \mathbf{B} = \mathbf{C}$, where $\mathbf{C} \in \mathbb{R}^{m \times n}$ is a matrix of the same dimensions as $\mathbf{A}$ and $\mathbf{B}$, and $c_{ij} = a_{ij} + b_{ij}$.

<div style="position: relative; text-align: center;">

$$
\mathbf{C} = \mathbf{A} + \mathbf{B} = \begin{bmatrix}
a_{11} + b_{11} & a_{12} + b_{12} & \cdots & a_{1n} + b_{1n} \\
a_{21} + b_{21} & a_{22} + b_{22} & \cdots & a_{2n} + b_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} + b_{m1} & a_{m2} + b_{m2} & \cdots & a_{mn} + b_{mn}
\end{bmatrix}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(2)</div>
</div>

For $\Large \textbf{matrix multiplication}$, we need to ensure that the number of columns in the first matrix is equal to the number of rows in the second matrix.

If $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n \times p}$, then $\mathbf{C} = \mathbf{AB} \in \mathbb{R}^{m \times p}$ is defined as:

<div style="position: relative; text-align: center;">

$$
\underbrace{\mathbf{A}}_{m \times n} \underbrace{\mathbf{B}}_{n \times p} = \underbrace{\mathbf{C}}_{m \times p}
$$


$$
c_{ij} = \displaystyle \sum_{k=1}^{n} a_{ik}b_{kj}, \quad \text{for } i \in \{1, \ldots, m\}, \quad j \in \{1, \ldots, p\}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(3)</div>
</div>

In such cases the product $\mathbf{BA}$ is not defined, since the number of columns in $\mathbf{B}$ is $p$ and the number of rows in $\mathbf{A}$ is $m$, and $p \neq m$.

One of the ways to matrix multiply is to use `np.einsum:`

```python
C = np.einsum('il, lj -> ij', A, B)
```

Here, $il$ and $lj$ represent the dummy representations for the number of rows, columns for inputs $\mathbf{A, B}$ and $ij$ represents the dummy number of rows, columns for the output $\mathbf{C}$.

Matrix Multiplication is:

1. $\textbf{Associative}: \forall \mathbf{A} \in \mathbb{R}^{m \times n}, \mathbf{B} \in \mathbb{R}^{n \times p}, \mathbf{C} \in \mathbb{R}^{p \times q}: \mathbf{(AB)C = A(BC)}$
2. $\textbf{Distributive}: \forall \mathbf{A} \in \mathbb{R}^{m \times n}, \mathbf{B} \in \mathbb{R}^{n \times p}, \mathbf{C} \in \mathbb{R}^{n \times p}: \mathbf{A(B + C) = AB + AC}$
3. $\textbf{Not Commutative}: \forall \mathbf{A} \in \mathbb{R}^{m \times n}, \mathbf{B} \in \mathbb{R}^{n \times p}: \mathbf{AB \neq BA}$

Another kind of matrix product that machine learning literature often uses is the $\textbf{Hadamard product}$, represented by the symbol $\odot$:

<div style="position: relative; text-align: center;">

$$
c_{ij} = a_{ij}b_{ij}, \quad \text{represented by} \quad C = A \odot B
$$ 

$$
\begin{aligned}
C = A \odot B = \begin{bmatrix}
a_{11} b_{11} & a_{12} b_{12} & \cdots & a_{1n} b_{1n} \\
a_{21} b_{21} & a_{22} b_{22} & \cdots & a_{2n} b_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} b_{m1} & a_{m2} b_{m2} & \cdots & a_{mn} b_{mn}
\end{bmatrix}
\end{aligned}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(4)</div>
</div>

One of the ways to perform the Hadamard Product is to use `np.multiply:`

```python
C = np.multiply(A, B)
```
## Identity Matrix

An $\Large \textbf{identity matrix}$ is a square matrix that has ones on the main diagonal and zeros everywhere else.

<div style="position: relative; text-align: center;">

$$
\mathbf{I}_n = \begin{bmatrix}
1 & 0 & 0 & \cdots & 0 \\
0 & 1 & 0 & \cdots & 0 \\
0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 1
\end{bmatrix} \in \mathbb{R}^{n \times n}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(5)</div>
</div>

Multiplication with an identity matrix does not change the matrix:

<div style="position: relative; text-align: center;">

$$
\forall \mathbf{A} \in \mathbb{R}^{m \times n}: \mathbf{AI_n} = \mathbf{I_m A} = \mathbf{A}
$$
<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(6)</div>
</div>

Note that $\mathbf{I_m} \neq \mathbf{I_n}$ if $m \neq n$.

## Inverse and Transpose of a Matrix

Consider a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$. Let matrix $\mathbf{B} \in \mathbb{R}^{n \times n}$ be such that $\mathbf{AB} = \mathbf{BA} = \mathbf{I}_n$. Then $\mathbf{B}$ is called the $\Large \textbf{inverse}$ of $\mathbf{A}$ and is denoted by $\mathbf{A}^{-1}$.

1. Not every matrix has an inverse. If the inverse doesnt exist, the matrix is called $\textbf{singular / non-invertible}$.
2. If a matrix has an inverse, it is $\textbf{unique}$.
3. If a matrix has an inverse, it is $\textbf{invertible / regular / non-singular}$.

For a $2 \times 2$ matrix $\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the inverse is given by:

<div style="position: relative; text-align: center;">

$$
\mathbf{A}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(7)</div>
</div>

For $\mathbf{A} \in \mathbb{R}^{n \times n}$, the matrix $\mathbf{B} \in \mathbb{R}^{n \times n}$ is called the $\Large \textbf{transpose}$ of $\mathbf{A}$ if $b_{ij} = a_{ji}$ for all $i \in \{1, \ldots, n\}, j \in \{1, \ldots, n\}$. The notation for the transpose of $\mathbf{A}$ is $\mathbf{B} := \mathbf{A}^\top$.

Some properties of transpose and inverse:

1. $\mathbf{A}^{-1} \mathbf{A} = \mathbf{A} \mathbf{A}^{-1} = \mathbf{I}_n$
2. $(\mathbf{A}^\top)^\top = \mathbf{A}$
3. $(\mathbf{A}^{-1})^\top = (\mathbf{A}^\top)^{-1}$
4. $(\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top$
5. $(\mathbf{AB})^{-1} = \mathbf{B}^{-1} \mathbf{A}^{-1}$
6. $(\mathbf{A} + \mathbf{B})^{-1} \neq \mathbf{A}^{-1} + \mathbf{B}^{-1}$
7. $(\mathbf{A} + \mathbf{B})^\top = \mathbf{A}^\top + \mathbf{B}^\top$
8. $(\mathbf{A} \odot \mathbf{B})^\top = \mathbf{A}^\top \odot \mathbf{B}^\top$
9. $(\mathbf{A} \odot \mathbf{B})^{-1} \neq \mathbf{A}^{-1} \odot \mathbf{B}^{-1}$


## Symmetric Matrix

For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, if $\mathbf{A} = \mathbf{A}^\top$, then $\mathbf{A}$ is called a $\Large \textbf{symmetric matrix}$.

Some properties of symmetric matrices:

1. Only $\textbf{square}$ matrices can be symmetric.
2. The $\textbf{sum}$ of two symmetric matrices $\textbf{is}$ a symmetric matrix.
3. The $\textbf{product}$ of two symmetric matrices $\textbf{is}$ a symmetric matrix if and only if the matrices $\textbf{commute}$, that is, $\mathbf{AB} = \mathbf{BA}$.
4. The $\textbf{transpose}$ of a symmetric matrix $\textbf{is}$ a symmetric matrix.
5. The $\textbf{inverse}$ of a symmetric matrix $\textbf{is}$ a symmetric matrix.

## Multiplication by a Scalar

For a scalar $\lambda \in \mathbb{R}$ and a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, the $\Large \textbf{scalar multiplication}$ $\lambda \mathbf{A}$ is given by:

<div style="position: relative; text-align: center;">

$$
\lambda \mathbf{A} = \begin{bmatrix}
\lambda a_{11} & \lambda a_{12} & \cdots & \lambda a_{1n} \\
\lambda a_{21} & \lambda a_{22} & \cdots & \lambda a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
\lambda a_{m1} & \lambda a_{m2} & \cdots & \lambda a_{mn}
\end{bmatrix}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(8)</div>
</div>  

Scalar multiplication is:

1. Associative: $\forall \lambda, \mu \in \mathbb{R}, \forall \mathbf{A} \in \mathbb{R}^{m \times n}: (\lambda \mu) \mathbf{A} = \lambda (\mu \mathbf{A})$
2. Distributive: $\forall \lambda, \mu \in \mathbb{R}, \forall \mathbf{A} \in \mathbb{R}^{m \times n}: (\lambda + \mu) \mathbf{A} = \lambda \mathbf{A} + \mu \mathbf{A}$


## Compact representation of a system of linear equations

A system of linear equations can be represented in a compact form using matrices and vectors. Consider the following system of linear equations:

<div style="position: relative; text-align: center;">

$$
\begin{aligned}
2x_1 + 3x_2 + 4x_3 = 1 \\
3x_1 + 4x_2 + 5x_3 = 2 \\
4x_1 + 5x_2 + 6x_3 = 3
\end{aligned}
$$

can be written in the compact form:

<div style="position: relative; text-align: center;">

$$
\begin{aligned}
\mathbf{A} \mathbf{x} = \mathbf{b}
\end{aligned}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(9)</div>
</div>  

where

<div style="position: relative; text-align: center;">

$$
\begin{aligned}
\mathbf{A} = \begin{bmatrix}
2 & 3 & 4 \\
3 & 4 & 5 \\
4 & 5 & 6
\end{bmatrix}, \quad
\mathbf{x} = \begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}, \quad
\mathbf{b} = \begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
\end{aligned}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(10)</div>
</div>  









