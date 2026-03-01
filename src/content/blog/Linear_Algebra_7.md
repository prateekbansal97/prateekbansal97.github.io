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

The difference between a generating set and a basis is that a $\textbf{generating set can have redundant vectors, while a basis cannot}$.

Let $V = (\mathcal{V}, +, \cdot)$ be a vector space. If $B \subseteq \mathcal{V}$, then:

- $B$ is a basis
- $B$ is a minimal generating set
- $B$ is a maximal linearly independent set
- Every vector in $V$ has a unique representation as a linear combination of vectors in $B$, i.e. if

<div style="position: relative; text-align: center;">

$$
\boldsymbol{x} = \displaystyle \sum_{i=1}^{n} \lambda_i \boldsymbol{b}_i = \displaystyle \sum_{i=1}^{n} \psi_i \boldsymbol{b}_i \implies \lambda_i = \psi_i \quad \forall i \in \{1, \dots, n\}, \quad \forall \boldsymbol{b}_i \in B
$$
<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(1)</div>
</div>

A basis exists for every vector space, and the number of vectors in a basis is unique for that vector space.



$\Large \textbf{canonical basis}$ of $\mathbb{R}^n$
$$\mathcal{E} = \left[\boldsymbol{e}_1, \boldsymbol{e}_2, \dots, \boldsymbol{e}_n\right]$$

where $\boldsymbol{e}_i$ is the vector with a 1 in the $i$-th position and 0s elsewhere.

## Dimension

The $\Large \textbf{dimension}$ of a vector space $V$ is the number of vectors in any basis of $V$. It is denoted by $\dim(V)$.

## Rank

The $\Large \textbf{rank}$ of a matrix $A$ is the dimension of its column space (or row space). It is the maximum number of linearly independent columns (or rows) in the matrix.

Denoted by $\text{rank}(A)$ or $\text{rk}(A)$.


The difference between a dimension and a rank is that a $\textbf{dimension is the number of vectors in a basis, while a rank is the number of linearly independent vectors in a matrix}$.

For example:

<div style="position: relative; text-align: center;">

$$
\boldsymbol{A} = \begin{bmatrix}
1 & 1 & 1 \\
3 & -1 & 1 \\
1 & 5 & 3
\end{bmatrix}
\text{can be reduced to the row echelon form as}
\boldsymbol{A} = \begin{bmatrix}
1 & 1 & 1 \\
0 & -4 & -2 \\
0 & 0 & 0
\end{bmatrix}
$$
<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(2)</div>
</div>

Here, the rank of the matrix $A$ is 2, because there are 2 non-zero rows in the row echelon form of $A$.
But the dimension of the vector space spanned by the columns of $A$ is 3, since the number of columns in $A$ is 3.

1. $\text{rk}(A) = \text{rk}(A^\top)$, i.e., the column rank equals the row rank.
2. The span of the columns of $\boldsymbol{A}$ is the image of $\boldsymbol{A}$, i.e., $\text{span}(\boldsymbol{A}) = \text{Im}(\boldsymbol{A})$. Remember that the image is a matrix concept, while the span is a general concept for any set of vectors.
3. If $\boldsymbol{A} \in \mathbb{R}^{n \times n}$, then $\text{rk}(A) = n$ proves that A is non-singular, i.e., $A$ is invertible.
4. A system of linear equations $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}$ has a solution if and only if $\text{rk}(A) = \text{rk}([A|b])$, where $[A|b]$ is the augmented matrix.


For $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ the subspace of solutions for $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{0}$ possesses dimension $n - \text{rk}(\boldsymbol{A})$. This subspace is called the $\Large \textbf{kernel}$ or the $\Large \textbf{null space}$ of $\boldsymbol{A}$.


Let us define a matrix $\boldsymbol{A}$ with dimensions $m=3$ (rows) and $n=4$ (columns):

$$
\boldsymbol{A} = \begin{bmatrix} 
1 & 2 & 0 & 1 \\ 
2 & 4 & 1 & 3 \\ 
0 & 0 & 2 & 2 
\end{bmatrix}
$$

Here, the number of unknowns (columns) is $n=4$. According to the $\textbf{Rank-Nullity Theorem}$, the dimension of the null space should be:
$$
\text{dim}(\text{Null}(\boldsymbol{A})) = n - \text{rk}(\boldsymbol{A})
$$


We determine the rank by performing Gaussian elimination to find the Row Echelon Form.

$$
\text{REF}(\boldsymbol{A}) = \begin{bmatrix} 
1 & 2 & 0 & 1 \\ 
0 & 0 & 1 & 1 \\ 
0 & 0 & 0 & 0 
\end{bmatrix}
$$

$\textbf{Result:} $ There are $\textbf{2 non-zero rows}$ (pivots).

$$
\text{rk}(\boldsymbol{A}) = 2
$$


$$
\text{Expected Dimension} = n - \text{rk}(\boldsymbol{A}) = 4 - 2 = 2
$$

Let's solve $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{0}$ to see if we actually find 2 linearly independent vectors. The system of equations from the reduced matrix is:

$$
\begin{align*}
x_1 + 2x_2 + x_4 &= 0 \\
x_3 + x_4 &= 0
\end{align*}
$$

Here, the pivot variables are $x_1$ and $x_3$. The $\textbf{free variables}$ are $x_2$ and $x_4$. We express the pivots in terms of the free variables:

$$
\begin{align*}
x_3 &= -x_4 \\
x_1 &= -2x_2 - x_4
\end{align*}
$$

Now, write the full solution vector $\boldsymbol{x}$:

$$
\boldsymbol{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix} 
= \begin{bmatrix} -2x_2 - x_4 \\ x_2 \\ -x_4 \\ x_4 \end{bmatrix}
$$

Separate the free variables $x_2$ and $x_4$ to reveal the basis vectors:

$$
\boldsymbol{x} = x_2 \begin{bmatrix} -2 \\ 1 \\ 0 \\ 0 \end{bmatrix} 
+ x_4 \begin{bmatrix} -1 \\ 0 \\ -1 \\ 1 \end{bmatrix}
$$

The general solution is a linear combination of exactly $\textbf{2}$ basis vectors:

$$
\text{Basis} = \left\{ \begin{bmatrix} -2 \\ 1 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} -1 \\ 0 \\ -1 \\ 1 \end{bmatrix} \right\}
$$

Therefore, the dimension of the null space is indeed $\textbf{2}$, confirming the theorem:

$$
\underbrace{2}_{\text{Nullity}} = \underbrace{4}_{n} - \underbrace{2}_{\text{Rank}}
$$

