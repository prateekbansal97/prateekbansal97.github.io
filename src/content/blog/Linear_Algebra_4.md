---
title: 'Solving Systems of Linear Equations'
description: 'A primer on Linear Algebra for Machine Learning'
pubDate: 'Feb 08 2026'
heroImage: '/Linear_algebra_coverimage.png'
series: 'Linear Algebra'
order: 4
---

## Particular and General Solutions

Consider a system of equations:
<div style="position: relative; text-align: center;">

$$
\begin{bmatrix}
1 & 0 & 8 & -4\\
0 & 1 & 2 & 12
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2\\
x_3\\
x_4
\end{bmatrix}
=
\begin{bmatrix}
42\\
8
\end{bmatrix}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(1)</div>
</div>

One solution of this system of equations is $\begin{bmatrix} 42 & 8 & 0 & 0 \end{bmatrix}^\top$ called a $\Large \textbf{particular solution}$.

A $\Large \textbf{general solution}$ is given by:


<div style="position: relative; text-align: center;">

$$
\left\{\boldsymbol{x} \in \mathbb{R}^4: \boldsymbol{x}=\left[\begin{array}{c}
42 \\
8 \\
0 \\
0
\end{array}\right]+\lambda_1\left[\begin{array}{c}
8 \\
2 \\
-1 \\
0
\end{array}\right]+\lambda_2\left[\begin{array}{c}
-4 \\
12 \\
0 \\
-1
\end{array}\right], \lambda_1, \lambda_2 \in \mathbb{R}\right\}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(2)</div>
</div>

## Elementary Transformations

For $a \in \mathbb{R}$, we seek all solutions. of the following system of linear equations:

<div style="position: relative; text-align: center;">

$$
\begin{aligned}
-2x_1 &+ 4x_2 &- 2x_3 &- x_4 &+ 4x_5 &= -3 \\
4x_1 &- 8x_2 &+ 3x_3 &- 3x_4 &+ x_5 &= 2 \\
x_1 &- 2x_2 &+ x_3 &- x_4 &+ x_5 &= 0 \\
x_1 &- 2x_2 \quad &- 3x_4 &+ 4x_5 &= a \\
\end{aligned}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(3)</div>
</div>

We start by writing the augmented matrix:

<div style="position: relative; text-align: center;">

$$
\left[
\begin{array}{ccccc|c}
-2 & 4 & -2 & -1 & 4 & -3 \\
4 & -8 & 3 & -3 & 1 & 2 \\
1 & -2 & 1 & -1 & 1 & 0 \\
1 & -2 & 0 & -3 & 4 & a
\end{array}
\right]
\begin{array}{c}
\text{Swap with} \; R_3 \\
\\
\text{Swap with} \; R_1 \\
\\
\end{array}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(4)</div>
</div>

Which leads to:

<div style="position: relative; text-align: center;">

$$
\left[
\begin{array}{ccccc|c}
1 & -2 & 1 & -1 & 1 & 0 \\
4 & -8 & 3 & -3 & 1 & 2 \\
-2 & 4 & -2 & -1 & 4 & -3 \\
1 & -2 & 0 & -3 & 4 & a
\end{array}
\right]
\begin{array}{c}
\\
-4R_1 \\
+2R_1 \\
-R_1 
\end{array}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(5)</div>
</div>

When we now apply the indicated transformations, we get:

<div style="position: relative; text-align: center;">

$$
\begin{aligned}
&\left[
\begin{array}{ccccc|c}
1 & -2 & 1 & -1 & 1 & 0 \\
4 & -8 & 3 & -3 & 1 & 2 \\
-2 & 4 & -2 & -1 & 4 & -3 \\
1 & -2 & 0 & -3 & 4 & a
\end{array}
\right]
\begin{array}{c}
\\
-4R_1 \\
+2R_1 \\
-R_1 
\end{array}

\\


\leadsto
&\left[
\begin{array}{ccccc|c}
1 & -2 & 1 & -1 & 1 & 0 \\
0 & 0 & -1 & 1 & -3 & 2 \\
0 & 0 & 0 & -3 & 6 & -3 \\
0 & 0 & -1 & -2 & 3 & a
\end{array}
\right]

\begin{array}{c}
\\
\\
\\
-R_2 - R_3 
\end{array}
 \\

\leadsto
&\left[
\begin{array}{ccccc|c}
1 & -2 & 1 & -1 & 1 & 0 \\
0 & 0 & -1 & 1 & -3 & 2 \\
0 & 0 & 0 & -3 & 6 & -3 \\
0 & 0 & -1 & -2 & 3 & a
\end{array}
\right]
\begin{array}{c}
\\
. \left( -1 \right) \\
. \left( -\frac{1}{3} \right) \\
\\
\end{array}
\\
\leadsto
&\left[
\begin{array}{ccccc|c}
1 & -2 & 1 & -1 & 1 & 0 \\
0 & 0 & 1 & -1 & 3 & -2 \\
0 & 0 & 0 & 1 & -2 & 1 \\
0 & 0 & 0 & 0 & 0 & a+1 \\
\end{array}
\right]

\end{aligned}
$$
</div>

Only if $a = -1$, the system of equations has a solution. In this case, a $\textbf{particular solution}$ is given by 
<div style="position: relative; text-align: center;">

$\begin{bmatrix} x_1 & x_2 & x_3 & x_4 & x_5 \end{bmatrix}^\top = \begin{bmatrix} 2 & 0 & -1 & 1 & 0 \end{bmatrix}^\top$
</div>

And a $\textbf{general solution}$ is given by:

<div style="position: relative; text-align: center;">

$$
\left\{\boldsymbol{x} \in \mathbb{R}^5: \boldsymbol{x}=\left[\begin{array}{c}
2 \\
0 \\
-1 \\
1 \\
0
\end{array}\right]+\lambda_1\left[\begin{array}{c}
2 \\
1 \\
0 \\
0 \\
0
\end{array}\right]+\lambda_2\left[\begin{array}{c}
2 \\
0 \\
-1 \\
2 \\
1
\end{array}\right], \lambda_1, \lambda_2 \in \mathbb{R}\right\}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(2)</div>
</div>

## Row Echelon Form

A matrix is said to be in $\textbf{row echelon form}$ if:

1. All rows consisting entirely of zeros are at the bottom of the matrix.
2. The first non-zero entry in each non-zero row (called the leading entry or pivot) is 1.
3. Each leading entry is in a column to the right of the leading entry of the row above it.
4. Each column that contains a leading entry has zeros everywhere else.


## Calculating the Inverse of a Matrix using Gaussian Elimination
To determine the $\textbf{inverse}$ of

$$
\boldsymbol{A}=\left[\begin{array}{llll}
1 & 0 & 2 & 0 \\
1 & 1 & 0 & 0 \\
1 & 2 & 0 & 1 \\
1 & 1 & 1 & 1
\end{array}\right]
$$

we write down the augmented matrix

$$
\left[\begin{array}{llll|llll}
1 & 0 & 2 & 0 & 1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 & 0 & 1 & 0 & 0 \\
1 & 2 & 0 & 1 & 0 & 0 & 1 & 0 \\
1 & 1 & 1 & 1 & 0 & 0 & 0 & 1
\end{array}\right]
$$

and use Gaussian elimination to bring it into reduced row-echelon form

$$
\left[\begin{array}{cccc|cccc}
1 & 0 & 0 & 0 & -1 & 2 & -2 & 2 \\
0 & 1 & 0 & 0 & 1 & -1 & 2 & -2 \\
0 & 0 & 1 & 0 & 1 & -1 & 1 & -1 \\
0 & 0 & 0 & 1 & -1 & 0 & -1 & 2
\end{array}\right],
$$

such that the desired inverse is given as its right-hand side:

$$
\boldsymbol{A}^{-1}=\left[\begin{array}{cccc}
-1 & 2 & -2 & 2 \\
1 & -1 & 2 & -2 \\
1 & -1 & 1 & -1 \\
-1 & 0 & -1 & 2
\end{array}\right] .
$$


We can verify that the above matrix is indeed the inverse by performing the multiplication $\boldsymbol{A} \boldsymbol{A}^{-1}$ and observing that we recover $\boldsymbol{I}_4$.