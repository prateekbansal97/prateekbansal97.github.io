---
title: 'Systems of Linear Equations'
description: 'A primer on Linear Algebra for Machine Learning'
pubDate: 'Feb 08 2026'
heroImage: '/Linear_algebra_coverimage.png'
series: 'Linear Algebra'
order: 2
---

Linear Equations are very useful in optimization problems, for example:

A company produces products $N_1, \ldots,  N_n$ for which resources $R_1, \ldots, R_m$ are required. To produce a unit of product $N_j$, $a_{ij}$ units of resource $R_i$ are required, where $i \in \{1, \ldots, m\}$ and $j \in \{1, \ldots, n\}$.

The objective to find an optimal production plan, i.e., a plan of how many units $x_j$ of product $N_j$ should be produced if a total of $b_i$ units of resource $R_i$ are available and ideally no resource should be left unused.

If we produce $x_1, \ldots, x_n$ units of corresponding products, we need a total of

<div style="position: relative; text-align: center;">

$$
a_{i1}x_1 + \ldots + a_{in}x_n
$$

<!-- <div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(1)</div> -->
</div>

many units of resource $R_i$. An optimal production plan $(x_1, \ldots, x_n)$ satisfies the following system of equations:

<div style="position: relative; text-align: center;">

$$
\begin{aligned}
a_{i1}x_1 + \ldots &+ a_{in}x_n = b_i \\
\vdots \\
a_{m1}x_1 + \ldots &+ a_{mn}x_n = b_m
\end{aligned}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(1)</div>
</div>

where $a_{ij} \in \mathbb{R}$ and $b_i \in \mathbb{R}$.

Equation $(1)$ is the general form of a system of linear equations, and $x_1, \ldots, x_n$ are the unknowns of the system. Every $n-$tuple $(x_1, \ldots, x_n)$ that satisfies the system of equations is called a $\textbf{solution}$ to the system.

For a system of linear equations, there are three possible cases:

1. The system has a $\textbf{unique solution}$.
2. The system has $\textbf{no solution}$.
3. The system has $\textbf{infinitely many solutions}$.

For example, consider the following system of linear equations:

<div style="position: relative; text-align: center;">

$$
\begin{aligned}
x_1 + x_2 &+ x_3 &= 3 \\
x_1 - x_2 &+ 2x_3 &= 2 \\
2x_1 \quad \quad &+ 3x_3 &= 1
\end{aligned}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(2)</div>
</div>

has $\textbf{no solution}$. Adding the first two equations, we get $2x_1 + 3x_3 = 5$, which is a contradiction to the third equation $2x_1 + 3x_3 = 1$.

Next, consider the following system of linear equations:

<div style="position: relative; text-align: center;">

$$
\begin{aligned}
x_1 + x_2 &+ x_3 &= 3 \\
x_1 - x_2 &+ 2x_3 &= 2 \\
\quad \quad x_2 &+ x_3 &= 2
\end{aligned}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(3)</div>
</div>

Has a $\textbf{unique solution}$ $(x_1, x_2, x_3) = (1, 1, 1)$.

Finally, consider the following system of linear equations:

<div style="position: relative; text-align: center;">

$$
\begin{aligned}
x_1 + x_2 &+ x_3 &= 3 \\
x_1 - x_2 &+ 2x_3 &= 2 \\
2x_1 \quad \quad &+ 3x_3 &= 5
\end{aligned}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(4)</div>
</div>


Since the third equation is a linear combination of the first two equations, the system has $\textbf{infinitely many solutions}$.

In general, for a system of equation in $n$ variables, it represents a line in $\mathbb{R}^n$. For example, consider the following system of linear equations in 2 variables:

<div style="position: relative; text-align: center;">

$$
\begin{aligned}
x_1 &+ x_2 &= 4 \\
-x_1 &+ 2x_2 &= 2
\end{aligned}
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(5)</div>
</div>

represents two lines in $\mathbb{R}^2$, which is the $xy$-plane: 

<img src="/lin_alg_blog_fig3.png" alt="Linear System" style="width: 100%; max-width: 600px; margin: 20px auto; display: block;">
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Figure 3  : Two lines in $\mathbb{R}^2$. The intersection point is the solution to the system of equations.</div>

For a systematic approach to solving a general system of linear equations, we use the following notation: 

<div style="position: relative; text-align: center;">

$$
\left[\begin{array}{c}
a_{11} \\
\vdots \\
a_{m 1}
\end{array}\right] x_1+\left[\begin{array}{c}
a_{12} \\
\vdots \\
a_{m 2}
\end{array}\right] x_2+\cdots+\left[\begin{array}{c}
a_{1 n} \\
\vdots \\
a_{m n}
\end{array}\right] x_n=\left[\begin{array}{c}
b_1 \\
\vdots \\
b_m
\end{array}\right]
$$

$$
\Longleftrightarrow\left[\begin{array}{ccc}
a_{11} & \cdots & a_{1 n} \\
\vdots & & \vdots \\
a_{m 1} & \cdots & a_{m n}
\end{array}\right]\left[\begin{array}{c}
x_1 \\
\vdots \\
x_n
\end{array}\right]=\left[\begin{array}{c}
b_1 \\
\vdots \\
b_m
\end{array}\right] .
$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(6)</div>
</div>

This motivates the definition of a matrix, part of the next section.





