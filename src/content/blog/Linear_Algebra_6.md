---
title: 'Linear Independence'
description: 'A primer on Linear Algebra for Machine Learning'
pubDate: 'Feb 08 2026'
heroImage: '/Linear_algebra_coverimage.png'
series: 'Linear Algebra'
order: 6
---

## Linear Combination

Consider a vector space $V = (\mathcal{V}, +, \cdot)$. A vector $\boldsymbol{x} \in \mathcal{V}$ is called a $\Large \textbf{linear combination}$ of vectors $\boldsymbol{v}_1, \boldsymbol{v}_2, \dots, \boldsymbol{v}_n \in \mathcal{V}$ if there exist scalars $\lambda_1, \lambda_2, \dots, \lambda_n \in \mathbb{R}$ such that

$$\boldsymbol{x} = \lambda_1 \boldsymbol{v}_1 + \lambda_2 \boldsymbol{v}_2 + \dots + \lambda_n \boldsymbol{v}_n = \displaystyle \sum \limits_{i=1}^{n} \lambda_i \boldsymbol{v}_i \in \mathcal{V}$$

## Linear Independence

Consider a vector space $V = (\mathcal{V}, +, \cdot)$. Vectors $\boldsymbol{v}_1, \boldsymbol{v}_2, \dots, \boldsymbol{v}_n \in \mathcal{V}$ are called $\Large \textbf{linearly independent}$ if the only solution to the equation

$$\lambda_1 \boldsymbol{v}_1 + \lambda_2 \boldsymbol{v}_2 + \dots + \lambda_n \boldsymbol{v}_n = \boldsymbol{0}$$

is $\lambda_1 = \lambda_2 = \dots = \lambda_n = 0$.

## Linear Dependence

Vectors $\boldsymbol{v}_1, \boldsymbol{v}_2, \dots, \boldsymbol{v}_n \in \mathcal{V}$ are called $\Large \textbf{linearly dependent}$ if there exist scalars $\lambda_1, \lambda_2, \dots, \lambda_n \in \mathbb{R}$, not all zero, such that

$$\lambda_1 \boldsymbol{v}_1 + \lambda_2 \boldsymbol{v}_2 + \dots + \lambda_n \boldsymbol{v}_n = \boldsymbol{0}$$

1. A set of k vectors is either linearly independent or linearly dependent.
2. If at least one of the vectors $\boldsymbol{x}_1, \boldsymbol{x}_2, \dots, \boldsymbol{x}_k$ is a zero vector, then the set of vectors is linearly dependent. The same holds if two vectors are identical.
3. If at least one of the vectors $\boldsymbol{x}_1, \boldsymbol{x}_2, \dots, \boldsymbol{x}_k$ (where $\boldsymbol{x}_i \neq \boldsymbol{0}$ for all $i = 1, \dots, k$, $k \geq 2$) is a scalar multiple of another vector, $\boldsymbol{x}_i = \lambda \boldsymbol{x}_j$, $\lambda \in \mathbb{R}$, then the set of vectors is linearly dependent.
4. In practice, $\textbf{Gaussian elimination}$ can be used to determine if a set of vectors is linearly independent or linearly dependent. If the row echelon form of the matrix has a pivot in every column, then the vectors are linearly independent. Otherwise, they are linearly dependent.
