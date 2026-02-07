---
title: 'Denoising Probabilistic Diffusion Models'
description: 'Code and explanation for Diffusion Models'
pubDate: 'Feb 02 2026'
heroImage: '/blog-placeholder-1.jpg'
---


Denoising Probabilistic Diffusion Models is a landmark study released in 2020 by Ho et al. This blog is a latex-friendly version of the stellar blog post by <a href="https://towardsdatascience.com/diffusion-loss-every-step-explained-8c19c5e1349b/">Saptashwa Bhattacharyya</a>.
### Notations & Definitions:

Let’s start with a few of the notations we will use several times.

$x_0$: This would denote an image at time-step 0, i.e. the original image, at the start of the process. Sometimes it also refers to the image recovered in the final step of the denoising process.

$x_T$: This would be the image at the final time step. At this point, the image is simply an isotropic Gaussian noise.

Forward Process: It is a multi-step process, and in each step, an input image is corrupted with a low-level Gaussian noise. The noisy versions obtained at each time step $x_1$, $x_2$, $\ldots$ $x_T$ are obtained via a Markovian process.

$q(x_t | x_{t−1}) \equiv$ Forward process; Given an image at time step $t−1$, returns the current image

### Markov Chain (MC): 
Refers to a stochastic process (‘memoryless’) describing transitions between different states and where the probability of transitioning to any particular state is dependent solely on the current state and time elapsed. A probabilistic definition is given below:

<div style="position: relative; text-align: center;">

$$ \large P(X_n = x_n | X_{0} = x_{0}, X_1 = x_1, \ldots, X_{n-1} = x_{n-1}) = P(X_n = x_n | X_{n-1} = x_{n-1})$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(1)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 1: Definition of Markov Chain: The current state only depends only on the last state.</div>

For any positive integer $n$ and possible states $x_0$, $x_1$, $x_2, \ldots$ of the random variables $X_1$, $X_2$, $X_3, \ldots$ the Markov property states that the probability of the current state at step $n$ $(x_n)$ depends solely on the state before at $n-1$.
<hr></hr>

### Forward Process & MC:

Formally, given a data distribution $x_0 \sim q(x_0)$, the forward Markov process generates a sequence of random variables $x_1,x_2,\ldots,x_T$ with a transition kernel $q(x_t | x_{t−1})$. Using the chain rule of probability and the Markov property we can write:

<div style="text-align: center;">

$$\large q(x_1,x_2,\ldots,x_T|x_0) = q(x_{1:T}|x_0) := \prod\limits_{t=1}^{T}q(x_t|x_{t-1})$$

</div>
<div style="text-align: left; margin: -10px 0;">where</div>
<div style="position: relative; text-align: center;">

$$\large  q(x_t|x_{t-1}) := \mathcal{N} (x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(2)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 2: Forward process in diffusion as an MC process where we progressively add Gaussian Noise</div>


Here, $\beta$ is some numbers between $0$ and $1$ i.e.$\beta \in (0,1)$. In the <a href = "https://arxiv.org/pdf/2006.11239">DDPM paper</a> the noise schedule was linear. This was shown to be okay for images with high resolution but in the <a href="https://arxiv.org/pdf/2102.09672">improving diffusion</a> paper, the authors proposed an improved learning schedule which also works for sub-optimal resolutions like 32×32. In general, $\beta_1 \lt \beta_2 \lt \beta_3 \lt \ldots \lt \beta_T$ and as we move through each time step, the new Gaussian will have a mean close to $0$. Thus $q(x_T | x_0) \approx \mathcal{N}(0,\mathbf{I})$. TL;DR: Forward process slowly injects noise into data until all structures are lost.

Given the definition of a single step of the forward process, we can repeat the process $100$ times to reach the image at the $100$th time step. But what if, instead of repeated application of noise, we could sample data points for any given time step using only one step? i.e if $q(x_t | x_{t−1})$ is known, can we find an expression for $q(x_t | x_0)$? Then starting from the original image i.e. $x_0$, we can sample the noisy image at a given time step t ($x_t$).

This is possible via the Reparametrization trick.

Reparametrization Trick: The main idea stems from the fact that all normal distributions are just scaled and translated versions of $\mathcal{N}(0, \mathbf{I})$. Given a normal random variable $X$, from a distribution of arbitrary $\mu$ $\text{\&}$ $\sigma$, we can write:

<div style="text-align: center;">

$$\large X \sim \mathcal{N}(\mu, \sigma^2) = \mu + \sigma \odot \epsilon; \quad \epsilon \in \mathcal{N} (0, \mathbf{I})$$

</div>

<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Reparameterization Trick</div>

Using the reparametrization trick, we can now think of an image at time step t:


<div style="position: relative; text-align: center;">

$$\large x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\epsilon; \quad \epsilon \in \mathcal{N} (0, \mathbf{I}) \quad \text{\&} \quad \alpha_t := 1 - \beta_t$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(3)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 3: Applying the reparametrization trick in Eq. 2 for a single-step forward process.</div>


If we know $x_t$ (say an image at time $t$), can we estimate how the image was at timestep $t-1$ and $t-2$ and so on? Let’s get started:

<div style="position: relative; text-align: center;">

$$\large x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\epsilon; \quad \epsilon \in \mathcal{N} (0, \mathbf{I})$$

$$\large x_{t-1} = \sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_{t-1}}\epsilon; \quad t \rightarrow t - 1$$

$$\large x_t = \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} +  \sqrt{\alpha_t - \alpha_t\alpha_{t-1}}\epsilon + \sqrt{1 - \alpha_t}\epsilon; \quad using \quad x_{t-1}$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(4)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 4: Steps to generalize Eq. 3 where we want to write $x_t$ in terms of $x_{t-2}$</div>

In Eq. $4$, first, we rewrote Eq.$3$ by changing $t$ to $t-1$ (second line). Then replace $x_{t-1}$ in the first line with this new definition. Can we simplify this expression a bit? Yes! For that, we need to apply two properties of Normal distribution to reach the third line; These properties are:

1. Scaled random variable: If
$$ X \sim \mathcal{N}(\mu, \sigma^2)$$,  then  $$k \gt 0; \quad kX \sim \mathcal{N} (k\mu, k^2\sigma^2)$$

2. Merging two Gaussians: If we merge $$ \mathcal{N_1}(0, \sigma_1^2\mathbf{I}) $$   and   $$ \mathcal{N_2}(0, \sigma_2^2\mathbf{I}) $$  then the merged distribution is $$\mathcal{N}_m(0, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$$

<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Two simple properties of Normal distribution that were used in Eq. 4 (from 2nd to 3rd line)</div>

Applying these two properties on the final line of Eq. $4$, we can rewrite:

<div style="position: relative; text-align: center;">


$$\large x_t=\sqrt{\alpha_t \alpha_{t-1}} x_{t-2}+\underbrace{\sqrt{\alpha_t-\alpha_t \alpha_{t-1}} \epsilon+\sqrt{1-\alpha_t} \epsilon}_{\text {Merge these two scaled normal distribution }};\quad \epsilon \sim \mathcal{N}(0, I) $$

$$\large x_t=\sqrt{\alpha_t \alpha_{t-1}} x_{t-2}+\sqrt{1-\alpha_t \alpha_{t-1}} \epsilon$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(5)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 5: Simplifying the final line of Eq. 4 with the above properties of the normal distribution</div>

Can we progressively go down the time step in Eq. $5$ where one can write $x_t$ (noisy image at time step t) in terms of $x_0$ (i.e. the original image)? The answer is yes; Below is how we do this:

<div style="position: relative; text-align: center;">



$$\large x_t =\sqrt{\alpha_t} x_{t-1}+\sqrt{1-\alpha_t} \epsilon$$

$$\large  =\sqrt{\alpha_t \alpha_{t-1}} x_{t-2}+\sqrt{1-\alpha_t \alpha_{t-1}} \epsilon $$

$$\large \vdots $$

$$\large =\sqrt{\alpha_t \alpha_{t-1} \ldots \alpha_1} x_0+\sqrt{1-\alpha_t \alpha_{t-1} \ldots \alpha_1} \epsilon $$

$$\large = \sqrt{\overline{\alpha_t}} x_0+\sqrt{1-\overline{\alpha_t}} \epsilon ; \quad \overline{\alpha_t}  := \prod\limits_{s=1}^{t} \alpha_s $$

$$\large q\left(x_t \mid x_0\right) = \mathcal{N}\left(x_t ; \sqrt{\overline{\alpha_t}} x_0,\left(1-\overline{\alpha_t}\right) \mathbf{I}\right) ; \text { reparametrization trick }!$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(6)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 6: Turning Eq. 5 into a more generalizable form and eventually writing the forward process in a single step</div>

What we have achieved here through simple math is a little bit of special expression: From Eq. $3$, where the sampling process at time step $t$ required knowledge of the data at $t-1$, to here in Eq. $6$, we can sample data points for any given step using only $1$ step and the initial input image. So now the forward process has been simplified greatly and any noisy image at time step t is directly related to the original image via the noise scheduler $\alpha$.

Regarding VAE, we can also think of the forward process as the data encoding step. The difference from VAE though is that in Diffusion it is a ‘fixed’ process and no neural network is involved. In terms of data spaces, we think of the forward process as pushing a data point from the original data distribution towards an isotropic Gaussian noise and the reverse process tries to achieve the opposite. Let’s go in a bit more detail for the reverse process.

<hr></hr>

### Reverse Process:

After the forward process has finished, we have a latent $x_T$ which is an isotropic Gaussian. If we know the exact reverse distribution $q(x_{t-1}|x_t)$, we can start sampling from the isotropic Gaussian noise $x_T \sim \mathcal{N}(0, \mathbf{I})$, then we can reverse the process (denoise).

But $q(x_{t-1}|x_t)$ is complicated to estimate and we can check this with the Bayes theorem:

<div style="position: relative; text-align: center;">

$$\Large q(x_{t-1}|x_t) = \frac{q(x_t|x_{t-1})q(x_{t-1})}{q(x_t)}$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(7)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 7: The reverse process is complicated to calculate because we don’t know the ratio q(x_{t-1})/q(x_t);</div>



As we already know the forward process $$q(x_t|x_{t−1})$$, to calculate the reverse step we need to know the ratio $$q(x_{t−1})/q(x_t)$$; Since this is difficult, we approximate $$q(x_{t−1} | x_t)$$ with a neural net. Whenever we think about approximating one distribution with another your mind should already head towards KL divergence and possibly variational lower bound (evidence lower bound/ELBO).

The reverse process is defined as an $\emph{MC}$ with $\textbf{LEARNED}$ Gaussian transitions and a single step for the reverse process then could be written as:

<div style="position: relative; text-align: center;">

$$\large p(x_{t-1}|x_t) := \mathcal{N}(x_{t-1}; \; \mu_\theta(x_t, t), \; \Sigma_\theta(x_t, t))$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(8)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 8: One step in the reverse process: Learned (μ, Σ is unknown initially) Normal distribution.</div>


Here, $\mu$, and $\Sigma$ i.e. the mean and the variance (covariance matrix for multi-variate normal) are parametrized and learned (neural-net comes here). The authors in the DDPM paper assumed that the covariance matrix is fixed to a certain variance schedule as below:

<div style="position: relative; text-align: center;">

$$\large \Sigma_\theta(x, t) = \sigma_t^2 \mathbf{I}; \rightarrow \sigma_t^2 = \beta_t; \quad or \quad \sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(9)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 9: The Σ in the Learned MC (Eq. 7) Normal is fixed to the noise scheduling parameter.</div>

Both the variance schedules in the equation above gave similar results. So what we are left with is to train the parametrized mean ($\mu$) and this is done in the Diffusion model; The complete reverse process can also be defined as a joint distribution of the isotropic Gaussian Noise $\mathcal{N}(0, \mathbf{I}) \equiv p(x_T)$ (which is the starting point of the reverse process) and the product of single-step reverse processes through 1 to T as below:

<div style="position: relative; text-align: center;">

$$\large p_\theta(x_{0:T}) := p(x_T) \prod\limits_{t=1}^T p_\theta(x_{t-1}|x_t)$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(10)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 10: Reverse process through timestep 0 to T as a joint distribution</div>


$x_1,\ldots,x_T$ are latents of the same dimensionality as the data i.e. before and after applying noise the image size remains the same. Here again, we can compare the Diffusion model to a VAE, where the forward process can be compared to the encoding and the reverse process can be compared with the decoding. But in Diffusion, by definition the forward process is fixed, so we need to train only 1 network; whereas in VAE we need to train both the encoder and decoder jointly. This also aligns at this stage very well with the derivation of the loss function and that’s also similar to VAE; Latent Variable Models (LVMs) which we have discussed before in the context of the Gaussian Mixture Model, usually go hand in hand with the Evidence Lower Bound (ELBO).


What we would like to minimize would be $−log(p_\theta(x_0))$; $p_\theta$ is the parametrized distribution and $x_0$ is the image at time step 0.




### Diffusion as a Latent Variable Model & ELBO:

Minimizing $−log(p_\theta(x_0))$ is not an easy-to-compute quantity and the reason for that is $x_0$ would depend on all other time-steps $x_1, x_2,\ldots,x_T$, very specifically we need to perform an intractable integration

<div style="position: relative; text-align: center;">

$$\large p_\theta(x_{0}) := \Huge  \int \large p_\theta(x_{0:T}) \, dx_{1:T}  $$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(10)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 10: Reverse process through timestep 0 to T as a joint distribution</div>

To obtain $p_\theta(x_0)$ we need to marginalize over all other time steps, i.e. all possible ways to arrive at the denoised image. This is simply not possible (there could be infinite ways); Here we can use Evidence Lower Bound to obtain a lower bound on the log-likelihood. Since I have discussed ELBO in the context of LVMs and Probabilistic PCA, let’s use the definition here: In the process of computing likelihood $log p(x)$, if we need to compute the posterior $p(z|x)$ where $z$ is a latent variable which is difficult to compute and can be approximated via another simple parametrized distribution $q(z|x)$ then:

<div style="position: relative; text-align: center;">

$$\large ELBO = \mathbb{E}_{q(z|x)} \left[log\; p(x, z)\right] - \mathbb{E}_{q(z|x)} \left[log\; q(z|x)\right]; $$

$$\large log p(x) \geq \mathbb{E}_{q(z|x)} \left[log\; p(x, z)\right] - \mathbb{E}_{q(z|x)} \left[log\; q(z|x)\right] = ELBO$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(11)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 11: Definition of Evidence lower bound where q(z|x) is a flexible approximate variational distribution.</div>

It’s also possible to rewrite the above expression with little mathematical manipulation using KL Divergence as below:

<div style="position: relative; text-align: center;">

$$\large ELBO = \mathbb{E}_{q(z|x)} \left[log\; p(x|z)\right] - KL(q(z|x)||p(z)) $$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(12)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 12: Writing Eq. 9 including the KL divergence term.</div>

We will use this expression (Eq. 12) for deriving the loss function. In the diffusion model $x_0$ (image at timestep 0) will represent the true data and $x_1,\\dots,x_T$ are the latent variables i.e. $x_{1:T}$. Let’s use this in Eq. 10:

<div style="position: relative; text-align: center;">

$$\large ELBO = \mathbb{E}_{q(x_{1:T}|x_0)} \left[log\; p(x_0|x_{t:T})\right] - KL(q(x_{1:T}|x_0)||p(x_{1:T})) $$

$$\large = \mathbb{E}_q\left[log\;p(x_0|x_{1:T})\right] - \mathbb{E}_q\left[log\;q(x_{1:T}|x_0)\right] + \mathbb{E}_q\left[log\;p(x_{1:T})\right] $$

$$\large = \mathbb{E}_q\left[log\;p(x_0|x_{1:T}) + log\; \frac{p(x_{1:T})}{q(x_{1:T}|x_0)} \right]$$

$$\large = \mathbb{E}_q\left[log \; \frac{p(x_{0:T})}{q(x_{1:T}|x_0)} \right]$$

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(13)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 13: Use the images at different timesteps x_1 through x_T as latent variables in Eq. 10.</div>

In the equation above, I am writing $q(x_{1:T}|x_0)$ as just q as a short notation. Also, I have used the definition of KL divergence in the second step. Since the above expression for ELBO contains the log of the ratio of two distributions we know from before (Eq. 2 and Eq. 9), we will just use them here:

<div style="position: relative; text-align: center;">

$$\large ELBO = \mathbb{E}_q\left[log \; \frac{p(x_{0:T})}{q(x_{1:T}|x_0)} \right]$$

$$\large = \mathbb{E}_q\left[log \; \frac{p(x_T) \prod\limits_{t=1}^{T} p_\theta(x_{t-1}|x_{t})}{\prod\limits_{t=1}^{T} q(x_t|x_{t-1})} \right]$$

$$\large = \mathbb{E}_q\left[log\; p(x_T) + \sum\limits_{t \geq 1} log\; \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})} \right]$$

$$\large = \mathbb{E}_q\left[log\; p(x_T) + \sum\limits_{t \geq 2} log\; \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})} + log\; \frac{p_\theta(x_0|x_1)}{q(x_1|x_0)} \right]$$


<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(14)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 14: Using the definition from Eq. 2 and Eq. 9 in Eq. 12.</div>


What happened here (Eq. 14) is that we have used the definition of the forward and reverse process and separated 3 different terms; The reason we have separated the term for t=1 and grouped all terms from t≥ 2 will be clear soon.

If we look at the summation term, the numerator refers to the reverse process, but the denominator is still the forward process; To turn it into a reverse process, we can use Bayes’ Theorem (Eq. 7) here with a twist; The terms on the right in Eq. 7 have high variance because it’s very difficult to know what actually was the original image since the reverse process starts from an isotropic Gaussian Noise. So, the authors conditioned these distributions also with the original image ($x_0$):


<div style="position: relative; text-align: center;">

$$\large ELBO = \mathbb{E}_q\left[log\; p(x_T) + \sum\limits_{t \geq 2} log\; \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1}, x_0)} \cdot \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)} + log\; \frac{p_\theta(x_0|x_1)}{q(x_1|x_0)} \right]$$

$$\large = \mathbb{E}_q\left[log\; p(x_T) + \sum\limits_{t \geq 2} log\; \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1}, x_0)} + \sum\limits_{t \geq 2} log\; \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)} + log\; \frac{p_\theta(x_0|x_1)}{q(x_1|x_0)} \right]

<div style="position: absolute; top: 50%; right: 0; transform: translateY(-50%); font-family: 'KaTeX_Main', 'Times New Roman', serif; font-size: 1.2em;">(14)</div>
</div>
<div style="text-align: center; font-size: 0.9em; color: gray; margin-top: -1em;">Eq. 14: Using the definition from Eq. 2 and Eq. 9 in Eq. 12.</div>

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
```

