---
title: 'Diffusion Models and Flow Matching'
description: 'A mathematical foundation for diffusion models and flow matching, from probability and path-space measures to ODEs, SDEs, and score matching.'
date: 2026-08-21
updatedDate: 2026-08-23
tags:
  - diffusion-models
  - flow-matching
  - generative-modeling
  - probability
language: 'en'
translationKey: '20260821---diffusion-models-and-flow-matching'
type: note
status: ready
draft: false
---

# Diffusion Models and Flow Matching

Generative modeling treats the objects to be generated as random vectors. An image, video, or
molecular configuration can usually be flattened into a vector $z\in\mathbb{R}^d$. The data are
modeled by an unknown probability law $p_{\mathrm{data}}$, and generation means constructing an
algorithm that returns approximate samples from that law.

Flow matching and diffusion models do this by transporting samples from a simple initial law
$p_{\mathrm{init}}$, usually a standard Gaussian, to $p_{\mathrm{data}}$. Flow models use ordinary
differential equations (ODEs); diffusion models allow stochastic differential equations (SDEs).

Throughout these notes, time runs from noise at $t=0$ to data at $t=1$:

$$
X_0\sim p_{\mathrm{init}},
\qquad
X_1\sim p_{\mathrm{data}}.
$$

## Contents

- [[#Generative modeling as sampling]]
- [[#Probability preliminaries]]
- [[#Transition kernels and path-space measures]]
- [[#Expectation notation in ML objectives]]
- [[#ODE and SDE preliminaries]]
- [[#Flow and diffusion models]]
- [[#Flow matching]]
- [[#Score functions and score matching]]

## Generative modeling as sampling

A dataset is modeled as independent samples

$$
z_1,\ldots,z_N\overset{\mathrm{i.i.d.}}{\sim}p_{\mathrm{data}}.
$$

The law $p_{\mathrm{data}}$ need not be known analytically. In practice, the empirical dataset is
the only direct access we have to it.

> [!note] Definition - Generative model
> A **generative model** is a learned sampling procedure whose output distribution approximates $p_{\mathrm{data}}$.

If generation is conditioned on side information $y$, such as a class label or text prompt, the
target becomes the conditional law

$$
z\sim p_{\mathrm{data}}(\cdot\mid y).
$$

The unconditional problem is the mathematical core. Conditioning changes the target law but not
the basic transport ideas developed below.

## Probability preliminaries

> [!info] Canonical probability foundation
> This section adapts [[Maths for RL#Probability spaces, random variables, and expectation]] and
> [[Maths for RL#Conditional probability and conditional expectation]]. Transition kernels,
> trajectory laws, application-specific sampling notation, and control-process material are
> intentionally omitted.

### Probability spaces, random variables, and laws

> [!note] Definition - Probability space
> A probability space is a triple $(\Omega,\mathcal{F},\mathbb{P})$:
>
> - $\Omega$ is the set of possible outcomes.
> - $\mathcal{F}$ is a $\sigma$-algebra of measurable events.
> - $\mathbb{P}$ is a probability measure on those events.

A random variable is a measurable map $X:\Omega\to\mathcal{X}$. Its distribution, or **law**, is
the pushforward measure

$$
\mu_X(A)=\mathbb{P}(X\in A),
\qquad
A\subseteq\mathcal{X}.
$$

For an integrable function $f$, expectation is integration with respect to a probability measure:

$$
\mathbb{E}[f(X)]
=\int_\Omega f(X(\omega))\,\mathbb{P}(d\omega)
=\int_\mathcal{X}f(x)\,\mu_X(dx).
$$

If $X$ has finite or countable support, the same integral is a sum:

$$
\mathbb{E}[f(X)]
=\sum_{x\in\mathcal{X}}f(x)\,\mathbb{P}(X=x).
$$

A probability mass function or density is a representation of a probability measure relative to
counting or Lebesgue measure. Not every probability measure has a density. A statement that fails
only on a set of probability zero holds **almost surely**, abbreviated a.s.

### Densities

If an $\mathbb{R}^d$-valued random variable $X$ has density $p_X$, then

$$
\mathbb{P}(X\in A)
=\int_A p_X(x)\,dx,
\qquad
\int_{\mathbb{R}^d}p_X(x)\,dx=1.
$$

### Multivariate Gaussian distributions

> [!note] Definition - Multivariate Gaussian
> A random vector $X\in\mathbb{R}^d$ is multivariate Gaussian with mean
> $\mu\in\mathbb{R}^d$ and covariance matrix $\Sigma\in\mathbb{R}^{d\times d}$ if
>
> $$
> X\sim\mathcal{N}(\mu,\Sigma).
> $$
>
> Its first two moments are
>
> $$
> \mathbb{E}[X]=\mu,
> \qquad
> \operatorname{Cov}(X)
> =\mathbb{E}\!\left[
> (X-\mu)(X-\mu)^\mathsf{T}
> \right]
> =\Sigma.
> $$
>
> The covariance matrix is symmetric and positive semidefinite.

If $\Sigma$ is positive definite, the distribution has density

$$
\mathcal{N}(x;\mu,\Sigma)
=\frac{1}{(2\pi)^{d/2}\det(\Sigma)^{1/2}}
\exp\!\left(
-\frac{1}{2}
(x-\mu)^\mathsf{T}\Sigma^{-1}(x-\mu)
\right).
$$

If $\Sigma$ is singular, the Gaussian law is still well-defined but is supported on a lower
dimensional affine subspace and does not have a density with respect to $d$-dimensional Lebesgue
measure.

#### Affine construction and sampling

> [!abstract] Proposition - Affine images of Gaussians
> If $\varepsilon\sim\mathcal{N}(0,I_m)$ and
> $X=\mu+A\varepsilon$ for $A\in\mathbb{R}^{d\times m}$, then
>
> $$
> X\sim\mathcal{N}(\mu,AA^\mathsf{T}).
> $$
>
> More generally, if $X\sim\mathcal{N}(\mu,\Sigma)$, then
>
> $$
> BX+b
> \sim
> \mathcal{N}\!\left(
> B\mu+b,\,
> B\Sigma B^\mathsf{T}
> \right).
> $$

To sample from $\mathcal{N}(\mu,\Sigma)$, choose a matrix $L$ satisfying
$LL^\mathsf{T}=\Sigma$, for example a Cholesky factor when $\Sigma$ is positive definite, and set

$$
X=\mu+L\varepsilon,
\qquad
\varepsilon\sim\mathcal{N}(0,I_d).
$$

If $X$ and $Y$ are independent Gaussian vectors, then

$$
X+Y
\sim
\mathcal{N}(
\mathbb{E}[X]+\mathbb{E}[Y],\,
\operatorname{Cov}(X)+\operatorname{Cov}(Y)
).
$$

#### Isotropic and diagonal Gaussians

> [!example] Example - Isotropic Gaussian
> If $\Sigma=\sigma^2I_d$, all coordinates have variance $\sigma^2$ and are independent. The
> density becomes
>
> $$
> \mathcal{N}(x;\mu,\sigma^2I_d)
> =(2\pi\sigma^2)^{-d/2}
> \exp\!\left(
> -\frac{\lVert x-\mu\rVert^2}{2\sigma^2}
> \right).
> $$
>
> The standard Gaussian $\mathcal{N}(0,I_d)$ is commonly used for $p_{\mathrm{init}}$ because it is
> easy to sample and stable under linear transformations.

A diagonal covariance matrix represents coordinates with different variances but no correlations:

$$
\Sigma
=\operatorname{diag}(
\sigma_1^2,\ldots,\sigma_d^2
).
$$

For the isotropic Brownian increment used in Euler-Maruyama,

$$
\sqrt{h}\,\sigma_t\varepsilon
\sim
\mathcal{N}(0,h\sigma_t^2I_d).
$$

This is why the implementation multiplies standard Gaussian noise by $\sqrt{h}\,\sigma_t$.

#### Score of a Gaussian

> [!abstract] Proposition - Gaussian score
> For a nondegenerate Gaussian density,
>
> $$
> \nabla_x\log\mathcal{N}(x;\mu,\Sigma)
> =-\Sigma^{-1}(x-\mu).
> $$
>
> In the isotropic case,
>
> $$
> \nabla_x\log\mathcal{N}(x;\mu,\sigma^2I_d)
> =-\frac{x-\mu}{\sigma^2}.
> $$
>
> This formula becomes the conditional score used later for Gaussian probability paths.

### Conditional probability and conditional expectation

> [!note] Definition - Conditional expectation
> Let $Y$ be integrable and let $\mathcal{G}\subseteq\mathcal{F}$ represent the information
> available. A conditional expectation $U=\mathbb{E}[Y\mid\mathcal{G}]$ is a
> $\mathcal{G}$-measurable random variable satisfying
>
> $$
> \int_G U\,d\mathbb{P}
> =\int_G Y\,d\mathbb{P}
> \qquad
> \text{for every }G\in\mathcal{G}.
> $$
>
> It is unique up to almost-sure equality.

On the standard Borel spaces used here, conditional laws can be represented by a regular
conditional distribution $K_{Y\mid X}(x,dy)$. For suitable $g$,

$$
\mathbb{E}[g(Y)\mid X]
=\int_\mathcal{Y}g(y)K_{Y\mid X}(X,dy).
$$

When $Y$ is discrete, this becomes

$$
\mathbb{E}[g(Y)\mid X=x]
=\sum_{y\in\mathcal{Y}}
g(y)\mathbb{P}(Y=y\mid X=x).
$$

> [!abstract]- Rules of conditional expectation
> The rules used most often are:
>
> - **Linearity:**
>
>   $$
>   \mathbb{E}[aY+bZ\mid\mathcal{G}]
>   =a\mathbb{E}[Y\mid\mathcal{G}]
>   +b\mathbb{E}[Z\mid\mathcal{G}].
>   $$
>
> - **Taking out what is known:** if $Z$ is $\mathcal{G}$-measurable and the terms are integrable,
>   then
>
>   $$
>   \mathbb{E}[ZY\mid\mathcal{G}]
>   =Z\mathbb{E}[Y\mid\mathcal{G}].
>   $$
>
> - **Tower property:** if $\mathcal{H}\subseteq\mathcal{G}$, then
>
>   $$
>   \mathbb{E}[
>   \mathbb{E}[Y\mid\mathcal{G}]
>   \mid\mathcal{H}]
>   =\mathbb{E}[Y\mid\mathcal{H}].
>   $$
>
> - **Total expectation:**
>
>   $$
>   \mathbb{E}[
>   \mathbb{E}[Y\mid\mathcal{G}]
>   ]
>   =\mathbb{E}[Y].
>   $$
>
> - **Independence:** if $Y$ is independent of $\mathcal{G}$, then
>
>   $$
>   \mathbb{E}[Y\mid\mathcal{G}]
>   =\mathbb{E}[Y]
>   \qquad\text{a.s.}
>   $$

### Laws of total probability and total expectation

> [!abstract]- Proposition - Disintegration over a conditioning variable
> If $X$ is discrete,
>
> $$
> \begin{aligned}
> \mathbb{P}(Y\in A)
>   &=\sum_x
>     \mathbb{P}(Y\in A\mid X=x)\mathbb{P}(X=x),\\
> \mathbb{E}[g(Y)]
>   &=\sum_x
>     \mathbb{E}[g(Y)\mid X=x]\mathbb{P}(X=x).
> \end{aligned}
> $$
>
> For a general conditioning variable, sums become integrals against its law $\mu_X$:
>
> $$
> \begin{aligned}
> \mathbb{P}(Y\in A)
>   &=\int K_{Y\mid X}(x,A)\,\mu_X(dx),\\
> \mathbb{E}[g(Y)]
>   &=\int
>     \mathbb{E}[g(Y)\mid X=x]\,\mu_X(dx).
> \end{aligned}
> $$
>
> If the relevant densities exist, the continuous law of total probability is
>
> $$
> p_Y(y)
> =\int p_{Y\mid X}(y\mid x)p_X(x)\,dx.
> $$

### Bayes' rule and posterior averages

For jointly distributed random variables $X$ and $Z$ with densities, Bayes' rule gives

$$
p_{Z\mid X}(z\mid x)
=\frac{p_{X\mid Z}(x\mid z)p_Z(z)}{p_X(x)}
$$

whenever $p_X(x)>0$. Conditional expectation averages with respect to this posterior:

$$
\mathbb{E}[g(Z)\mid X=x]
=\int g(z)p_{Z\mid X}(z\mid x)\,dz.
$$

This posterior-average form is the basis of the marginalization identities used in both flow
matching and score matching.

> [!tip] Theorem - Conditional expectation as least-squares regression
> Among all measurable functions $f(X)$, the minimizer of
>
> $$
> \mathbb{E}\!\left[
> \lVert Y-f(X)\rVert^2
> \right]
> $$
>
> is
>
> $$
> f(X)=\mathbb{E}[Y\mid X]
> \qquad\text{a.s.}
> $$
>
> Therefore, regression against a random conditional target recovers its posterior mean.

## Transition kernels and path-space measures

> [!info] Canonical kernel foundation
> This section adapts the framework-neutral part of
> [[Maths for RL#Transition kernels and measures on paths]]. The specialization to policies,
> environments, and controlled trajectories is omitted.

### Stochastic kernels

> [!note] Definition - Stochastic kernel
> Let $(\mathcal{X},\mathscr{X})$ and $(\mathcal{Y},\mathscr{Y})$ be measurable spaces. A
> **stochastic kernel** $K$ from $\mathcal{X}$ to $\mathcal{Y}$ assigns a probability measure
> $K(x,\cdot)$ on $\mathcal{Y}$ to each $x\in\mathcal{X}$ such that:
>
> 1. $B\mapsto K(x,B)$ is a probability measure for each fixed $x$.
> 2. $x\mapsto K(x,B)$ is measurable for each fixed $B\in\mathscr{Y}$.

We often write $K(x,dy)$. Integrating a measurable function $g$ against the conditional law gives

$$
\int_\mathcal{Y}g(y)K(x,dy).
$$

On a finite space, define $K(x,y)=K(x,\{y\})$. Then

$$
\sum_{y\in\mathcal{Y}}K(x,y)=1,
\qquad
\int_\mathcal{Y}g(y)K(x,dy)
=\sum_{y\in\mathcal{Y}}g(y)K(x,y).
$$

A Markov transition kernel is a stochastic kernel whose input represents the current state and
whose output is a distribution over the next state.

### From kernels to a joint law

An initial distribution and a sequence of kernels determine the joint law of all coordinates.
Let $\mu_0$ be a probability measure on $\mathcal{X}_0$, and let

$$
K_k(x_0,\ldots,x_k,dx_{k+1})
$$

select the next coordinate given the history.

> [!tip]- Theorem - Ionescu-Tulcea construction
> The initial law $\mu_0$ and kernels $(K_k)_{k\geq 0}$ determine a unique probability measure
> $\mathbb{P}$ on the product space
>
> $$
> \Omega=\prod_{k=0}^{\infty}\mathcal{X}_k.
> $$
>
> Its probabilities on cylinder events are given by iterated integration:
>
> $$
> \begin{aligned}
> &\mathbb{P}(X_0\in A_0,\ldots,X_n\in A_n)\\
> &\quad=
> \int_{A_0}\mu_0(dx_0)
> \int_{A_1}K_0(x_0,dx_1)\cdots
> \int_{A_n}K_{n-1}(x_0,\ldots,x_{n-1},dx_n).
> \end{aligned}
> $$

For a measurable function $F$ of a finite prefix,

$$
\begin{aligned}
\mathbb{E}_{\mathbb{P}}[F(X_0,\ldots,X_n)]
&=
\int \mu_0(dx_0)
\int K_0(x_0,dx_1)\cdots\\
&\qquad
\int K_{n-1}(x_0,\ldots,x_{n-1},dx_n)
F(x_0,\ldots,x_n).
\end{aligned}
$$

The compact expression

$$
\mathbb{P}(dx_0,\ldots,dx_n)
=\mu_0(dx_0)
K_0(x_0,dx_1)\cdots
K_{n-1}(x_0,\ldots,x_{n-1},dx_n)
$$

is shorthand for this iterated construction, not ordinary multiplication of numbers.

If the process is Markov, the next-coordinate kernel depends only on the current coordinate:

$$
K_k(x_0,\ldots,x_k,dx_{k+1})
=K_k(x_k,dx_{k+1}).
$$

### Path laws for ODEs and SDEs

For a deterministic flow from time $s$ to $t$, the transition kernel is a Dirac measure:

$$
K_{s,t}(x,dy)
=\delta_{\psi_{s,t}(x)}(dy).
$$

For a Markov SDE, the transition kernel is

$$
K_{s,t}(x,B)
=\mathbb{P}(X_t\in B\mid X_s=x).
$$

> [!example] Example - Euler-Maruyama transition
> For the discretization
>
> $$
> X_{t_{k+1}}
> =X_{t_k}
> +h\,u_{t_k}(X_{t_k})
> +\sqrt{h}\,\sigma_{t_k}\varepsilon_k,
> \qquad
> \varepsilon_k\sim\mathcal{N}(0,I_d),
> $$
>
> the one-step transition kernel is Gaussian:
>
> $$
> K_k(x,\cdot)
> =\mathcal{N}\!\left(
> x+h\,u_{t_k}(x),
> h\sigma_{t_k}^2I_d
> \right).
> $$

On a time grid, the initial law and these kernels induce a joint distribution on
$(X_{t_0},\ldots,X_{t_n})$. In continuous time, an ODE or SDE with continuous trajectories induces a **path law** on

$$
C([0,1],\mathbb{R}^d),
$$

the space of continuous functions from time to state space. Expressions such as
$\mathbb{E}_{\mathbb{P}}[F((X_t)_{0\leq t\leq 1})]$ mean integration of a path functional $F$ against this law.

## Expectation notation in ML objectives

> [!info] Relation to the canonical notation section
> This section specializes the general discussion in
> [[Maths for RL#Probability notation versus ML/RL notation]] to matching objectives and ordinary
> supervised learning.

### Sampling subscripts specify a joint law

ML notation often describes an expectation by listing how its random inputs are sampled. For example,

$$
\mathbb{E}_{\substack{
t\sim\operatorname{Unif}[0,1],\,
Z\sim p_{\mathrm{data}},\\
X_t\sim p_t(\cdot\mid Z)
}}
[\ell(t,Z,X_t)]
$$

means:

1. sample $t$ uniformly from $[0,1]$;
2. sample $Z$ from the data law;
3. conditionally on $(t,Z)$, sample $X_t$ from $p_t(\cdot\mid Z)$;
4. evaluate $\ell(t,Z,X_t)$ and average.

Let $\mu_{\mathrm{data}}$ denote the probability measure written informally as
$p_{\mathrm{data}}$. The sampling instructions define the joint measure

$$
\nu(dt,dz,dx)
=dt\,\mu_{\mathrm{data}}(dz)\,p_t(dx\mid z)
$$

on $[0,1]\times\mathcal{Z}\times\mathcal{X}$. The expectation is the iterated integral

$$
\begin{aligned}
&\mathbb{E}_{\substack{
t\sim\operatorname{Unif}[0,1],\,
Z\sim p_{\mathrm{data}},\\
X_t\sim p_t(\cdot\mid Z)
}}
[\ell(t,Z,X_t)]\\
&\qquad=
\int_0^1
\int_\mathcal{Z}
\int_\mathcal{X}
\ell(t,z,x)\,
p_t(dx\mid z)\,
\mu_{\mathrm{data}}(dz)\,dt.
\end{aligned}
$$

> [!note] How to read multiple subscripts
> The subscript is a compact description of one joint probability law. Conditional phrases such
> as $X_t\sim p_t(\cdot\mid Z)$ specify kernels, while independently sampled quantities contribute
> product factors. The notation does not represent several unrelated expectations.

### Example: the conditional flow-matching loss

For a Gaussian probability path, the loss

$$
\mathcal{L}_{\mathrm{CFM}}(\theta)
=\mathbb{E}_{\substack{
t\sim\operatorname{Unif}[0,1],\,
Z\sim p_{\mathrm{data}},\\
\varepsilon\sim\mathcal{N}(0,I_d)
}}
\left[
\left\lVert
u_t^\theta(\alpha_tZ+\beta_t\varepsilon)
-(\dot{\alpha}_tZ+\dot{\beta}_t\varepsilon)
\right\rVert^2
\right]
$$

is shorthand for an integral over the joint law

$$
dt\,\mu_{\mathrm{data}}(dz)\,\gamma(d\varepsilon),
\qquad
\gamma=\mathcal{N}(0,I_d).
$$

Equivalently, it can be read as a nested expectation:

$$
\mathbb{E}_{t}
\mathbb{E}_{Z}
\mathbb{E}_{\varepsilon}
\left[
\left\lVert
u_t^\theta(\alpha_tZ+\beta_t\varepsilon)
-(\dot{\alpha}_tZ+\dot{\beta}_t\varepsilon)
\right\rVert^2
\right],
$$

where the displayed sampling laws determine each expectation.

### Population expectations and empirical means

The **population objective** integrates over the unknown data law:

$$
\mathcal{L}(\theta)
=\mathbb{E}_{Z\sim p_{\mathrm{data}}}
[f_\theta(Z)].
$$

Given a dataset $\mathcal{D}=\{z_1,\ldots,z_N\}$, its empirical measure is

$$
\widehat{\mu}_N
=\frac{1}{N}\sum_{i=1}^N\delta_{z_i}.
$$

Expectation with respect to the empirical measure is exactly the sample mean:

$$
\mathbb{E}_{Z\sim\widehat{\mu}_N}[f_\theta(Z)]
=\frac{1}{N}\sum_{i=1}^N f_\theta(z_i).
$$

If each data point is combined with fresh auxiliary randomness $\xi$, such as a time and Gaussian
noise, the full empirical objective is

$$
\widehat{\mathcal{L}}_N(\theta)
=\frac{1}{N}\sum_{i=1}^N
\mathbb{E}_{\xi}
[\ell_\theta(z_i,\xi)].
$$

Training usually replaces this by a minibatch Monte Carlo estimator:

$$
\widehat{\mathcal{L}}_{\mathcal{B}}(\theta)
=\frac{1}{|\mathcal{B}|}
\sum_{i\in\mathcal{B}}
\ell_\theta(z_i,\xi_i),
$$

where the $\xi_i$ are newly sampled for the current update.

> [!warning] Population, empirical, and minibatch notation are often conflated
> Authors sometimes write $\mathbb{E}_{Z\sim p_{\mathrm{data}}}$ while the implementation actually
> averages over a finite dataset or minibatch. These are related but distinct:
>
> - the **population expectation** averages over the unknown true law;
> - the **empirical expectation** averages over all observed samples;
> - the **minibatch mean** is a stochastic estimator of the empirical objective;
> - if the dataset itself is i.i.d., the empirical objective also estimates the population one.

## ODE and SDE preliminaries

### Time-dependent laws and deterministic flows

A **stochastic process** $(X_t)_{0\leq t\leq 1}$ is a family of random variables indexed by time.
Its time-$t$ law is denoted by $p_t$:

$$
X_t\sim p_t.
$$

The family $(p_t)_{0\leq t\leq 1}$ records only the marginal distribution at each time. It does not
by itself specify how $X_s$ and $X_t$ are jointly coupled across time.

A time-dependent vector field is a function

$$
u:\mathbb{R}^d\times[0,1]\to\mathbb{R}^d,
\qquad
(x,t)\mapsto u_t(x).
$$

The ODE generated by $u_t$ is

$$
\frac{d}{dt}X_t=u_t(X_t),
\qquad
X_0=x_0.
$$

Under standard regularity conditions, its solution defines a flow map
$\psi_t:\mathbb{R}^d\to\mathbb{R}^d$ such that $X_t=\psi_t(X_0)$. If $X_0$ is random, then the
law of $X_t$ is the pushforward

$$
p_t=(\psi_t)_{\#}p_0.
$$

### Brownian motion

> [!note] Definition - Brownian motion
> A $d$-dimensional Brownian motion $(W_t)_{t\geq 0}$ is a stochastic process satisfying:
>
> 1. $W_0=0$ almost surely.
> 2. Its paths are continuous almost surely.
> 3. For $0\leq s<t$, $W_t-W_s\sim\mathcal{N}(0,(t-s)I_d)$.
> 4. Increments over disjoint time intervals are independent.

For a small step $h>0$, a Brownian increment can therefore be sampled as

$$
W_{t+h}-W_t\overset{d}{=}\sqrt{h}\,\varepsilon_t,
\qquad
\varepsilon_t\sim\mathcal{N}(0,I_d).
$$

Brownian paths are continuous but almost surely nowhere differentiable. Consequently, the symbol
$dW_t/dt$ should not be interpreted as an ordinary derivative.

### Stochastic differential equations

> [!note] Definition - Itô stochastic differential equation
> An Itô SDE with scalar diffusion coefficient $\sigma_t\geq 0$ is written
>
> $$
> dX_t=u_t(X_t)\,dt+\sigma_t\,dW_t.
> $$
>
> Its rigorous meaning is the integral equation
>
> $$
> X_t
> =X_0+\int_0^t u_s(X_s)\,ds+\int_0^t\sigma_s\,dW_s,
> $$
>
> where the last term is an Itô stochastic integral. Setting $\sigma_t=0$ recovers an ODE.

The simplest numerical methods are Euler's method for an ODE,

$$
X_{t+h}=X_t+h\,u_t(X_t),
$$

and Euler-Maruyama for an SDE,

$$
X_{t+h}
=X_t+h\,u_t(X_t)+\sqrt{h}\,\sigma_t\varepsilon_t,
\qquad
\varepsilon_t\sim\mathcal{N}(0,I_d).
$$

The factors $h$ and $\sqrt{h}$ are different because Brownian increments have variance proportional
to elapsed time.

### Continuity and Fokker-Planck equations

> [!tip] Theorem - Evolution of marginal densities
> If the deterministic ODE $dX_t=u_t(X_t)\,dt$ has density $p_t$, conservation of probability mass
> is expressed by the **continuity equation**
>
> $$
> \partial_t p_t(x)
> =-\operatorname{div}\!\bigl(p_t(x)u_t(x)\bigr).
> $$
>
> For the SDE $dX_t=u_t(X_t)\,dt+\sigma_t\,dW_t$, the marginal density instead satisfies the
> **Fokker-Planck equation**
>
> $$
> \partial_t p_t(x)
> =-\operatorname{div}\!\bigl(p_t(x)u_t(x)\bigr)
> +\frac{\sigma_t^2}{2}\Delta p_t(x).
> $$

where

$$
\operatorname{div}v
=\sum_{i=1}^d\partial_{x_i}v_i,
\qquad
\Delta f
=\sum_{i=1}^d\partial_{x_i}^2f.
$$

The continuity equation transports probability with the vector field. The Laplacian term accounts
for the spreading of mass caused by Brownian noise.

## Flow and diffusion models

### Flow models

A flow model replaces the unknown target vector field by a neural network
$u_t^\theta:\mathbb{R}^d\to\mathbb{R}^d$ and samples by solving

$$
X_0\sim p_{\mathrm{init}},
\qquad
dX_t=u_t^\theta(X_t)\,dt.
$$

The desired endpoint condition is

$$
X_1\sim p_{\mathrm{data}}.
$$

The ODE is deterministic conditional on $X_0$; all randomness initially comes from the draw
$X_0\sim p_{\mathrm{init}}$.

### Diffusion models

A diffusion model uses an SDE

$$
X_0\sim p_{\mathrm{init}},
\qquad
dX_t=u_t^\theta(X_t)\,dt+\sigma_t\,dW_t,
$$

with a fixed diffusion schedule $\sigma_t$. Even after conditioning on $X_0$, the Brownian motion
makes the trajectory random. Sampling can be approximated by Euler-Maruyama:

$$
X_{t+h}
=X_t+h\,u_t^\theta(X_t)+\sqrt{h}\,\sigma_t\varepsilon_t.
$$

Thus a flow model is the special case $\sigma_t=0$. The main learning problem in both cases is to
construct a vector field whose induced marginals connect $p_{\mathrm{init}}$ to
$p_{\mathrm{data}}$.

## Flow matching

### Conditional and marginal probability paths

> [!note] Definition - Conditional probability path
> A **conditional probability path** $(p_t(\cdot\mid z))_{0\leq t\leq 1}$ interpolates between the
> initial distribution and a single data point $z$:
>
> $$
> p_0(\cdot\mid z)=p_{\mathrm{init}},
> \qquad
> p_1(\cdot\mid z)=\delta_z.
> $$
>
> Here $\delta_z$ is the Dirac measure concentrated at $z$.

Averaging over
$z\sim p_{\mathrm{data}}$ gives the **marginal probability path**

$$
p_t(x)
=\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz.
$$

Equivalently, the marginal can be sampled hierarchically:

$$
Z\sim p_{\mathrm{data}},
\qquad
X_t\mid Z=z\sim p_t(\cdot\mid z)
\quad\Longrightarrow\quad
X_t\sim p_t.
$$

The endpoint conditions imply

$$
p_0=p_{\mathrm{init}},
\qquad
p_1=p_{\mathrm{data}}.
$$

> [!example] Example - Gaussian probability path
> The most important conditional path is
>
> $$
> p_t(\cdot\mid z)
> =\mathcal{N}(\alpha_tz,\beta_t^2I_d),
> $$
>
> where the schedules satisfy
>
> $$
> \alpha_0=0,\quad \alpha_1=1,
> \qquad
> \beta_0=1,\quad \beta_1=0.
> $$
>
> It can be sampled by
>
> $$
> Z\sim p_{\mathrm{data}},
> \qquad
> \varepsilon\sim\mathcal{N}(0,I_d),
> \qquad
> X_t=\alpha_tZ+\beta_t\varepsilon.
> $$
>
> The choice $\alpha_t=t$ and $\beta_t=1-t$ is the Gaussian conditional optimal transport, or
> **CondOT**, path.

### Conditional and marginal vector fields

For each data point $z$, suppose a conditional vector field $u_t^{\mathrm{target}}(x\mid z)$
generates the conditional path:

$$
X_0\sim p_{\mathrm{init}},
\qquad
dX_t=u_t^{\mathrm{target}}(X_t\mid z)\,dt
\quad\Longrightarrow\quad
X_t\sim p_t(\cdot\mid z).
$$

For the Gaussian path, the conditional flow can be chosen as

$$
\psi_t(x_0\mid z)=\alpha_tz+\beta_tx_0.
$$

Differentiating the flow and eliminating $x_0$ yields

$$
u_t^{\mathrm{target}}(x\mid z)
=\left(\dot{\alpha}_t-\frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z
+\frac{\dot{\beta}_t}{\beta_t}x.
$$

> [!tip] Theorem - Marginalization trick
> Although the conditional field sends every trajectory to the fixed endpoint $z$, its posterior
> average defines the **marginal vector field**
>
> $$
> u_t^{\mathrm{target}}(x)
> =\int u_t^{\mathrm{target}}(x\mid z)
> \frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}\,dz.
> $$
>
> Equivalently,
>
> $$
> u_t^{\mathrm{target}}(x)
> =\mathbb{E}\!\left[
> u_t^{\mathrm{target}}(x\mid Z)
> \mid X_t=x
> \right].
> $$
>
> This field generates the marginal path:
>
> $$
> X_0\sim p_{\mathrm{init}},
> \qquad
> dX_t=u_t^{\mathrm{target}}(X_t)\,dt
> \quad\Longrightarrow\quad
> X_t\sim p_t.
> $$
>
> The result follows by substituting the posterior average into the continuity equation. In
> particular, $X_1\sim p_{\mathrm{data}}$.

### Learning the marginal vector field

Using the convention from [[#Expectation notation in ML objectives]], one would ideally fit the
neural vector field by minimizing the marginal flow-matching loss

$$
\mathcal{L}_{\mathrm{FM}}(\theta)
=\mathbb{E}_{t\sim\operatorname{Unif}[0,1],\,X_t\sim p_t}
\left[
\left\lVert
u_t^\theta(X_t)-u_t^{\mathrm{target}}(X_t)
\right\rVert^2
\right].
$$

This objective is intractable because the posterior average
$u_t^{\mathrm{target}}(x)$ is unknown. The conditional target is tractable, so flow matching instead
uses

$$
\mathcal{L}_{\mathrm{CFM}}(\theta)
=\mathbb{E}_{\substack{
t\sim\operatorname{Unif}[0,1],\,
Z\sim p_{\mathrm{data}},\\
X_t\sim p_t(\cdot\mid Z)
}}
\left[
\left\lVert
u_t^\theta(X_t)-u_t^{\mathrm{target}}(X_t\mid Z)
\right\rVert^2
\right].
$$

> [!tip] Theorem - Conditional and marginal flow matching have the same gradient
> Expanding the squared norms and using the posterior-average identity gives
>
> $$
> \mathcal{L}_{\mathrm{FM}}(\theta)
> =\mathcal{L}_{\mathrm{CFM}}(\theta)+C,
> $$
>
> where $C$ is independent of $\theta$. Therefore,
>
> $$
> \nabla_\theta\mathcal{L}_{\mathrm{FM}}(\theta)
> =\nabla_\theta\mathcal{L}_{\mathrm{CFM}}(\theta).
> $$
>
> Thus an explicitly sampled conditional target learns the intractable marginal vector field.

> [!example] Example - Flow matching for a Gaussian path
> Substitute
>
> $$
> X_t=\alpha_tZ+\beta_t\varepsilon,
> \qquad
> \varepsilon\sim\mathcal{N}(0,I_d),
> $$
>
> to obtain
>
> $$
> \mathcal{L}_{\mathrm{CFM}}(\theta)
> =\mathbb{E}_{\substack{
> t\sim\operatorname{Unif}[0,1],\,
> Z\sim p_{\mathrm{data}},\\
> \varepsilon\sim\mathcal{N}(0,I_d)
> }}
> \left[
> \left\lVert
> u_t^\theta(\alpha_tZ+\beta_t\varepsilon)
> -(\dot{\alpha}_tZ+\dot{\beta}_t\varepsilon)
> \right\rVert^2
> \right].
> $$
>
> For the CondOT schedules $\alpha_t=t$ and $\beta_t=1-t$, this becomes
>
> $$
> \mathcal{L}_{\mathrm{CFM}}(\theta)
> =\mathbb{E}_{\substack{
> t\sim\operatorname{Unif}[0,1],\,
> Z\sim p_{\mathrm{data}},\\
> \varepsilon\sim\mathcal{N}(0,I_d)
> }}
> \left[
> \left\lVert
> u_t^\theta(tZ+(1-t)\varepsilon)
> -(Z-\varepsilon)
> \right\rVert^2
> \right].
> $$

Training is simulation-free: one samples $t$, a data point $Z$, and Gaussian noise $\varepsilon$,
then performs an ordinary regression update. ODE simulation is needed only after training, when
samples are generated from $p_{\mathrm{init}}$.

## Score functions and score matching

### Conditional and marginal scores

> [!note] Definition - Score function
> For a differentiable positive density $q$, its **score function** is
>
> $$
> s_q(x)=\nabla_x\log q(x).
> $$
>
> It points in the direction in which the log-density increases most rapidly. For a probability
> path, the conditional and marginal scores are
>
> $$
> s_t(x\mid z)=\nabla_x\log p_t(x\mid z),
> \qquad
> s_t(x)=\nabla_x\log p_t(x).
> $$

> [!abstract] Proposition - Score marginalization
> Differentiating the marginal density under the integral gives
>
> $$
> s_t(x)
> =\int s_t(x\mid z)
> \frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}\,dz
> =\mathbb{E}[s_t(x\mid Z)\mid X_t=x].
> $$
>
> Thus conditional-to-marginal averaging works for scores exactly as it does for vector fields.

> [!example] Example - Score of the Gaussian path
> For the Gaussian path,
>
> $$
> s_t(x\mid z)
> =-\frac{x-\alpha_tz}{\beta_t^2}.
> $$
>
> Since $x=\alpha_tz+\beta_t\varepsilon$, the same expression is
>
> $$
> s_t(x\mid z)=-\frac{\varepsilon}{\beta_t}.
> $$

### Scores, vector fields, and denoisers

> [!abstract] Proposition - Gaussian score-vector-field conversion
> For a Gaussian path, the conditional vector field and score are affine reparameterizations:
>
> $$
> u_t^{\mathrm{target}}(x\mid z)
> =a_t\,s_t(x\mid z)+b_t x,
> $$
>
> where, for interior times at which the expressions are defined,
>
> $$
> a_t
> =\beta_t^2\frac{\dot{\alpha}_t}{\alpha_t}
> -\dot{\beta}_t\beta_t,
> \qquad
> b_t=\frac{\dot{\alpha}_t}{\alpha_t}.
> $$
>
> Posterior averaging gives the marginal identity
>
> $$
> u_t^{\mathrm{target}}(x)
> =a_t\,s_t(x)+b_tx.
> $$

> [!note] Definition - Denoiser
> Another equivalent parameterization is the posterior mean
>
> $$
> D_t(x)=\mathbb{E}[Z\mid X_t=x].
> $$
>
> For the Gaussian path,
>
> $$
> s_t(x)
> =\frac{\alpha_tD_t(x)-x}{\beta_t^2}.
> $$

Learning the marginal vector field, marginal score, or posterior mean therefore identifies the
same conditional information, expressed in different coordinates.

### Adding stochasticity without changing the marginals

> [!tip] Theorem - SDE extension trick
> Suppose the ODE
>
> $$
> dX_t=u_t^{\mathrm{target}}(X_t)\,dt
> $$
>
> follows the probability path $p_t$. For any diffusion schedule $\sigma_t\geq 0$, the SDE
>
> $$
> dX_t
> =\left[
> u_t^{\mathrm{target}}(X_t)
> +\frac{\sigma_t^2}{2}s_t(X_t)
> \right]dt
> +\sigma_t\,dW_t
> $$
>
> has the same marginal laws $X_t\sim p_t$. Indeed,
>
> $$
> \operatorname{div}\!\left(
> p_t\frac{\sigma_t^2}{2}\nabla\log p_t
> \right)
> =\frac{\sigma_t^2}{2}\Delta p_t,
> $$
>
> so the score drift cancels the diffusion term in the Fokker-Planck equation. Individual
> trajectories become stochastic while the prescribed marginal path remains unchanged.

For a Gaussian path, the score-vector-field conversion means that one learned network is enough
to parameterize both the deterministic ODE sampler and a family of SDE samplers.

### Score matching and denoising score matching

Let $s_t^\theta(x)$ be a neural score model. With the same sampling-subscript convention, the ideal
score-matching objective is

$$
\mathcal{L}_{\mathrm{SM}}(\theta)
=\mathbb{E}_{\substack{
t\sim\operatorname{Unif}[0,1],\\
X_t\sim p_t
}}
\left[
\left\lVert
s_t^\theta(X_t)-s_t(X_t)
\right\rVert^2
\right].
$$

The marginal score $s_t(x)$ is unavailable, so we instead regress against the conditional score:

$$
\mathcal{L}_{\mathrm{DSM}}(\theta)
=\mathbb{E}_{\substack{
t\sim\operatorname{Unif}[0,1],\,
Z\sim p_{\mathrm{data}},\\
X_t\sim p_t(\cdot\mid Z)
}}
\left[
\left\lVert
s_t^\theta(X_t)-s_t(X_t\mid Z)
\right\rVert^2
\right].
$$

> [!tip] Theorem - Denoising score matching
> As with conditional flow matching,
>
> $$
> \mathcal{L}_{\mathrm{SM}}(\theta)
> =\mathcal{L}_{\mathrm{DSM}}(\theta)+C,
> $$
>
> where $C$ is independent of $\theta$. Consequently,
>
> $$
> \nabla_\theta\mathcal{L}_{\mathrm{SM}}(\theta)
> =\nabla_\theta\mathcal{L}_{\mathrm{DSM}}(\theta).
> $$
>
> Regressing against the tractable conditional score therefore learns the marginal score.

This objective is called **denoising score matching** because $X_t$ is a corrupted version of the
clean sample $Z$.

> [!example] Example - Gaussian denoising score matching
> For the Gaussian path,
>
> $$
> X_t=\alpha_tZ+\beta_t\varepsilon,
> \qquad
> s_t(X_t\mid Z)=-\frac{\varepsilon}{\beta_t},
> $$
>
> and hence
>
> $$
> \mathcal{L}_{\mathrm{DSM}}(\theta)
> =\mathbb{E}_{\substack{
> t\sim\operatorname{Unif}[0,1],\,
> Z\sim p_{\mathrm{data}},\\
> \varepsilon\sim\mathcal{N}(0,I_d)
> }}
> \left[
> \left\lVert
> s_t^\theta(\alpha_tZ+\beta_t\varepsilon)
> +\frac{\varepsilon}{\beta_t}
> \right\rVert^2
> \right].
> $$
>
> Define a noise-prediction network by
>
> $$
> \varepsilon_t^\theta(x)=-\beta_t s_t^\theta(x).
> $$
>
> The weighted score objective becomes
>
> $$
> \mathcal{L}_{\mathrm{DSM}}(\theta)
> =\mathbb{E}_{\substack{
> t\sim\operatorname{Unif}[0,1],\,
> Z\sim p_{\mathrm{data}},\\
> \varepsilon\sim\mathcal{N}(0,I_d)
> }}
> \left[
> \frac{1}{\beta_t^2}
> \left\lVert
> \varepsilon_t^\theta(\alpha_tZ+\beta_t\varepsilon)-\varepsilon
> \right\rVert^2
> \right].
> $$

> [!warning] Remark - Simplified noise-prediction loss
> A commonly used objective drops the time-dependent prefactor:
>
> $$
> \mathcal{L}_{\mathrm{noise}}(\theta)
> =\mathbb{E}_{\substack{
> t\sim\operatorname{Unif}[0,1],\,
> Z\sim p_{\mathrm{data}},\\
> \varepsilon\sim\mathcal{N}(0,I_d)
> }}
> \left[
> \left\lVert
> \varepsilon_t^\theta(\alpha_tZ+\beta_t\varepsilon)-\varepsilon
> \right\rVert^2
> \right].
> $$
>
> Dropping $1/\beta_t^2$ changes the relative weighting of noise levels even though the pointwise
> regression target remains $\varepsilon$.

Thus score matching can be implemented by the following regression procedure:

1. Sample a data point $Z\sim p_{\mathrm{data}}$.
2. Sample $t\sim\operatorname{Unif}[0,1]$ and
   $\varepsilon\sim\mathcal{N}(0,I_d)$.
3. Form the noisy input $X_t=\alpha_tZ+\beta_t\varepsilon$.
4. Predict either the conditional score $-\varepsilon/\beta_t$, the noise $\varepsilon$, or an
   equivalent denoised target.
5. Update the network parameters with the chosen squared-error loss.

After training, the learned score can be converted into the marginal vector field for ODE sampling
or inserted into the score-corrected SDE for stochastic sampling.

## Source

These notes follow Sections 1–4 and Appendix A of Peter Holderrieth and Ezra Erives,
[_An Introduction to Flow Matching and Diffusion Models_](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf),
MIT 6.S184 lecture notes (2026; [course page](https://diffusion.csail.mit.edu/2026/),
[arXiv:2506.02070](https://arxiv.org/abs/2506.02070)), with additional measure-theoretic
clarification and an explicit probability/SDE preliminaries section.
