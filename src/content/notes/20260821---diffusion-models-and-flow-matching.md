---
title: '扩散模型与流匹配'
description: '扩散模型与流匹配的数学基础：从概率、路径空间测度到 ODE、SDE 与分数匹配。'
date: 2026-08-21
tags:
  - diffusion-models
  - flow-matching
  - generative-modeling
  - probability
language: 'zh-CN'
type: note
status: ready
draft: false
---

> **翻译说明**
>
> 本文最初以英语写成；中文镜像版本由 Codex 辅助翻译，并以英文原稿为准。

# 扩散模型与流匹配

生成建模把待生成的对象视为随机向量。图像、视频或分子构型通常都可以展平为向量
$z\in\mathbb{R}^d$。数据由一个未知的概率律 $p_{\mathrm{data}}$ 建模，而生成就是构造一个
能够从该概率律中返回近似样本的算法。

流匹配和扩散模型通过把样本从一个简单的初始概率律 $p_{\mathrm{init}}$（通常是标准高斯分布）
输运到 $p_{\mathrm{data}}$ 来完成这一任务。流模型使用常微分方程（ODE）；扩散模型则允许使用
随机微分方程（SDE）。

在整篇笔记中，时间从 $t=0$ 的噪声走向 $t=1$ 的数据：

$$
X_0\sim p_{\mathrm{init}},
\qquad
X_1\sim p_{\mathrm{data}}.
$$

## 目录

- [[#将生成建模理解为采样]]
- [[#概率论预备知识]]
- [[#转移核与路径空间测度]]
- [[#机器学习目标中的期望记号]]
- [[#ODE 与 SDE 预备知识]]
- [[#流模型与扩散模型]]
- [[#流匹配]]
- [[#分数函数与分数匹配]]

## 将生成建模理解为采样

数据集被建模为一组独立样本：

$$
z_1,\ldots,z_N\overset{\mathrm{i.i.d.}}{\sim}p_{\mathrm{data}}.
$$

$p_{\mathrm{data}}$ 不必具有已知的解析形式。实践中，经验数据集通常是我们直接接触它的唯一途径。

> [!note] 定义——生成模型
> **生成模型**是一种学习得到的采样过程，其输出分布近似于 $p_{\mathrm{data}}$。

如果生成过程以类别标签或文本提示等辅助信息 $y$ 为条件，目标就变为条件概率律

$$
z\sim p_{\mathrm{data}}(\cdot\mid y).
$$

无条件问题是其中的数学核心。条件信息改变目标概率律，但不会改变下文建立的基本输运思想。

## 概率论预备知识

> [!info] 规范的概率论基础
> 本节改编自 [[Maths for RL#Probability spaces, random variables, and expectation]] 和
> [[Maths for RL#Conditional probability and conditional expectation]]。这里有意省略了转移核、
> 轨迹概率律、特定应用的采样记号以及控制过程相关内容。

### 概率空间、随机变量与概率律

> [!note] 定义——概率空间
> 概率空间是一个三元组 $(\Omega,\mathcal{F},\mathbb{P})$：
>
> - $\Omega$ 是所有可能结果的集合。
> - $\mathcal{F}$ 是由可测事件组成的 $\sigma$-代数。
> - $\mathbb{P}$ 是定义在这些事件上的概率测度。

随机变量是一个可测映射 $X:\Omega\to\mathcal{X}$。它的分布，也称为**概率律**，是推前测度

$$
\mu_X(A)=\mathbb{P}(X\in A),
\qquad
A\subseteq\mathcal{X}.
$$

对于可积函数 $f$，期望就是关于概率测度的积分：

$$
\mathbb{E}[f(X)]
=\int_\Omega f(X(\omega))\,\mathbb{P}(d\omega)
=\int_\mathcal{X}f(x)\,\mu_X(dx).
$$

如果 $X$ 的支撑集有限或可数，同一个积分就写成求和：

$$
\mathbb{E}[f(X)]
=\sum_{x\in\mathcal{X}}f(x)\,\mathbb{P}(X=x).
$$

概率质量函数或概率密度，是概率测度相对于计数测度或勒贝格测度的一种表示。并非每个概率测度
都有密度。若一个命题仅在概率为零的集合上不成立，则称它**几乎必然**成立，简写为 a.s.。

### 概率密度

如果取值于 $\mathbb{R}^d$ 的随机变量 $X$ 具有密度 $p_X$，那么

$$
\mathbb{P}(X\in A)
=\int_A p_X(x)\,dx,
\qquad
\int_{\mathbb{R}^d}p_X(x)\,dx=1.
$$

### 多元高斯分布

> [!note] 定义——多元高斯分布
> 如果随机向量 $X\in\mathbb{R}^d$ 的均值为 $\mu\in\mathbb{R}^d$、协方差矩阵为
> $\Sigma\in\mathbb{R}^{d\times d}$，并且
>
> $$
> X\sim\mathcal{N}(\mu,\Sigma),
> $$
>
> 则称 $X$ 服从多元高斯分布。它的前两个矩为
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
> 协方差矩阵是对称半正定矩阵。

如果 $\Sigma$ 正定，该分布具有密度

$$
\mathcal{N}(x;\mu,\Sigma)
=\frac{1}{(2\pi)^{d/2}\det(\Sigma)^{1/2}}
\exp\!\left(
-\frac{1}{2}
(x-\mu)^\mathsf{T}\Sigma^{-1}(x-\mu)
\right).
$$

如果 $\Sigma$ 奇异，高斯概率律仍然有良好定义，但其支撑位于一个低维仿射子空间上，因而不存在
相对于 $d$ 维勒贝格测度的密度。

#### 仿射构造与采样

> [!abstract] 命题——高斯分布的仿射像
> 如果 $\varepsilon\sim\mathcal{N}(0,I_m)$，且
> $X=\mu+A\varepsilon$，其中 $A\in\mathbb{R}^{d\times m}$，那么
>
> $$
> X\sim\mathcal{N}(\mu,AA^\mathsf{T}).
> $$
>
> 更一般地，如果 $X\sim\mathcal{N}(\mu,\Sigma)$，那么
>
> $$
> BX+b
> \sim
> \mathcal{N}\!\left(
> B\mu+b,\,
> B\Sigma B^\mathsf{T}
> \right).
> $$

要从 $\mathcal{N}(\mu,\Sigma)$ 中采样，可以选择满足 $LL^\mathsf{T}=\Sigma$ 的矩阵 $L$；
例如，当 $\Sigma$ 正定时可取 Cholesky 分解因子。然后令

$$
X=\mu+L\varepsilon,
\qquad
\varepsilon\sim\mathcal{N}(0,I_d).
$$

如果 $X$ 和 $Y$ 是相互独立的高斯向量，那么

$$
X+Y
\sim
\mathcal{N}(
\mathbb{E}[X]+\mathbb{E}[Y],\,
\operatorname{Cov}(X)+\operatorname{Cov}(Y)
).
$$

#### 各向同性与对角高斯分布

> [!example] 示例——各向同性高斯分布
> 如果 $\Sigma=\sigma^2I_d$，所有坐标的方差都为 $\sigma^2$，并且彼此独立。密度变为
>
> $$
> \mathcal{N}(x;\mu,\sigma^2I_d)
> =(2\pi\sigma^2)^{-d/2}
> \exp\!\left(
> -\frac{\lVert x-\mu\rVert^2}{2\sigma^2}
> \right).
> $$
>
> 标准高斯分布 $\mathcal{N}(0,I_d)$ 常被用作 $p_{\mathrm{init}}$，因为它容易采样，
> 并且在线性变换下具有稳定的性质。

对角协方差矩阵表示各坐标方差可以不同，但坐标之间没有相关性：

$$
\Sigma
=\operatorname{diag}(
\sigma_1^2,\ldots,\sigma_d^2
).
$$

对于 Euler–Maruyama 方法中使用的各向同性布朗增量，

$$
\sqrt{h}\,\sigma_t\varepsilon
\sim
\mathcal{N}(0,h\sigma_t^2I_d).
$$

这正是实现中把标准高斯噪声乘以 $\sqrt{h}\,\sigma_t$ 的原因。

#### 高斯分布的分数

> [!abstract] 命题——高斯分数
> 对于非退化高斯密度，
>
> $$
> \nabla_x\log\mathcal{N}(x;\mu,\Sigma)
> =-\Sigma^{-1}(x-\mu).
> $$
>
> 在各向同性情形下，
>
> $$
> \nabla_x\log\mathcal{N}(x;\mu,\sigma^2I_d)
> =-\frac{x-\mu}{\sigma^2}.
> $$
>
> 这个公式稍后会成为高斯概率路径所使用的条件分数。

### 条件概率与条件期望

> [!note] 定义——条件期望
> 设 $Y$ 可积，$\mathcal{G}\subseteq\mathcal{F}$ 表示可用信息。如果随机变量
> $U=\mathbb{E}[Y\mid\mathcal{G}]$ 是 $\mathcal{G}$-可测的，并且满足
>
> $$
> \int_G U\,d\mathbb{P}
> =\int_G Y\,d\mathbb{P}
> \qquad
> \text{对每个 }G\in\mathcal{G},
> $$
>
> 则称 $U$ 是一个条件期望。它在几乎处处相等的意义下唯一。

在这里使用的标准 Borel 空间上，条件概率律可以用正则条件分布 $K_{Y\mid X}(x,dy)$ 表示。
对于合适的 $g$，

$$
\mathbb{E}[g(Y)\mid X]
=\int_\mathcal{Y}g(y)K_{Y\mid X}(X,dy).
$$

当 $Y$ 是离散变量时，它变为

$$
\mathbb{E}[g(Y)\mid X=x]
=\sum_{y\in\mathcal{Y}}
g(y)\mathbb{P}(Y=y\mid X=x).
$$

> [!abstract]- 条件期望的运算规则
> 最常用的规则包括：
>
> - **线性性：**
>
>   $$
>   \mathbb{E}[aY+bZ\mid\mathcal{G}]
>   =a\mathbb{E}[Y\mid\mathcal{G}]
>   +b\mathbb{E}[Z\mid\mathcal{G}].
>   $$
>
> - **提出已知量：**如果 $Z$ 是 $\mathcal{G}$-可测的，并且各项可积，那么
>
>   $$
>   \mathbb{E}[ZY\mid\mathcal{G}]
>   =Z\mathbb{E}[Y\mid\mathcal{G}].
>   $$
>
> - **塔式法则：**如果 $\mathcal{H}\subseteq\mathcal{G}$，那么
>
>   $$
>   \mathbb{E}[
>   \mathbb{E}[Y\mid\mathcal{G}]
>   \mid\mathcal{H}]
>   =\mathbb{E}[Y\mid\mathcal{H}].
>   $$
>
> - **全期望公式：**
>
>   $$
>   \mathbb{E}[
>   \mathbb{E}[Y\mid\mathcal{G}]
>   ]
>   =\mathbb{E}[Y].
>   $$
>
> - **独立性：**如果 $Y$ 与 $\mathcal{G}$ 独立，那么
>
>   $$
>   \mathbb{E}[Y\mid\mathcal{G}]
>   =\mathbb{E}[Y]
>   \qquad\text{a.s.}
>   $$

### 全概率公式与全期望公式

> [!abstract]- 命题——关于条件变量的分解
> 如果 $X$ 是离散变量，那么
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
> 对一般的条件变量，求和被关于其概率律 $\mu_X$ 的积分取代：
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
> 如果相关密度都存在，连续情形的全概率公式为
>
> $$
> p_Y(y)
> =\int p_{Y\mid X}(y\mid x)p_X(x)\,dx.
> $$

### Bayes 公式与后验平均

对于具有联合密度的随机变量 $X$ 和 $Z$，Bayes 公式给出

$$
p_{Z\mid X}(z\mid x)
=\frac{p_{X\mid Z}(x\mid z)p_Z(z)}{p_X(x)}
$$

其中要求 $p_X(x)>0$。条件期望就是关于该后验分布的平均：

$$
\mathbb{E}[g(Z)\mid X=x]
=\int g(z)p_{Z\mid X}(z\mid x)\,dz.
$$

这种后验平均形式，是流匹配和分数匹配中边缘化恒等式的基础。

> [!tip] 定理——作为最小二乘回归的条件期望
> 在所有可测函数 $f(X)$ 中，使
>
> $$
> \mathbb{E}\!\left[
> \lVert Y-f(X)\rVert^2
> \right]
> $$
>
> 最小的函数是
>
> $$
> f(X)=\mathbb{E}[Y\mid X]
> \qquad\text{a.s.}
> $$
>
> 因此，对一个随机条件目标做回归，可以恢复它的后验均值。

## 转移核与路径空间测度

> [!info] 规范的核理论基础
> 本节改编自 [[Maths for RL#Transition kernels and measures on paths]] 中与具体框架无关的部分，
> 省略了对策略、环境和受控轨迹的专门讨论。

### 随机核

> [!note] 定义——随机核
> 设 $(\mathcal{X},\mathscr{X})$ 与 $(\mathcal{Y},\mathscr{Y})$ 是可测空间。从
> $\mathcal{X}$ 到 $\mathcal{Y}$ 的**随机核** $K$，为每个 $x\in\mathcal{X}$ 指定
> $\mathcal{Y}$ 上的一个概率测度 $K(x,\cdot)$，并满足：
>
> 1. 对每个固定的 $x$，映射 $B\mapsto K(x,B)$ 是概率测度。
> 2. 对每个固定的 $B\in\mathscr{Y}$，映射 $x\mapsto K(x,B)$ 是可测的。

我们常把它写成 $K(x,dy)$。将可测函数 $g$ 关于这个条件概率律积分，得到

$$
\int_\mathcal{Y}g(y)K(x,dy).
$$

在有限空间上，定义 $K(x,y)=K(x,\{y\})$，则

$$
\sum_{y\in\mathcal{Y}}K(x,y)=1,
\qquad
\int_\mathcal{Y}g(y)K(x,dy)
=\sum_{y\in\mathcal{Y}}g(y)K(x,y).
$$

Markov 转移核是一种随机核：其输入表示当前状态，输出则是下一个状态的概率分布。

### 从核构造联合概率律

一个初始分布和一列核共同决定所有坐标的联合概率律。设 $\mu_0$ 是 $\mathcal{X}_0$ 上的概率测度，
并令

$$
K_k(x_0,\ldots,x_k,dx_{k+1})
$$

根据历史选择下一个坐标。

> [!tip]- 定理——Ionescu–Tulcea 构造
> 初始概率律 $\mu_0$ 与核 $(K_k)_{k\geq 0}$ 唯一确定乘积空间
>
> $$
> \Omega=\prod_{k=0}^{\infty}\mathcal{X}_k
> $$
>
> 上的一个概率测度 $\mathbb{P}$。它在柱事件上的概率由迭代积分给出：
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

对于有限前缀的可测函数 $F$，

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

简洁写法

$$
\mathbb{P}(dx_0,\ldots,dx_n)
=\mu_0(dx_0)
K_0(x_0,dx_1)\cdots
K_{n-1}(x_0,\ldots,x_{n-1},dx_n)
$$

是上述迭代构造的缩写，而不是普通数值的乘法。

如果过程是 Markov 过程，下一个坐标的核只依赖当前坐标：

$$
K_k(x_0,\ldots,x_k,dx_{k+1})
=K_k(x_k,dx_{k+1}).
$$

### ODE 与 SDE 的路径概率律

对于从时刻 $s$ 到 $t$ 的确定性流，转移核是 Dirac 测度：

$$
K_{s,t}(x,dy)
=\delta_{\psi_{s,t}(x)}(dy).
$$

对于 Markov SDE，转移核为

$$
K_{s,t}(x,B)
=\mathbb{P}(X_t\in B\mid X_s=x).
$$

> [!example] 示例——Euler–Maruyama 转移
> 对离散化
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
> 单步转移核是高斯分布：
>
> $$
> K_k(x,\cdot)
> =\mathcal{N}\!\left(
> x+h\,u_{t_k}(x),
> h\sigma_{t_k}^2I_d
> \right).
> $$

在时间网格上，初始概率律与这些核会在 $(X_{t_0},\ldots,X_{t_n})$ 上诱导一个联合分布。
在连续时间中，具有连续轨迹的 ODE 或 SDE 会在

$$
C([0,1],\mathbb{R}^d)
$$

上诱导一个**路径概率律**；这里是从时间到状态空间的连续函数所组成的空间。类似
$\mathbb{E}_{\mathbb{P}}[F((X_t)_{0\leq t\leq 1})]$ 的表达式，表示将路径泛函 $F$
关于该概率律积分。

## 机器学习目标中的期望记号

> [!info] 与规范记号章节的关系
> 本节将 [[Maths for RL#Probability notation versus ML/RL notation]] 中的一般讨论，
> 专门应用于匹配目标和普通监督学习。

### 采样下标指定一个联合概率律

机器学习记号常通过列出随机输入的采样方式来描述期望。例如，

$$
\mathbb{E}_{\substack{
t\sim\operatorname{Unif}[0,1],\,
Z\sim p_{\mathrm{data}},\\
X_t\sim p_t(\cdot\mid Z)
}}
[\ell(t,Z,X_t)]
$$

表示：

1. 从 $[0,1]$ 上的均匀分布采样 $t$；
2. 从数据概率律中采样 $Z$；
3. 给定 $(t,Z)$，从 $p_t(\cdot\mid Z)$ 中采样 $X_t$；
4. 计算 $\ell(t,Z,X_t)$ 并取平均。

用 $\mu_{\mathrm{data}}$ 表示非正式写成 $p_{\mathrm{data}}$ 的概率测度。这些采样指令在
$[0,1]\times\mathcal{Z}\times\mathcal{X}$ 上定义联合测度

$$
\nu(dt,dz,dx)
=dt\,\mu_{\mathrm{data}}(dz)\,p_t(dx\mid z).
$$

相应期望就是迭代积分

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

> [!note] 如何理解多个下标
> 下标是对同一个联合概率律的紧凑描述。$X_t\sim p_t(\cdot\mid Z)$ 这样的条件语句指定核，
> 独立采样的量则贡献乘积因子。这个记号并不表示若干个互不相关的期望。

### 示例：条件流匹配损失

对于高斯概率路径，损失

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

是关于联合概率律

$$
dt\,\mu_{\mathrm{data}}(dz)\,\gamma(d\varepsilon),
\qquad
\gamma=\mathcal{N}(0,I_d)
$$

的积分的缩写。等价地，它也可以理解为嵌套期望：

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

其中展示出的采样概率律决定每个期望的含义。

### 总体期望与经验均值

**总体目标**关于未知的数据概率律积分：

$$
\mathcal{L}(\theta)
=\mathbb{E}_{Z\sim p_{\mathrm{data}}}
[f_\theta(Z)].
$$

给定数据集 $\mathcal{D}=\{z_1,\ldots,z_N\}$，它的经验测度为

$$
\widehat{\mu}_N
=\frac{1}{N}\sum_{i=1}^N\delta_{z_i}.
$$

关于经验测度的期望恰好就是样本均值：

$$
\mathbb{E}_{Z\sim\widehat{\mu}_N}[f_\theta(Z)]
=\frac{1}{N}\sum_{i=1}^N f_\theta(z_i).
$$

如果每个数据点都与新的辅助随机量 $\xi$ 组合，例如时间和高斯噪声，那么完整的经验目标是

$$
\widehat{\mathcal{L}}_N(\theta)
=\frac{1}{N}\sum_{i=1}^N
\mathbb{E}_{\xi}
[\ell_\theta(z_i,\xi)].
$$

训练通常用小批量 Monte Carlo 估计量替代它：

$$
\widehat{\mathcal{L}}_{\mathcal{B}}(\theta)
=\frac{1}{|\mathcal{B}|}
\sum_{i\in\mathcal{B}}
\ell_\theta(z_i,\xi_i),
$$

其中 $\xi_i$ 在当前更新中重新采样。

> [!warning] 总体、经验与小批量记号常被混用
> 作者有时写 $\mathbb{E}_{Z\sim p_{\mathrm{data}}}$，而实现实际上对有限数据集或小批量求平均。
> 这些概念相关，但并不相同：
>
> - **总体期望**对未知的真实概率律求平均；
> - **经验期望**对全部观测样本求平均；
> - **小批量均值**是经验目标的随机估计量；
> - 如果数据集本身由 i.i.d. 样本组成，经验目标也在估计总体目标。

## ODE 与 SDE 预备知识

### 随时间变化的概率律与确定性流

**随机过程** $(X_t)_{0\leq t\leq 1}$ 是一族以时间为索引的随机变量。它在时刻 $t$ 的概率律
记为 $p_t$：

$$
X_t\sim p_t.
$$

族 $(p_t)_{0\leq t\leq 1}$ 只记录每个时刻的边缘分布，本身并不说明 $X_s$ 和 $X_t$
在不同时间之间如何联合耦合。

随时间变化的向量场是一个函数

$$
u:\mathbb{R}^d\times[0,1]\to\mathbb{R}^d,
\qquad
(x,t)\mapsto u_t(x).
$$

$u_t$ 生成的 ODE 为

$$
\frac{d}{dt}X_t=u_t(X_t),
\qquad
X_0=x_0.
$$

在标准正则条件下，它的解定义一个流映射 $\psi_t:\mathbb{R}^d\to\mathbb{R}^d$，使得
$X_t=\psi_t(X_0)$。如果 $X_0$ 是随机变量，那么 $X_t$ 的概率律就是推前测度

$$
p_t=(\psi_t)_{\#}p_0.
$$

### 布朗运动

> [!note] 定义——布朗运动
> $d$ 维布朗运动 $(W_t)_{t\geq 0}$ 是满足下列条件的随机过程：
>
> 1. $W_0=0$ 几乎必然成立。
> 2. 其路径几乎必然连续。
> 3. 对 $0\leq s<t$，有 $W_t-W_s\sim\mathcal{N}(0,(t-s)I_d)$。
> 4. 不相交时间区间上的增量彼此独立。

因此，对于一个小步长 $h>0$，布朗增量可以采样为

$$
W_{t+h}-W_t\overset{d}{=}\sqrt{h}\,\varepsilon_t,
\qquad
\varepsilon_t\sim\mathcal{N}(0,I_d).
$$

布朗路径连续，但几乎必然处处不可微。因此，符号 $dW_t/dt$ 不能理解为普通导数。

### 随机微分方程

> [!note] 定义——Itô 随机微分方程
> 具有标量扩散系数 $\sigma_t\geq 0$ 的 Itô SDE 写作
>
> $$
> dX_t=u_t(X_t)\,dt+\sigma_t\,dW_t.
> $$
>
> 它的严格含义是积分方程
>
> $$
> X_t
> =X_0+\int_0^t u_s(X_s)\,ds+\int_0^t\sigma_s\,dW_s,
> $$
>
> 其中最后一项是 Itô 随机积分。令 $\sigma_t=0$ 就恢复为 ODE。

最简单的数值方法分别是用于 ODE 的 Euler 方法

$$
X_{t+h}=X_t+h\,u_t(X_t),
$$

以及用于 SDE 的 Euler–Maruyama 方法

$$
X_{t+h}
=X_t+h\,u_t(X_t)+\sqrt{h}\,\sigma_t\varepsilon_t,
\qquad
\varepsilon_t\sim\mathcal{N}(0,I_d).
$$

$h$ 与 $\sqrt{h}$ 两个因子不同，是因为布朗增量的方差与经过的时间成正比。

### 连续性方程与 Fokker–Planck 方程

> [!tip] 定理——边缘密度的演化
> 如果确定性 ODE $dX_t=u_t(X_t)\,dt$ 具有密度 $p_t$，概率质量守恒由**连续性方程**表示：
>
> $$
> \partial_t p_t(x)
> =-\operatorname{div}\!\bigl(p_t(x)u_t(x)\bigr).
> $$
>
> 对于 SDE $dX_t=u_t(X_t)\,dt+\sigma_t\,dW_t$，其边缘密度则满足
> **Fokker–Planck 方程**：
>
> $$
> \partial_t p_t(x)
> =-\operatorname{div}\!\bigl(p_t(x)u_t(x)\bigr)
> +\frac{\sigma_t^2}{2}\Delta p_t(x).
> $$

其中

$$
\operatorname{div}v
=\sum_{i=1}^d\partial_{x_i}v_i,
\qquad
\Delta f
=\sum_{i=1}^d\partial_{x_i}^2f.
$$

连续性方程随向量场输运概率；Laplacian 项则描述布朗噪声造成的概率质量扩散。

## 流模型与扩散模型

### 流模型

流模型用神经网络 $u_t^\theta:\mathbb{R}^d\to\mathbb{R}^d$ 替代未知的目标向量场，
并通过求解下式进行采样：

$$
X_0\sim p_{\mathrm{init}},
\qquad
dX_t=u_t^\theta(X_t)\,dt.
$$

期望的终点条件是

$$
X_1\sim p_{\mathrm{data}}.
$$

给定 $X_0$ 后，ODE 是确定性的；全部随机性最初都来自 $X_0\sim p_{\mathrm{init}}$ 的抽样。

### 扩散模型

扩散模型使用 SDE

$$
X_0\sim p_{\mathrm{init}},
\qquad
dX_t=u_t^\theta(X_t)\,dt+\sigma_t\,dW_t,
$$

其中扩散调度 $\sigma_t$ 是固定的。即使给定 $X_0$，布朗运动仍使轨迹保持随机。采样可以用
Euler–Maruyama 方法近似：

$$
X_{t+h}
=X_t+h\,u_t^\theta(X_t)+\sqrt{h}\,\sigma_t\varepsilon_t.
$$

因此，流模型是 $\sigma_t=0$ 的特殊情形。两类模型的核心学习问题都是构造一个向量场，使其诱导的
边缘分布把 $p_{\mathrm{init}}$ 连接到 $p_{\mathrm{data}}$。

## 流匹配

### 条件概率路径与边缘概率路径

> [!note] 定义——条件概率路径
> **条件概率路径** $(p_t(\cdot\mid z))_{0\leq t\leq 1}$ 在初始分布与单个数据点 $z$
> 之间插值：
>
> $$
> p_0(\cdot\mid z)=p_{\mathrm{init}},
> \qquad
> p_1(\cdot\mid z)=\delta_z.
> $$
>
> 这里 $\delta_z$ 是集中在 $z$ 处的 Dirac 测度。

对 $z\sim p_{\mathrm{data}}$ 求平均，得到**边缘概率路径**

$$
p_t(x)
=\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz.
$$

等价地，可以用分层采样得到边缘分布：

$$
Z\sim p_{\mathrm{data}},
\qquad
X_t\mid Z=z\sim p_t(\cdot\mid z)
\quad\Longrightarrow\quad
X_t\sim p_t.
$$

端点条件意味着

$$
p_0=p_{\mathrm{init}},
\qquad
p_1=p_{\mathrm{data}}.
$$

> [!example] 示例——高斯概率路径
> 最重要的条件路径是
>
> $$
> p_t(\cdot\mid z)
> =\mathcal{N}(\alpha_tz,\beta_t^2I_d),
> $$
>
> 其中调度满足
>
> $$
> \alpha_0=0,\quad \alpha_1=1,
> \qquad
> \beta_0=1,\quad \beta_1=0.
> $$
>
> 它可以通过下式采样：
>
> $$
> Z\sim p_{\mathrm{data}},
> \qquad
> \varepsilon\sim\mathcal{N}(0,I_d),
> \qquad
> X_t=\alpha_tZ+\beta_t\varepsilon.
> $$
>
> 选择 $\alpha_t=t$ 和 $\beta_t=1-t$，得到高斯条件最优输运路径，简称 **CondOT** 路径。

### 条件向量场与边缘向量场

对每个数据点 $z$，假设条件向量场 $u_t^{\mathrm{target}}(x\mid z)$ 生成条件路径：

$$
X_0\sim p_{\mathrm{init}},
\qquad
dX_t=u_t^{\mathrm{target}}(X_t\mid z)\,dt
\quad\Longrightarrow\quad
X_t\sim p_t(\cdot\mid z).
$$

对于高斯路径，可以选择条件流

$$
\psi_t(x_0\mid z)=\alpha_tz+\beta_tx_0.
$$

对这个流求导并消去 $x_0$，得到

$$
u_t^{\mathrm{target}}(x\mid z)
=\left(\dot{\alpha}_t-\frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z
+\frac{\dot{\beta}_t}{\beta_t}x.
$$

> [!tip] 定理——边缘化技巧
> 尽管条件向量场把每条轨迹都送到固定终点 $z$，它的后验平均定义了**边缘向量场**
>
> $$
> u_t^{\mathrm{target}}(x)
> =\int u_t^{\mathrm{target}}(x\mid z)
> \frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}\,dz.
> $$
>
> 等价地，
>
> $$
> u_t^{\mathrm{target}}(x)
> =\mathbb{E}\!\left[
> u_t^{\mathrm{target}}(x\mid Z)
> \mid X_t=x
> \right].
> $$
>
> 这个向量场生成边缘路径：
>
> $$
> X_0\sim p_{\mathrm{init}},
> \qquad
> dX_t=u_t^{\mathrm{target}}(X_t)\,dt
> \quad\Longrightarrow\quad
> X_t\sim p_t.
> $$
>
> 把后验平均代入连续性方程即可得到这一结果。特别地，$X_1\sim p_{\mathrm{data}}$。

### 学习边缘向量场

按照 [[#机器学习目标中的期望记号]] 中的约定，理想情况下可以通过最小化边缘流匹配损失来拟合
神经向量场：

$$
\mathcal{L}_{\mathrm{FM}}(\theta)
=\mathbb{E}_{t\sim\operatorname{Unif}[0,1],\,X_t\sim p_t}
\left[
\left\lVert
u_t^\theta(X_t)-u_t^{\mathrm{target}}(X_t)
\right\rVert^2
\right].
$$

这个目标不可直接计算，因为后验平均 $u_t^{\mathrm{target}}(x)$ 未知。条件目标却可以计算，
因此流匹配改用

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

> [!tip] 定理——条件流匹配与边缘流匹配具有相同的梯度
> 展开平方范数并使用后验平均恒等式，可得
>
> $$
> \mathcal{L}_{\mathrm{FM}}(\theta)
> =\mathcal{L}_{\mathrm{CFM}}(\theta)+C,
> $$
>
> 其中 $C$ 与 $\theta$ 无关。因此，
>
> $$
> \nabla_\theta\mathcal{L}_{\mathrm{FM}}(\theta)
> =\nabla_\theta\mathcal{L}_{\mathrm{CFM}}(\theta).
> $$
>
> 因而，对显式采样的条件目标进行学习，就能得到原本难以直接计算的边缘向量场。

> [!example] 示例——高斯路径的流匹配
> 代入
>
> $$
> X_t=\alpha_tZ+\beta_t\varepsilon,
> \qquad
> \varepsilon\sim\mathcal{N}(0,I_d),
> $$
>
> 得到
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
> 对于 CondOT 调度 $\alpha_t=t$ 和 $\beta_t=1-t$，它变为
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

训练过程不需要模拟动力系统：只需采样 $t$、数据点 $Z$ 和高斯噪声 $\varepsilon$，然后执行一次普通的
回归更新。只有训练完成后，从 $p_{\mathrm{init}}$ 生成样本时才需要求解 ODE。

## 分数函数与分数匹配

### 条件分数与边缘分数

> [!note] 定义——分数函数
> 对于可微且处处为正的密度 $q$，其**分数函数**定义为
>
> $$
> s_q(x)=\nabla_x\log q(x).
> $$
>
> 它指向对数密度增长最快的方向。对于概率路径，条件分数和边缘分数分别为
>
> $$
> s_t(x\mid z)=\nabla_x\log p_t(x\mid z),
> \qquad
> s_t(x)=\nabla_x\log p_t(x).
> $$

> [!abstract] 命题——分数边缘化
> 在积分号下对边缘密度求导，得到
>
> $$
> s_t(x)
> =\int s_t(x\mid z)
> \frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}\,dz
> =\mathbb{E}[s_t(x\mid Z)\mid X_t=x].
> $$
>
> 因此，从条件量到边缘量的平均对分数同样成立，与向量场的情形完全一致。

> [!example] 示例——高斯路径的分数
> 对高斯路径，
>
> $$
> s_t(x\mid z)
> =-\frac{x-\alpha_tz}{\beta_t^2}.
> $$
>
> 由于 $x=\alpha_tz+\beta_t\varepsilon$，同一个表达式也可以写成
>
> $$
> s_t(x\mid z)=-\frac{\varepsilon}{\beta_t}.
> $$

### 分数、向量场与去噪器

> [!abstract] 命题——高斯分数与向量场的转换
> 对高斯路径，条件向量场和分数是彼此的仿射重参数化：
>
> $$
> u_t^{\mathrm{target}}(x\mid z)
> =a_t\,s_t(x\mid z)+b_t x,
> $$
>
> 对于表达式有定义的内部时刻，
>
> $$
> a_t
> =\beta_t^2\frac{\dot{\alpha}_t}{\alpha_t}
> -\dot{\beta}_t\beta_t,
> \qquad
> b_t=\frac{\dot{\alpha}_t}{\alpha_t}.
> $$
>
> 对后验分布求平均，得到边缘恒等式
>
> $$
> u_t^{\mathrm{target}}(x)
> =a_t\,s_t(x)+b_tx.
> $$

> [!note] 定义——去噪器
> 另一种等价参数化是后验均值
>
> $$
> D_t(x)=\mathbb{E}[Z\mid X_t=x].
> $$
>
> 对高斯路径，
>
> $$
> s_t(x)
> =\frac{\alpha_tD_t(x)-x}{\beta_t^2}.
> $$

因此，学习边缘向量场、边缘分数或后验均值，实际上都在确定同一份条件信息，只是所用坐标不同。

### 在不改变边缘分布的前提下加入随机性

> [!tip] 定理——SDE 扩展技巧
> 假设 ODE
>
> $$
> dX_t=u_t^{\mathrm{target}}(X_t)\,dt
> $$
>
> 沿概率路径 $p_t$ 演化。对于任意扩散调度 $\sigma_t\geq 0$，SDE
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
> 具有相同的边缘概率律 $X_t\sim p_t$。事实上，
>
> $$
> \operatorname{div}\!\left(
> p_t\frac{\sigma_t^2}{2}\nabla\log p_t
> \right)
> =\frac{\sigma_t^2}{2}\Delta p_t,
> $$
>
> 所以分数漂移项会抵消 Fokker–Planck 方程中的扩散项。单条轨迹变成随机的，但预先指定的
> 边缘路径保持不变。

对于高斯路径，分数与向量场的转换意味着，只用一个学习得到的网络，就足以参数化确定性 ODE 采样器
以及一族 SDE 采样器。

### 分数匹配与去噪分数匹配

设 $s_t^\theta(x)$ 是一个神经分数模型。沿用相同的采样下标约定，理想的分数匹配目标是

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

边缘分数 $s_t(x)$ 无法直接获得，因此我们改为对条件分数做回归：

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

> [!tip] 定理——去噪分数匹配
> 与条件流匹配相同，
>
> $$
> \mathcal{L}_{\mathrm{SM}}(\theta)
> =\mathcal{L}_{\mathrm{DSM}}(\theta)+C,
> $$
>
> 其中 $C$ 与 $\theta$ 无关。因此，
>
> $$
> \nabla_\theta\mathcal{L}_{\mathrm{SM}}(\theta)
> =\nabla_\theta\mathcal{L}_{\mathrm{DSM}}(\theta).
> $$
>
> 因而，对可计算的条件分数做回归，就能学习边缘分数。

之所以称为**去噪分数匹配**，是因为 $X_t$ 是干净样本 $Z$ 经过扰动后的版本。

> [!example] 示例——高斯去噪分数匹配
> 对高斯路径，
>
> $$
> X_t=\alpha_tZ+\beta_t\varepsilon,
> \qquad
> s_t(X_t\mid Z)=-\frac{\varepsilon}{\beta_t},
> $$
>
> 因此
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
> 定义噪声预测网络
>
> $$
> \varepsilon_t^\theta(x)=-\beta_t s_t^\theta(x).
> $$
>
> 加权分数目标变为
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

> [!warning] 说明——简化的噪声预测损失
> 一个常用目标会去掉与时间有关的前置因子：
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
> 去掉 $1/\beta_t^2$ 会改变不同噪声水平的相对权重，尽管逐点回归目标仍然是 $\varepsilon$。

因此，分数匹配可以通过以下回归过程实现：

1. 采样数据点 $Z\sim p_{\mathrm{data}}$。
2. 采样 $t\sim\operatorname{Unif}[0,1]$ 和 $\varepsilon\sim\mathcal{N}(0,I_d)$。
3. 构造带噪输入 $X_t=\alpha_tZ+\beta_t\varepsilon$。
4. 预测条件分数 $-\varepsilon/\beta_t$、噪声 $\varepsilon$，或与之等价的去噪目标。
5. 使用选定的平方误差损失更新网络参数。

训练完成后，可以把学到的分数转换为用于 ODE 采样的边缘向量场，也可以将其放入经过分数校正的
SDE 中进行随机采样。

## 来源

本笔记主要依据 Peter Holderrieth 与 Ezra Erives 的 MIT 6.S184（2026）课程材料
*An Introduction to Flow Matching and Diffusion Models* 第 1–4 节及附录 A，并补充了测度论层面的
说明以及明确的概率论与 SDE 预备知识。
