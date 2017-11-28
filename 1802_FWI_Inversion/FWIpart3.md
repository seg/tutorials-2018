---
title: Full-Waveform Inversion - Part 3``:`` Optimization
author: |
	Philipp Witte^1^\*, Mathias Louboutin^1^, Michael Lange^2^, Navjot Kukreja^2^, Fabio Luporini^2^, Gerard Gorman^2^, and Felix J. Herrmann^1,3^\
	^1^ Seismic Laboratory for Imaging and Modeling (SLIM), The University of British Columbia \
	^2^ Imperial College London, London, UK\
	^3^ now at Georgia Institute of Technology, USA \
bibliography:
	- bib_tuto.bib
---


## Introduction

This tutorial is the third part of a full-waveform inversion (FWI) tutorial series with a step-by-step walkthrough of setting up forward and adjoint wave equations and building a basic FWI inversion framework. For discretizing and solving wave equations, we use Devito, a Python domain-specific language for automated finite-difference code generation [@lange2016dtg]. The first two parts of this tutorial [@louboutin2017fwi; @louboutin2017bfwi] demonstrated how to solve the acoustic wave equation for modeling seismic shot records and how to compute the gradient of the FWI objective function through adjoint modeling. With these two key ingredients, we will now build an inversion framework for minimizing the FWI least-squares objective function and test it on a small 2D data set using the Overthrust model.

From the optimization point of view, full-waveform inversion is an extremely challenging problem, since not only do we need to solve expensive wave equations for a large number of shot positions and iterations, but the FWI objective function is also non-convex, meaning there exist (oftentimes many) local minima and saddle points. Furthermore, FWI is typically ill-posed, which means it is not possible to uniquely recover the parametrization of the subsurface from the seismic data alone that is collected at the surface. For these reasons, FWI forms a broad field of research with the focus lying on which misfit functions to choose, optimal parameterizations of the wave equations, optimization algorithms or how to include geological constraints and penalties [e.g. @vanleeuwen2013; @warner2014; @Peters2017].

This tutorial will demonstrate how we can set up a basic FWI framework with gradient-based optimization algorithms, such as the steepest descent, (quasi-) Newton or conjugate directions methods [@nocedal2006]. For the reader, this can serve as a starting point for implementing customized and problem-specific modifications of FWI such as multi-parameter FWI, inversion with alternative misfit functions or constraints and penalties. Since building a full framework for waveform inversion (including routines for data IO and parallelization) is outside the scope of a single tutorial, we will implement our inversion framework with Julia Devito, a Julia software package for seismic modeling and inversion based on Devito. Julia Devito provides mathematical abstractions and functions wrappers that allow to implement FWI and least-squares migration algorithms that closely follow the mathematical notation, while using Devito's automatic code generation for solving the underlying wave equations.

## Optimizing the full-waveform inversion objective function

In the previous tutorial, we demonstrated how to calculate the gradient of the FWI objective function with the $\ell_2$-misfit:

```math {#FWI}
	\mathop{\hbox{minimize}}_{\mathbf{m}} \hspace{.2cm} \Phi(\mathbf{m})= \sum_{i=1}^{n_s} \frac{1}{2} \left\lVert \mathbf{d}^\mathrm{pred}_i (\mathbf{m}, \mathbf{q}) - \mathbf{d}_i^\mathrm{obs} \right\rVert_2^2,
```

where $\mathbf{d}^\mathrm{pred}_i$ and $\mathbf{d}^\mathrm{obs}_i$ are the predicted and observed seismic shot records of the $i$th source location and $\mathbf{m}$ is the velocity model in slowness squared. As mentioned in the introduction, this objective function is non-convex, making it difficult to optimize, and its properties depend on many physical and environmental factors such as the acquisition geometry, the geology of the target area or frequency content of the observed data. Even though called *full-waveform inversion*, FWI with the $\ell_2$-norm misfit relies primarily on transmitted waves, such as diving and turning waves, while utilizing reflections for FWI is much harder and subject of current research [e.g. @xu2012full].

The most straight-forward approach for optimizing the FWI objective function is with local (gradient-based) optimization methods. Unlike numerically very expensive global methods, local methods find a minimum in vicinity of the starting point, with no guarantee that the solution is in fact the global minimum. The success of FWI therefore relies heavily on the initial guess, i.e. on the accuracy of the starting model. Initial velocity models that generate predicted shot records of which the events are shifted by more than half a wavelength (widely referred to as cycle skipping), cause local optimization algorithms to converge to local minima. Despite these issues, local gradient-based optimization algorithms are still the most widely used methods in practice, because the FWI gradient is comparatively easy and cheap to compute.

Algorithm #basic_fwi outlines the basic structure of gradient-based full-waveform inversion. The first step is a loop over the total number of source positions, in which we compute the predicted data $\mathbf{d}_k^\mathrm{pred}$ for the $k^{th}$ source location, as well as the function value $f$ and the gradient $\mathbf{g}$. The operator $\mathcal{F}(\mathbf{m}, \mathbf{q}_k)$ denotes the forward modeling scheme for the current source $\mathbf{q}_k$ as implemented in the first part of this tutorial seriers [@louboutin2017fwi]. Calculating the gradient as described in the second tutorial [@louboutin2017bfwi], can be expressed as the action of a linear operator $\nabla \mathcal{F}^\top$ that acts on the data residual. The operator $\nabla \mathcal{F}$ is the partial derivative of the forward modeling operator and commonly known as the Jacobian or demigration operator and its adjoint $\nabla \mathcal{F}^\top$ is the (reverse-time) migration operator (e.g. Symes). 


#### Algorithm: {#basic_fwi}
| Input: observed data $\mathbf{d}^\mathrm{obs}$, source wavelets $\mathbf{q}$, initial model $\mathbf{m}_0$
| **for** \ ``j=1 \text{ to } n_{iter}``
|
| 		# Calculate FWI function value and gradient for $n_s$ shots
| 		$f = 0, \mathbf{g} = \mathbf{0}$
| 		**for** \ ``k=1 \text{ to } n_{s}``
|				$\mathbf{d}^\mathrm{pred}_k = \mathcal{F}(\mathbf{m}_0, \mathbf{q}_k)$
|				$f = f + \frac{1}{2} \| \mathbf{d}^\mathrm{pred}_k - \mathbf{d}^\mathrm{obs}_k \|^2_2$
|				$\mathbf{g} = \mathbf{g} + \nabla \mathcal{F}^\top \Big( \mathbf{d}^\mathrm{pred}_k - \mathbf{d}^\mathrm{obs}_k \Big)$
| 		**end**
|
|		# Find search direction and update model
|		$\mathbf{d}_j = -\mathbf{D}\mathbf{g}$
|		$\mathbf{m}_0 = \mathbf{m}_0 + \alpha \mathbf{d}_j$
| **end**
: Basic structure of gradient-based full-waveform inversion algorithms.

To update the velocity model,  we need to find a search direction $\mathbf{d}_j$, which for gradient-based optimization is the negative gradient multiplied with a positive semi-definit matrix $\mathbf{D}$. The matrix can be thought of as a scaling operator that weights individual parts of the gradient. Its choice is defined by the optimization algorithm and in the simplest case, we have $\mathbf{D} = \mathbf{I}$, where $\mathbf{I}$ is the identity matrix (steepest descent). When the objective function can be well approximated by a quadratic, $\mathbf{D}$ can also be chosen to be the inverse hessian (Newton's method) or one of its approximations (Gauss-Newton and Quasi-Newton methods). Conjugate directions methods such as the conjugate gradient method, try to improve the linear convergence rate of steepest descent without expensive evaluations of second derivatives, by finding a set of conjugate directions $d_k$, that take previous gradients into account [@nocedal2006].

In the following section, we will demonstrate how to implement some of the above-mentioned algorithms using 

## FWI with Julia Devito

what is Julia Devito. Inversion framework. Translate algorithm to runnable Julia code. Use Devito to solve PDE. Set up PDE in Python as function
based around linear operators for forward/adjoint modeling, demigration/migration operators, and fwi_objective function. Examples with modeling..

fwi_objective(model, source, data) -> compute gradient + function value for wavelets and data. Like Python/Devito gradient but parallel loop over shots and sum gradients/function values.

-> use Julia Devito to translate algorithm to code


Set up 

```julia
 # Input: dobs, q, model
maxiter = 20
batchsize = 20
proj(x) = reshape(median([vec(mmin), vec(x), vec(mmax)]), model.n)

for j=1:maxiter

	# select current batch of shots
	idx = randperm(dobs.nsrc)[1:batchsize]
	
	# FWI objective function value and gradient
	d_pred = Pr*F*Ps'*q
	fval = .5f0*norm(d_pred - d_obs)^2
	grad = J'*(d_pred - d_obs)

	# line search and update model
	alpha = backtracking_linesearch(model, q[idx], dobs[idx], fval, grad, proj; alpha=1f-6)
	model.m -= reshape(alpha*grad, model.n)

	# update model and bound projection
	model.m = proj(model.m)
end
```

```
# Gauss-Newton method
x = grad
for k=1:maxiter_GN
	res = J*x - (d_pred - d_obs)
	g_gn = J'*res
	t = norm(res)^2/norm(g_gn)^2
	x -= t*g_gn
end
model.m -= reshape(x, model.n)	# alpha=1

```



#### Figure: {#result_marmousi_sgd}
![](Figures/fwi_overthrust.pdf){width=80%}
: Overthrust velocity model (top), FWI starting model (center) and inversion result after 20 iterations of stochastic gradient descent with bound constraints (bottom).

#### Figure: {#data_marmousi_sgd}
![](Figures/shot_records.pdf){width=95%}
: "Observed" seismic shot record (right), which is modeled using the true Overthrust model. The predicted shot record using the smooth initial model (center) is missing the reflections and has an incorrect turning wave, while the shot record modeled with the inversion result (right), looks very close to the original data.

easily extend to CG, different line search ...

alternative: interface optimization libraries to access L-BSFGS, SPG,
example with minConf?


## Discussion

## Installation

This tutorial is based on Devito version 3.1.0. It requires the installation of the full software with examples, not only the code generation API. To install Devito, run

	git clone -b v3.1.0 https://github.com/opesci/devito
	cd devito
	conda env create -f environment.yml
	source activate devito
	pip install -e .
 
### Useful links

- [Devito documentation](http://www.opesci.org/)
- [Devito source code and examples](https://github.com/opesci/Devito)
- [Tutorial notebooks with latest Devito/master](https://github.com/opesci/Devito/examples/seismic/tutorials)


## Acknowledgments

This research was carried out as part of the SINBAD II project with the support of the member organizations of the SINBAD Consortium. This work was financially supported in part by EPSRC grant EP/L000407/1 and the Imperial College London Intel Parallel Computing Centre.

## References


