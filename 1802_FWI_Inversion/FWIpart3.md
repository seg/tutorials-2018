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

From the optimization point of view, full-waveform inversion is an extremely challenging problem, since not only do we need to solve expensive wave equations for a large number of shot positions and iterations, but the FWI objective function is also non-convex, meaning there exist (oftentimes many) local minima and saddle points. Furthermore, FWI is typically ill-posed, which means it is not possible to uniquely recover the parametrization of the subsurface from the seismic data alone that is collected at the surface. For these reasons, FWI forms a broad field of research with focus lying on which misfit functions to choose, optimal parameterizations of the wave equations, optimization algorithms or how to include geological constraints and penalties [e.g. @vanleeuwen2013; @warner2014; @Peters2017].

This tutorial will demonstrate how we can set up a basic FWI framework with gradient-based optimization algorithms, such as the steepest descent, (Quasi-) Newton or conjugate directions methods [@nocedal2006]. For the reader, this can serve as a starting point for implementing customized and problem-specific modifications of FWI such as multi-parameter FWI, inversion with alternative misfit functions or constraints and penalties. Since building a full framework for waveform inversion (including routines for data IO and parallelization) is outside the scope of a single tutorial, we will implement our inversion framework with Julia Devito, a Julia software package for seismic modeling and inversion based on Devito. Julia Devito provides mathematical abstractions and functions wrappers that allow to implement FWI and least-squares migration algorithms that closely follow the mathematical notation, while using Devito's automatic code generation for solving the underlying wave equations.

## Optimizing the full-waveform inversion objective function

In the previous tutorial, we demonstrated how to calculate the gradient of the FWI objective function with the $\ell_2$-misfit:

```math {#FWI}
	\mathop{\hbox{minimize}}_{\mathbf{m}} \hspace{.2cm} \Phi(\mathbf{m})= \sum_{i=1}^{n_s} \frac{1}{2} \left\lVert \mathbf{d}^\mathrm{pred}_i (\mathbf{m}, \mathbf{q}) - \mathbf{d}_i^\mathrm{obs} \right\rVert_2^2,
```

where $\mathbf{d}^\mathrm{pred}_i$ and $\mathbf{d}^\mathrm{obs}_i$ are the predicted and observed seismic shot records of the $i$th source location and $\mathbf{m}$ is the velocity model in slowness squared. As mentioned in the introduction, this objective function is non-convex, making it difficult to optimize, and its properties depend on many physical and environmental factors such as the acquisition geometry, the geology of the target area and frequency content of the observed data. Even though called *full-waveform inversion*, FWI with the $\ell_2$-norm misfit relies primarily on transmitted waves, such as diving and turning waves, while utilizing reflections for FWI is much harder and subject of current research [e.g. @xu2012full].

The most straight-forward approach for optimizing the FWI objective function is with local (gradient-based) optimization methods. Unlike numerically very expensive global methods, local methods find a minimum in vicinity of the starting point, with no guarantee that the solution is in fact the global minimum. The success of FWI therefore relies heavily on the initial guess, i.e. on the accuracy of the starting model. Initial velocity models that generate predicted shot records of which the events are shifted by more than half a wavelength, widely referred to as cycle skipping, cause local optimization algorithms to converge to local minima. Despite these issues, local gradient-based optimization algorithms are still the most widely used methods in practice, because the FWI gradient is comparatively easy and cheap to compute.

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

To update the velocity model,  we need to find a search direction $\mathbf{d}_j$, which for gradient-based optimization is the negative gradient multiplied with a positive semi-definit matrix $\mathbf{D}$. The matrix can be thought of as a scaling operator that weights individual parts of the gradient. Its choice is defined by the optimization algorithm and in the simplest case, we have $\mathbf{D} = \mathbf{I}$, where $\mathbf{I}$ is the identity matrix (steepest descent). When the objective function can be well approximated by a quadratic, $\mathbf{D}$ can also be chosen to be the inverse hessian (Newton's method) or one of its approximations (Gauss-Newton and Quasi-Newton methods). Conjugate directions methods such as the conjugate gradient method, try to improve the linear convergence rate of steepest descent without expensive evaluations of second derivatives, by finding a set of conjugate directions $d_k$, that take previous gradients into account [@nocedal2006]. The step length $\alpha$ is a scalar that determines how far we move along the search direction $\mathbf{d}_k$ and is equal to one for Newton's method. For all other methods, $\alpha$ has to be determined with an (inexact) line search, so that updating the model with $\mathbf{m}_0 + \alpha \mathbf{d}_j$ leads to a decrease of the FWI function value. We will now move on to the practial part of this tutorial and implement some of these algorithm using a concrete example.


## The jSeis framework for modeling and inversion

For implementing our full-waveform inversion examples, we will use jSeis, a seismic inversion framework based on Devito and built around matrix-free linear operators and data containers that allow to quickly translate algorithms to runnable Julia scripts. The underlying wave equations are set up and solved using Devito and the Python code is interfaced from Julia using the PyCall package [@Johnson2017]. jSeis provides a set of matrix-free linear operators for forward and time reversal modeling, as well as source/receiver projections and migration/demigration operators that encapsulate the wave equation solvers. We will demonstrate the basic functionalities of jSeis using a small test data set that was generated with the 2D Overthrust velocity model [ref]. 

We start by downloading the velocity model and the data set, which consists of 97 shot records. For reading and writing SEG-Y data, jInv uses the SeisIO package [@Keegan2017], a sophisticated SEG-Y reader that allows to scan large 3D data sets for creating look-up tables with header summaries. However, since our data set is relatively small and consists only of a singe SEG-Y file, we will directly load the full file into memory. The `segy_read` command takes the file name as an input and returns a dense data block, from which we construct an abstract jSeis data vector:

```julia
  using seisIO, jSeis
  block = segy_read("overthrust_2d_shots.segy")
  d_obs = joData(block)
```

The `d_obs` object is an abstract vector, that can be used like a regular Julia vector, i.e. we can compute norms `norm(d_obs)` or dot products `dot(d_obs, d_obs)`, while containing the shot records in their original dimension, thus avoiding reapeated vectorizing and reshaping of data. Shot records can be accessed via their respective shot number with `d_obs.data[shot_no]`, while the header information can be accessed with `d_obs.geometry`. Since a seismic data sets contains the source coordinates, but not the source function itself, we read the source geometry from the SEG-Y file and then set up a source vector `q` with a 15 Hertz Ricker wavelet:

```julia
  src_geometry = Geometry(block)
  src_data = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.015)
  q = joData(src_geometry, src_data)
```

Since our data set consists of 97 shot records, both `d_obs` and `q` contain the data and geometries for all source positions. We can check the number of source positions with `d_obs.nsrc` and `q.nsrc` and we can extract the part of the vector that corresponds to one or multiple shots with `d_obs[shot_no], q[shot_no]`. Having set up vectors for the observed data and sources, we can now construct matrix-free linear operators for wave equations and gradients. This will allow us to express modeling a shot record as `d_pred = Pr*F*Ps'*q`, where `Pr` and `Ps` are matrix-free operators around the Devito sparse point injection and interpolation [@louboutin2017fwi] and `F` is the inverse of the acoustic wave equation. Multiplications with `Ps` and `Pr` represent sampling the wavefield at source/receiver locations, while their adjoints `Pr', Ps'` denote injecting either shot records or source wavelets. Since the dimensions of the inverse wave equation operator depend on the number of computational time steps, we calculate this number using the `get_computational_nt` function and set up an `info` object that contains some dimensionality information required by all operators. The projection and modelig operators can then be set up in Julia in the following way:

```julia
  ntComp = get_computational_nt(q.geometry, d_obs.geometry, model0)
  info = Info(prod(model0.n), d_obs.nsrc, ntComp)
  Pr = joProjection(info, d_obs.geometry)
  Ps = joProjection(info, q.geometry)
  F = joModeling(info, model)
```

We can now forward model all 97 predicted shot records by typing `d_pred = Pr*F*Ps'*q` into the Julia terminal. If we started our Julia session with more than one worker, then the wave equation solves are parallelized over source locations and all shots are collected in the `d_pred` vector. We can also model a single or subset of shots by indexing the operators with the respective shot numbers,e.g. if we only want to model the first two shots, we define `i=[1,2]` and then run `d_sub = Pr[i]*F[i]*Ps[i]'*q[i]`. Accordingly, if we want to solve an adjoint wave equation with the observed data as the adjoint source and restrictions of the wavefields back to the source locations, we can run `qad = Ps*F'*Pr'*d_obs`. Furthermore, jSeis allows to create a linearized modeling operator `J` from a forward modeling operator and a specified source vector. As mentioned in the introduction, this operator is also known as the Jacobian or demigration operator and its adjoint `J'` is the migration or gradient operator. In Julia we can set up the Jacobian and compute the FWI gradient for one source location in the following way:

```julia
  J = joJacobian(Pr*F*Ps',q)
  i = 10
  d_pred = Pr[i]*F[i]*Ps[i]'*q[i]
  g = J[i]'*(d_pred - d_obs[i])
```

One draw back of this notation, is that the forward wavefields for the gradient calculation have to be recomputed, since the forward modeling operator only returns the shot records and not the wavefields. For this reason jSeis has an additional function for computing the gradients of the FWI objective function `f,g = fwi_objective(model,q[i],d_obs[i])`, which takes the current model, source and data vectors as an input and computes the function value and gradient in parallel and without having to recompute the forward wavefields.


## FWI using jSeis

With expressions for modeling operators, Jacobians and gradients and function values of the FWI objective, we can now implement several different FWI algorithms in a few lines of code. We will start with a basic gradient descent (GD) example with a line search, as outlined in Algorithm #basic_fwi, but including two minor modifications. To reduce the computational cost of full gradient descent, we will use a stochastic approach in which we only compute the gradient and function value for a randomized subset of source locations, i.e. the inner sum in Algorithm #basic_fwi does not run over all 97 shots, but a smaller subset of that. In jSeis, this is accomplished by choosing a random vector of integers between 1 and 97 and indexing the data vectors as described earlier. The second modification is applying bound constraints to the updated velocity model, to prevent velocities (squared slownesses) to become negative or too large and therefore violate the CFL condition, which ensures stability of the modeling operators. Bound constraints are applied to the updated model in the form of a projection operator `proj(x)`, which clips values of the slowness that exceed their allowed range. The full algoritm for stochastic gradient descent with bound constraints is then implemented as follows:

```julia
maxiter = 20
batchsize = 10
proj(x) = reshape(median([vec(mmin), vec(x), vec(mmax)]), model.n)

for j=1:maxiter

	# indices for randomized subset of shot locations
	i = randperm(dobs.nsrc)[1:batchsize]
	
	# FWI objective function value and gradient
	fval, grad = fwi_objective(model0, q[i], d_obs[i])

	# line search and update model
	alpha = backtracking_linesearch(model0, q[i], dobs[i], fval, grad, proj; alpha=1f-6)
	model.m += reshape(-alpha*grad, model0.n)

	# apply bound constraints
	model.m = proj(model0.m)
end
```

The function `backtracking_linesearch` performs an approximate line search for a value of `alpha` that leads to a suffient decrease of the FWI function value (Armijo condition) [@nocedal2006]. While the convergence rate of this algorithm depends on the objective function, it is at best linear for full GD and sublinear for stochastic GD. Without having to compute second derivatives or one of their approximations, this rate can be improved to superlinear with the non-linear conjugate gradient method (CG). The CG algorithm only requires to additionaly store the gradient `g_prev` and update direction `d_prev` of the previous iteration (both are set to zero for the first iteration). A new search direction is computed as a linear combination of the current gradient and previous search direction: $\mathbf{d}_j = -\mathbf{g}_j + \beta \mathbf{d}_{j-1}$, with $\beta$ given by one of the various CG update rules such as $\beta = \frac{\mathbf{g}_j^\top \mathbf{g}_j}{\mathbf{g}_{j-1}^\top \mathbf{g}_{j-1}}$ (Fletcher-Reeves) or $\beta = \frac{\mathbf{g}_j^\top (\mathbf{g}_j - \mathbf{g}_{j-1}}{\mathbf{g}_j^\top \mathbf{g}_j}$ (Polak-Ribiere), with the latter being the most popular choice [@nocedal2006]. In Julia, we can easily modify our previous algorithm by replacing lines xx to xx with:

```
  beta = dot(g, g - g_prev)/dot(g_prev, g_prev)	# Polak-Ribiere
  d = -g + beta*d_prev
  alpha = backtracking_linesearch(model0, q[i], dobs[i], fval, d, proj; alpha=1f-6)
  model.m += reshape(alpha*d, model0.n)
  g_prev = g
  d_prev = d
```

In cases where we have already made decent progress towards the solution or start with a very good initial model, convergence can be further improved by using second order methods, i.e. algorithms that use curvature information of the objective function. Second order methods, namely Newton's method or one of its derivatives, assume a quadratic shape of the objective function and their benefits are lost if this assumption is severly violated. For large-scale problems such as FWI, the exact Newton's method is prohibitively expensive, since it involves computing the second derivative matrix `$\mathbf{H}$ (Hessian) and solving a linear system $\mathbf{H} \mathbf{d} = \mathbf{g}$ for obtaining a search direction. For least-squares problems, the full hessian can be approximated by the Gauss-Newton hessian $\mathbf{J}^\top \mathbf{J}$, the matrix still needs to be inverted for finding a search direction. Since the GN hessian is often ill-conditioned, it is usually preferable to solve the overdetermined linear system $\mathbf{J} \mathbf{d} = \mathbf{d}^\mathrm{pred}_k - \mathbf{d}^\mathrm{obs}_k$ instead (also widely known as least-squares reverse-time migration). For practical purposes, it is sufficient to perform only a few GD iterations for the the least squares subproblem, which corresponds to a few demigrations-migrations. The implementation of FWI with the (truncated) GN method is straight-forward:

```
# Gauss-Newton method
for j=1:maxiter
    i = randperm(d_obs.nsrc)[1:batchsize]
    d_pred = Pr[i]*F[i]*Ps[i]'*q[i]
    d = zeros(Float32, info.n)
    for k=1:maxiter_GN
      r = J[i]*d - (d_pred - d_obs[i])
      g_gn = J[i]'*r
      t = norm(r)^2/norm(g_gn)^2
      d -= t*g_gn
    end
    model.m = proj(model0.m - reshape(d, model.n))	# alpha=1
end
```

An alternative for solving an additional least squares sub-problem for obtaining a search direction are Quasi-Newton methods, which build up an approximation of the Hessian from previous gradients without the need for additional PDE solves. Implementing an efficient and correct version of this method, such as the L-BFGS algorithm, exceeds a few lines of code, which is why we refer the reader to the literature [ref to LBFGS]. Instead of implementing more complicated algorithms by hand, it is also possible to interface third-party optimization libraries and an example for this is given in the notebook *fwi_overthrust_minConf.jl*. 


#### Figure: {#result_marmousi_sgd}
![](Figures/fwi_overthrust.pdf){width=80%}
: Overthrust velocity model (top), FWI starting model (center) and inversion result after 20 iterations of stochastic gradient descent with bound constraints (bottom).

#### Figure: {#data_marmousi_sgd}
![](Figures/shot_records.pdf){width=95%}
: "Observed" seismic shot record (right), which is modeled using the true Overthrust model. The predicted shot record using the smooth initial model (center) is missing the reflections and has an incorrect turning wave, while the shot record modeled with the inversion result (right), looks very close to the original data.

## Conclusions



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


