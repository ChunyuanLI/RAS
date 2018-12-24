# Constrained Domain Sampling

Two dstribution examples are provided to compare different sampling method: 1-mode and 2-mode. Their support is defined in [c1,c2].

We the following table to show how to use the code.


| Method | 1-mode  |   2-mode 
|-------------------------|:-------------------------:|:-------------------------:
| RAS with Beta ref. | [L2S_constrained_beta_1mode.ipynb](./L2S_constrained_beta_1mode.ipynb)  |  [L2S_constrained_beta_2mode.ipynb](./L2S_constrained_beta_2mode.ipynb)
| RAS with Gaussian ref. | [L2S_constrained_Gaussian_1mode.ipynb](./L2S_constrained_Gaussian_1mode.ipynb)  |  [L2S_constrained_Gaussian_2mode.ipynb](./L2S_constrained_Gaussian_2mode.ipynb)
| (Amortized) SVGD  | [demo_amortized_svgd_1mode.ipynb](./demo_amortized_svgd_1mode.ipynb)  |  [demo_amortized_svgd_2mode.ipynb](./demo_amortized_svgd_2mode.ipynb)
| (Amortized) MCMC  | [demo_amortized_mcmc_1mode.ipynb](./demo_amortized_mcmc_1mode.ipynb)  |  [demo_amortized_mcmc_2mode.ipynb](./demo_amortized_mcmc_2mode.ipynb)
| **Plot Figs** | [plot_estimation_mode1.ipynb](./plot_estimation_mode1.ipynb)  |  [plot_estimation_mode1.ipynb](./plot_estimation_mode1.ipynb)
   


TODO: MCMC and its amorized version


We propose to use Beta distribution for constrained domains. Note that: (1) the default two-parameter Beta distribution typically has support [0,1]. For Beta distributions defined in interval  [c1,c2],  one can use the “reparameterization trick”: X = (Y-c1)/(c2-c1), which transforms Y in [c1, c2] to X in [0,1]. (2) The generalized Beta distribution B(a,b,c1,c2) has four parameters, with [c1, c2] defining the support. When c1=0, c2=1, we recover the typical two-parameter Beta distribution.
