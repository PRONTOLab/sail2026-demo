# SAIL 2026 demo: Reactant, Enzyme, Impulse

`SAIL2026_demo.ipynb` is the notebook from our demo at the NSF Summit for AI Institutes Leadership (SAIL 2026). It is saved with all outputs, so every figure is visible without running anything.

| section | what it shows |
|---|---|
| Reactant | the same code compiled for CPUs and accelerators; the Oceananigans ocean model, hand-written GPU kernels included, compiled and run live; scaling to supercomputers (GB-25) |
| Enzyme | compiler-generated gradients of existing code: a toy example, and SkyMatch's galaxy model differentiated as written |
| Black hole | imaging M87* from 2017 Event Horizon Telescope data with [Comrade.jl](https://github.com/ptiede/Comrade.jl), fitted with Enzyme gradients on the GPU |
| Impulse | Bayesian inference for the black-hole image, with the entire sampler compiled into one GPU program; a textbook model (McElreath's *Statistical Rethinking*, m14.10) that Impulse's compiler makes about 20× faster from the same code |

## Run it on Google Colab

1. Open the notebook in Colab: https://colab.research.google.com/github/PRONTOLab/sail2026-demo/blob/main/SAIL2026_demo.ipynb

   <img src="assets/colab_qr.png" width="141" alt="QR code: this notebook in Google Colab">

2. *Runtime → Change runtime type → Julia*, with a **T4 GPU**.
3. Run the Setup cell. The first run installs everything (expect about half an hour); then run the cells top to bottom.

The setup (`colab/colab_setup.jl`) installs:
- Reactant.jl with Impulse ([`impulse` branch](https://github.com/EnzymeAD/Reactant.jl/tree/impulse));
- a prebuilt CUDA 12 `libReactantExtra.so` for that branch, from this repository's `v1` release (built on Ubuntu 22.04, so it loads on Colab);
- Comrade.jl v0.11.33 with `colab/comrade_sail2026.patch` (three lines, for the newer Impulse API);
- Oceananigans 0.116.1 and CUDA.jl (Reactant compiles Oceananigans' GPU kernels through CUDA.jl's compiler).

## Files

- `demo_utils.jl`: setup and helpers for the live cells; `model.jl`: the Comrade M87 model (Comrade's StokesIImaging tutorial); `primates.jl`: the m14.10 model compiled with Impulse optimizations off and on; `ocean.jl`: Oceananigans' baroclinic adjustment on `ReactantState()` with the GB-25 time-stepping recipe; `cuda_libs.jl`: links cuDNN/NVRTC/NCCL from the `Reactant_jll` artifact next to the prebuilt library.
- `skymatch.jl`: a small CPU galaxy-luminosity and sensitivity example using the pinned SkyMatch package; no external catalog or SED tables needed.
- `data/`: M87 2017 visibilities (from Comrade's examples) and the 151-species primate data.
- `runs/demo_research/warm_state.jls`: the black-hole sampler's saved tuning state, so setup skips the warm-up.
