# SAIL 2026 demo: Reactant · Enzyme · Impulse

`SAIL2026_demo.ipynb` is the notebook from our demo at the NSF Summit for AI Institutes Leadership (SAIL 2026). It is saved with all outputs, so every figure is visible without running anything.

| section | what it shows |
|---|---|
| Reactant | ordinary Julia code compiled for the GPU with one line; the Oceananigans ocean model compiled by changing one word, live; scaling to supercomputers (GB-25) |
| Enzyme | the gradient of a whole simulation with one line; recovering a hidden pattern by working backwards through a heat simulation |
| Black hole | imaging M87* from 2017 Event Horizon Telescope data with [Comrade.jl](https://github.com/ptiede/Comrade.jl) |
| Impulse | Bayesian uncertainty for the black hole; a textbook model (McElreath's *Statistical Rethinking*, m14.10) that Impulse's compiler makes about 20× faster without any rewriting |

## Run it on Google Colab

1. Open the notebook in Colab: `https://colab.research.google.com/github/PRONTOLab/sail2026-demo/blob/main/SAIL2026_demo.ipynb`
2. *Runtime → Change runtime type → Julia*, with a **T4 GPU**.
3. While this repository is private, run `ENV["SAIL_GITHUB_TOKEN"] = readline()` in a new cell and paste a GitHub token that can read `PRONTOLab/sail2026-demo`.
4. Run the Setup cell. The first run installs everything (expect about half an hour); then run the cells top to bottom.

The setup (`colab/colab_setup.jl`) installs:
- Reactant.jl with Impulse ([`impulse` branch](https://github.com/EnzymeAD/Reactant.jl/tree/impulse));
- a prebuilt CUDA 12 `libReactantExtra.so` for that branch, from this repository's `v1` release (built on Ubuntu 22.04, so it loads on Colab);
- Comrade.jl v0.11.33 with `colab/comrade_sail2026.patch` (three lines, for the newer Impulse API);
- Oceananigans 0.116.1 and CUDA.jl (Reactant compiles Oceananigans' GPU kernels through CUDA.jl's compiler).

## Files

- `demo_utils.jl`: setup and helpers for the live cells; `model.jl`: the Comrade M87 model (Comrade's StokesIImaging tutorial); `primates.jl`: the m14.10 model compiled with Impulse optimizations off and on; `ocean.jl`: Oceananigans' baroclinic adjustment on `ReactantState()` with the GB-25 time-stepping recipe; `cuda_libs.jl`: links cuDNN/NVRTC/NCCL from the `Reactant_jll` artifact next to the prebuilt library.
- `data/`: M87 2017 visibilities (from Comrade's examples) and the 151-species primate data.
- `runs/demo_research/warm_state.jls`: the black-hole sampler's saved tuning state, so setup skips the warm-up.
