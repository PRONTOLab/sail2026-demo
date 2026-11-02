# Draws the black-hole images that the notebook replays: the Impulse sampler on one GPU, 8 calls of 10.
include(joinpath(@__DIR__, "demo_utils.jl"))
seconds = precompute_sampling!()
println("80 images in ", round(seconds[end]; digits = 1), " s; saved to ", SAMPLING)
