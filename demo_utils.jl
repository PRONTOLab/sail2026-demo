include(joinpath(@__DIR__, "ocean.jl"))
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "primates.jl"))

using Optimisers, CairoMakie, Statistics, Printf, Serialization, Random
using Reactant: ProbProg
using .Primates
using .Ocean

CairoMakie.activate!(; type = "png", px_per_unit = 1.5)

const RUNS = joinpath(@__DIR__, "runs", "demo_research")
const WARM_STATE = joinpath(RUNS, "warm_state.jls")
const NADAPT = 500
const CHUNK = 10
const FIT_LR = 0.01

function median_time(f, args...)
    t1 = @elapsed f(args...)
    n = clamp(round(Int, 0.5 / max(t1, 1.0e-6)), 3, 50)
    return median([@elapsed(f(args...)) for _ in 1:n])
end

"""Run `f` on the CPU and compiled for the GPU; print both times and both answers."""
function cpu_vs_gpu(f, x, x_gpu)
    f_gpu = @compile sync = true f(x_gpu)
    tcpu, tgpu = median_time(f, x), median_time(f_gpu, x_gpu)
    @printf "CPU: %8.2f ms   answer %.6g\n" 1e3 * tcpu f(x)
    @printf "GPU: %8.2f ms   answer %.6g\n" 1e3 * tgpu Float64(f_gpu(x_gpu))
    @printf "one added line: %s× faster, same answer\n" round(Int, tcpu / tgpu)
    return nothing
end

const QUIET = Ref(false)

"""Oceananigans on Reactant: a warm/cold front breaking into eddies, one frame per 50 time steps."""
ocean_eddies!(; frames = 36) = run_ocean!(frame!; frames)

function frame!(fig)
    QUIET[] && (Makie.colorbuffer(fig); return nothing)
    Main.IJulia.clear_output(true)
    display(fig)
    return nothing
end

"""Render every live figure once off screen, so plotting code is compiled before the talk."""
function warm_plots!()
    QUIET[] = true
    run_ocean!(frame!; frames = 0)
    heat_spread!(; every = HEAT_STEPS)
    Makie.colorbuffer(heat_reveal(unblur!(; steps = 30, every = 30)))
    xb = fit!(; steps = 10, every = 10)
    imgs = sample_live!(; nchunks = 1)
    Makie.colorbuffer(show_draws(imgs))
    Makie.colorbuffer(compare(xb, imgs))
    Makie.colorbuffer(plotfields(dvis, :U, :V; figure_kwargs = (; size = (640, 520))).figure)
    QUIET[] = false
    return nothing
end

skyimage(θ) = intensitymap(skymodel(post, θ), gimg)

function blackhole_axis(pos; title = "")
    ax = Axis(pos; xreversed = true, aspect = 1, title, titlesize = 18)
    hidedecorations!(ax)
    return ax
end

function opt_step(opt_state, tpost, x)
    res = Enzyme.gradient(Enzyme.ReverseWithPrimal, Const(logdensityof), Const(tpost), x)
    opt_state, x = Optimisers.update!(opt_state, x, -res.derivs[2])
    return opt_state, x, res.val
end

comrade_ldf(x, tpost) = logdensityof(tpost, x)

function nuts_chunk(state, tpost, ns)
    return ProbProg.mcmc_logpdf(
        state, comrade_ldf, tpost;
        algorithm = :NUTS, num_warmup = 0, num_samples = ns,
        max_tree_depth = 10, max_delta_energy = 1000.0,
        adapt_step_size = false, adapt_mass_matrix = false, strong_zero = true,
    )
end

const X_START = prior_sample(StableRNG(42), tpost)
const COMPILED = Dict{Symbol, Any}()

# ---------------------------------------------------------------------------
# Enzyme: run loop-style simulation code backwards
# ---------------------------------------------------------------------------

const HEAT_N = 64
const HEAT_STEPS = 60
const HEAT_LAMBDA = 3.0e-3
const GLYPHS = Dict(
    'A' => [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    'I' => ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"],
)

function word_image(word; n = HEAT_N, scale = 4)
    rows = [join((GLYPHS[c][r] for c in word), ".") for r in 1:7]
    bits = [rows[r][c] == '#' ? 1.0 : 0.0 for r in 1:7, c in 1:length(rows[1])]
    big = kron(bits, ones(scale, scale))
    img = zeros(n, n)
    r0, c0 = (n - size(big, 1)) ÷ 2, (n - size(big, 2)) ÷ 2
    img[r0 + 1:r0 + size(big, 1), c0 + 1:c0 + size(big, 2)] .= big
    return img
end

"""Heat spreading on a plate, written the way simulation codes are written: a border of
zeros, two buffers, and loops over time steps and grid points that update arrays in place."""
function simulate!(u, unew, steps = HEAT_STEPS)
    for step in 1:steps
        for j in 2:HEAT_N+1, i in 2:HEAT_N+1
            unew[i, j] = u[i, j] + 0.2 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - 4u[i, j])
        end
        u, unew = unew, u
    end
    return u
end

function plate_after(u0, steps)
    u, unew = zeros(HEAT_N + 2, HEAT_N + 2), zeros(HEAT_N + 2, HEAT_N + 2)
    u[2:HEAT_N+1, 2:HEAT_N+1] .= u0
    return simulate!(u, unew, steps)[2:HEAT_N+1, 2:HEAT_N+1]
end

function edge_penalty(x)
    s = 0.0
    for j in 1:HEAT_N, i in 1:HEAT_N
        i < HEAT_N && (s += sqrt((x[i+1, j] - x[i, j])^2 + 1.0e-6))
        j < HEAT_N && (s += sqrt((x[i, j+1] - x[i, j])^2 + 1.0e-6))
    end
    return s
end

"""How far the plate simulated from the guess `u0` is from what we observe, plus a
preference for sharp edges."""
function mismatch(u0, observed)
    u, unew = zeros(HEAT_N + 2, HEAT_N + 2), zeros(HEAT_N + 2, HEAT_N + 2)
    u[2:HEAT_N+1, 2:HEAT_N+1] .= u0
    v = simulate!(u, unew)
    s = 0.0
    for j in 1:HEAT_N, i in 1:HEAT_N
        s += (v[i+1, j+1] - observed[i, j])^2
    end
    return s + HEAT_LAMBDA * edge_penalty(u0)
end

heat_gradient(u0) = Enzyme.gradient(Enzyme.Reverse, mismatch, u0, Const(HEAT_OBS))[1]

const HEAT_TRUTH = word_image("AI")
const HEAT_OBS = plate_after(HEAT_TRUTH, HEAT_STEPS) .+ 0.01 .* randn(StableRNG(7), HEAT_N, HEAT_N)

function field_axis(pos; title = "")
    ax = Axis(pos; title, titlesize = 18, aspect = 1, yreversed = true)
    hidedecorations!(ax)
    return ax
end
fieldplot!(ax, u::AbstractMatrix; kw...) = heatmap!(ax, permutedims(u); colormap = :inferno, kw...)
fieldplot!(ax, u::Observable; kw...) = heatmap!(ax, lift(permutedims, u); colormap = :inferno, kw...)

"""Show the hidden pattern spreading out, frame by frame."""
function heat_spread!(; every = 10)
    obs = Observable(copy(HEAT_TRUTH))
    fig = Figure(; size = (520, 560))
    ax = field_axis(fig[1, 1]; title = "time step 0")
    fieldplot!(ax, obs; colorrange = (0, 1))
    frame!(fig)
    for k in every:every:HEAT_STEPS
        obs[] = plate_after(HEAT_TRUTH, k)
        ax.title = "time step $k"
        frame!(fig)
        sleep(0.15)
    end
    return nothing
end

"""Recover the hidden starting pattern from the blurred observation by gradient descent
through the whole loop simulation (Enzyme on the CPU), redrawing every `every` steps."""
function unblur!(; steps = 600, every = 30)
    u0 = zeros(HEAT_N, HEAT_N)
    opt_state = Optimisers.setup(Optimisers.Adam(0.05), u0)
    rec = Observable(copy(u0))
    fig = Figure(; size = (1000, 540))
    fieldplot!(field_axis(fig[1, 1]; title = "what we observe"), HEAT_OBS)
    ax = field_axis(fig[1, 2]; title = "recovered starting pattern: step 0")
    fieldplot!(ax, rec; colorrange = (0, 1))
    frame!(fig)
    t = 0.0
    for i in 1:steps
        t += @elapsed opt_state, u0 = Optimisers.update!(opt_state, u0, heat_gradient(u0))
        if i % every == 0 || i == steps
            rec[] = copy(u0)
            ax.title = "recovered starting pattern: step $i"
            frame!(fig)
        end
    end
    if !QUIET[]
        tsim = median([@elapsed(mismatch(u0, HEAT_OBS)) for _ in 1:20])
        tgrad = median([@elapsed(heat_gradient(u0)) for _ in 1:20])
        @printf "%d gradients through all %d time steps of the loops: %.1f s on one CPU core\n" steps HEAT_STEPS t
        @printf "each gradient costs about as much as %.1f runs of the simulation\n" tgrad / tsim
    end
    return u0
end

function heat_reveal(rec)
    fig = Figure(; size = (1200, 430))
    fieldplot!(field_axis(fig[1, 1]; title = "hidden starting pattern"), HEAT_TRUTH)
    fieldplot!(field_axis(fig[1, 2]; title = "what we observed"), HEAT_OBS)
    fieldplot!(field_axis(fig[1, 3]; title = "recovered with gradients"), rec; colorrange = (0, 1))
    return fig
end

function prepare_demo!()
    mkpath(RUNS)
    xr = Reactant.to_rarray(rand(Float32, 8))
    @jit sum(abs2, xr)

    xo = Reactant.to_rarray(X_START)
    COMPILED[:setup] = @compile Optimisers.setup(Optimisers.AdamW(FIT_LR), xo)
    opt_state = COMPILED[:setup](Optimisers.AdamW(FIT_LR), xo)
    COMPILED[:step] = @compile sync = true opt_step(opt_state, tpostr, xo)

    heat_gradient(zeros(HEAT_N, HEAT_N))
    prepare_ocean!()

    if !isfile(WARM_STATE)
        xbest = fit!(; steps = 5000, every = 0)
        res = sample(
            Reactant.ReactantRNG(Reactant.to_rarray(UInt64[2026, 11])), postr,
            ReactantNUTS(; n_adapts = NADAPT), CHUNK;
            initial_params = Comrade.transform(tpost, xbest), chunk_size = CHUNK,
            warmup_callback = Returns(nothing),
        )
        ProbProg.save_state(WARM_STATE, res.state)
    end
    state = ProbProg.load_state(WARM_STATE)
    COMPILED[:nuts] = Reactant.Compiler.compile(nuts_chunk, (state, tpostr, CHUNK); optimize = :probprog)
    prepare_primates!()
    warm_plots!()
    return nothing
end

"""
Gradient ascent on the log-posterior with AdamW, redrawing the image every `every` steps.
Returns the final position (unconstrained, on the host).
"""
function fit!(; steps = 5000, every = 250)
    xo = Reactant.to_rarray(X_START)
    opt_state = COMPILED[:setup](Optimisers.AdamW(FIT_LR), xo)
    step_r = COMPILED[:step]
    if every > 0
        img = Observable(skyimage(Comrade.transform(tpost, X_START)))
        fig = Figure(; size = (520, 520))
        ax = blackhole_axis(fig[1, 1]; title = "step 0")
        image!(ax, img; colormap = :afmhot)
        frame!(fig)
    end
    tgpu = 0.0
    for i in 1:steps
        tgpu += @elapsed (opt_state, xo, _) = step_r(opt_state, tpostr, xo)
        if every > 0 && (i % every == 0 || i == steps)
            img[] = skyimage(Comrade.transform(tpost, Array(xo)))
            ax.title = "gradient step $i / $steps"
            frame!(fig)
        end
    end
    every > 0 && !QUIET[] && println(@sprintf("%d gradient steps on %d unknowns: %.1f s of GPU time", steps, length(X_START), tgpu))
    return Array(xo)
end

"""
Draw `nchunks × $(CHUNK)` posterior samples with the precompiled NUTS kernel, starting from
the warmup checkpoint, and redraw the latest draw and the running average per chunk.
Returns the vector of sampled images.
"""
function sample_live!(; nchunks = 8)
    state = ProbProg.load_state(WARM_STATE)
    cfn = COMPILED[:nuts]
    imgs = typeof(skyimage(Comrade.transform(tpost, X_START)))[]
    fig = Figure(; size = (900, 500))
    obs_last = Observable(skyimage(Comrade.transform(tpost, X_START)))
    obs_mean = Observable(obs_last[])
    image!(blackhole_axis(fig[1, 1]; title = "one plausible image"), obs_last; colormap = :afmhot)
    image!(blackhole_axis(fig[1, 2]; title = "average of all plausible images so far"), obs_mean; colormap = :afmhot)
    supertitle = Label(fig[0, 1:2], "sampling…"; fontsize = 22)
    tgpu = 0.0
    for k in 1:nchunks
        tgpu += @elapsed begin
            samples, _, _, _, state = cfn(state, tpostr, CHUNK)
            raw = Array(samples)
        end
        append!(imgs, [skyimage(Comrade.transform(tpost, r)) for r in eachrow(raw)])
        obs_mean[] = mean(imgs)
        obs_last[] = imgs[end]
        supertitle.text = @sprintf("%d plausible black holes so far  (%.0f s on one GPU)", length(imgs), tgpu)
        frame!(fig)
    end
    return imgs
end

function show_draws(imgs; n = 6)
    idx = round.(Int, range(1, length(imgs); length = n))
    fig = Figure(; size = (1050, 760))
    for (j, i) in enumerate(idx)
        ax = blackhole_axis(fig[(j - 1) ÷ 3 + 1, (j - 1) % 3 + 1])
        image!(ax, imgs[i]; colormap = :afmhot)
    end
    Label(fig[0, 1:3], "Different black holes, all consistent with the same telescope data"; fontsize = 22)
    return fig
end

function compare(xbest, imgs)
    fig = Figure(; size = (900, 470))
    ax1 = blackhole_axis(fig[1, 1]; title = "one best guess (optimization)")
    ax2 = blackhole_axis(fig[1, 2]; title = "average over plausible images (Bayesian)")
    image!(ax1, skyimage(Comrade.transform(tpost, xbest)); colormap = :afmhot)
    image!(ax2, mean(imgs); colormap = :afmhot)
    return fig
end
