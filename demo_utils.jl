include(joinpath(@__DIR__, "ocean.jl"))
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "primates.jl"))
include(joinpath(@__DIR__, "skymatch.jl"))

using Optimisers, CairoMakie, Statistics, Printf, Serialization, Random, Base64
using Optimisers: Adam, AdamW
using Enzyme: Reverse, Forward, Const, Duplicated
using Oceananigans: RectilinearGrid, HydrostaticFreeSurfaceModel, BetaPlane, BuoyancyTracer,
    WENO, WENOVectorInvariant, SplitExplicitFreeSurface, set!, Periodic, Bounded
using Oceananigans.Units: kilometers
using Oceananigans.Architectures: ReactantState
using Reactant: ProbProg
using .Primates
using .Ocean
using .SkyMatchDemo

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

const FRAMES = String[]
const FRAME_WIDTH = Ref(0)

"""Add `fig` as the next frame of the animation that `show_animation` plays."""
function frame!(fig)
    QUIET[] && (Makie.colorbuffer(fig); return nothing)
    dir = isempty(FRAMES) ? mktempdir() : dirname(first(FRAMES))
    isempty(FRAMES) && (FRAME_WIDTH[] = round(Int, fig.scene.viewport[].widths[1]))
    push!(FRAMES, joinpath(dir, @sprintf("%03d.png", length(FRAMES) + 1)))
    save(last(FRAMES), fig)
    return nothing
end

"""Write the frames added since the last call to `path` as a looping GIF that holds its last frame."""
function save_animation(path; fps = 6, hold = 2)
    dir = dirname(first(FRAMES))
    ffmpeg = CairoMakie.Makie.FFMPEG_jll.ffmpeg()
    filter = "tpad=stop_mode=clone:stop_duration=$hold,split[a][b];[a]palettegen[p];[b][p]paletteuse"
    run(`$ffmpeg -loglevel error -y -framerate $fps -i $(joinpath(dir, "%03d.png")) -vf $filter -loop 0 $path`)
    empty!(FRAMES)
    return path
end

"""Show the frames added since the last call as a looping GIF."""
function show_animation(; fps = 6, hold = 2)
    isempty(FRAMES) && return nothing
    width = FRAME_WIDTH[]
    gif = save_animation(joinpath(mktempdir(), "animation.gif"); fps, hold)
    display(MIME("text/html"), "<img src=\"data:image/gif;base64,$(base64encode(read(gif)))\" width=\"$width\">")
    return nothing
end

"""One frame of a walk downhill: the landscape, the current point `x`, and the slope there."""
function show_step(landscape, slope, x; title = "")
    xs = range(-6, 6; length = 400)
    s = slope(x)
    fig = Figure(; size = (720, 380))
    ax = Axis(fig[1, 1]; title = @sprintf("%s:  x = %.2f,  slope = %.2f", title, x, s), titlesize = 18,
        xlabel = "x", ylabel = "landscape(x)")
    lines!(ax, xs, landscape.(xs); color = :gray30, linewidth = 2)
    lines!(ax, [x - 0.8, x + 0.8], [landscape(x) - 0.8s, landscape(x) + 0.8s]; color = :orangered, linewidth = 3)
    scatter!(ax, [x], [landscape(x)]; color = :orangered, markersize = 16)
    ylims!(ax, -2.2, 5.5)
    frame!(fig)
    return nothing
end

"""Run `days` of ocean with the compiled time stepping and show the sea surface as a looping GIF."""
function run_ocean!(model; days = 25)
    path = joinpath(RUNS, "ocean.gif")
    steps, seconds = Ocean.step_and_record!(model, path; days_total = days)
    QUIET[] && return nothing
    display(MIME("text/html"), "<img src=\"data:image/gif;base64,$(base64encode(read(path)))\" width=\"640\">")
    @printf "%.0f days of ocean, %d time steps: %.1f s on one GPU\n" days steps seconds
    return nothing
end

# ---------------------------------------------------------------------------
# Black hole: Comrade compiled by Reactant, differentiated by Enzyme, sampled by Impulse
# ---------------------------------------------------------------------------

skyimage(θ) = intensitymap(skymodel(post, θ), gimg)
sky_images(samples) = [skyimage(Comrade.transform(tpost, r)) for r in eachrow(Array(samples))]

"""Every measurement of the night: one dot per pair of dishes and time, placed by spatial frequency."""
show_measurements(data) = plotfields(data, :U, :V; legend = false,
    axis_kwargs = (; xlabel = "spatial frequency, east–west", ylabel = "spatial frequency, north–south"),
    figure_kwargs = (; size = (640, 520)))

function blackhole_axis(pos; title = "")
    ax = Axis(pos; xreversed = true, aspect = 1, title, titlesize = 18)
    hidedecorations!(ax)
    return ax
end

function show_sky(x; title = "")
    fig = Figure(; size = (520, 520))
    image!(blackhole_axis(fig[1, 1]; title), skyimage(Comrade.transform(tpost, Array(x))); colormap = :afmhot)
    frame!(fig)
    return nothing
end

function show_sampling(imgs, seconds)
    fig = Figure(; size = (900, 500))
    image!(blackhole_axis(fig[1, 1]; title = "one posterior sample"), imgs[end]; colormap = :afmhot)
    image!(blackhole_axis(fig[1, 2]; title = "posterior mean so far"), mean(imgs); colormap = :afmhot)
    Label(fig[0, 1:2], @sprintf("%d posterior samples  (%.0f s on one GPU)", length(imgs), seconds); fontsize = 22)
    frame!(fig)
    return nothing
end

function opt_step(opt, post, x)
    res = Enzyme.gradient(Enzyme.ReverseWithPrimal, Const(logdensityof), Const(post), x)
    opt, x = Optimisers.update!(opt, x, -res.derivs[2])
    return opt, x, res.val
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

const SAMPLING = joinpath(RUNS, "sampling.jls")

"""Tune the Impulse sampler (once), run it `chunks` times on the GPU from the tuned state, and save
the images with the cumulative seconds after each call. The notebook replays this file."""
function precompute_sampling!(; chunks = 8)
    mkpath(RUNS)
    if !isfile(WARM_STATE)
        x = Reactant.to_rarray(X_START)
        opt_setup = @compile Optimisers.setup(AdamW(FIT_LR), x)
        opt = opt_setup(AdamW(FIT_LR), x)
        fit_step = @compile sync = true opt_step(opt, tpostr, x)
        for _ in 1:5000
            opt, x, _ = fit_step(opt, tpostr, x)
        end
        res = sample(
            Reactant.ReactantRNG(Reactant.to_rarray(UInt64[2026, 11])), postr,
            ReactantNUTS(; n_adapts = NADAPT), CHUNK;
            initial_params = Comrade.transform(tpost, Array(x)), chunk_size = CHUNK,
            warmup_callback = Returns(nothing),
        )
        ProbProg.save_state(WARM_STATE, res.state)
    end
    state = ProbProg.load_state(WARM_STATE)
    nuts = Reactant.Compiler.compile(nuts_chunk, (state, tpostr, CHUNK); optimize = :probprog, sync = true)
    nuts(state, tpostr, CHUNK)
    state = ProbProg.load_state(WARM_STATE)
    imgs, seconds, t = [], Float64[], 0.0
    for _ in 1:chunks
        t += @elapsed (samples, _, _, _, state) = nuts(state, tpostr, CHUNK)
        append!(imgs, sky_images(samples))
        push!(seconds, t)
    end
    serialize(SAMPLING, (; images = [Float32.(collect(parent(img))) for img in imgs], seconds))
    @assert isapprox(mean(first(load_sampling())), mean(imgs); rtol = 1e-5)
    return seconds
end

"""The images and cumulative seconds saved by `precompute_sampling!`."""
function load_sampling()
    saved = deserialize(SAMPLING)
    return [IntensityMap(Float64.(a), gimg) for a in saved.images], saved.seconds
end

"""Render every live figure once off screen, so plotting code is compiled before the talk."""
function warm_plots!(ocean_model, skymatch_case)
    QUIET[] = true
    run_ocean!(ocean_model; days = 0)
    Makie.colorbuffer(show_skymatch(skymatch_case...))
    show_sky(X_START)
    show_step(sin, cos, 0.0; title = "warm-up")
    save(joinpath(mktempdir(), "warm.png"), Figure())
    QUIET[] = false
    return nothing
end

function prepare_demo!()
    mkpath(RUNS)
    xr = Reactant.to_rarray(rand(Float32, 8))
    @jit sum(abs2, xr)

    xo = Reactant.to_rarray(X_START)
    global opt_setup = @compile Optimisers.setup(AdamW(FIT_LR), xo)
    opt = opt_setup(AdamW(FIT_LR), xo)
    global fit_step = @compile sync = true opt_step(opt, tpostr, xo)

    Enzyme.gradient(Reverse, x -> 0.1x^2 + sin(x) + 0.5sin(3x), 1.0)
    ocean_model = prepare_ocean!()
    skymatch_case = prepare_skymatch!()
    prepare_primates!()
    warm_plots!(ocean_model, skymatch_case)
    return nothing
end
