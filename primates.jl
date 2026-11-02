"""
Primate brain size, model m14.10 of McElreath's *Statistical Rethinking*, written the way the
textbook writes it, compiled with Impulse's compiler optimizations off and on. Runs on the CPU
client, while the rest of the demo stays on the GPU.
"""
module Primates

using Reactant, JSON, Statistics, Printf, LinearAlgebra, CairoMakie
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber

export prepare_primates!, show_family_tree_matrix, sample_textbook_model, race_figure, show_answer, framework_bars

const DATA = JSON.parsefile(joinpath(@__DIR__, "data", "brownian_primates_n151.json"))
const N_SPECIES = Int(DATA["attrs"]["n"])
const R = [Float64(DATA["R"][i][j]) for i in 1:N_SPECIES, j in 1:N_SPECIES]
const M = [Float64(DATA["X"][i][2]) for i in 1:N_SPECIES]
const G = [Float64(DATA["X"][i][3]) for i in 1:N_SPECIES]
const B = Float64.(DATA["Y"])

const Normal = ProbProg.Normal
const Exponential = ProbProg.Exponential
MvNormal(μ, Σ) = ProbProg.MultivariateNormal(μ, Σ, (length(μ),))

is_tilde(ex) = Meta.isexpr(ex, :call) && length(ex.args) == 3 && ex.args[1] === :~

"""
`@model function f(args...) ... end` with Turing-style `x ~ D` statements: each becomes an
Impulse sample site named `:x`, and `f` gains a leading `rng` argument.
"""
macro model(fdef)
    sig, body = fdef.args
    stmts = Any[]
    last_site = nothing
    for ex in body.args
        if is_tilde(ex)
            lhs, rhs = ex.args[2], ex.args[3]
            push!(stmts, :((_, $lhs) = ProbProg.sample(rng, $rhs; symbol=$(QuoteNode(lhs)))))
            last_site = lhs
        else
            push!(stmts, ex)
        end
    end
    last_site === nothing || push!(stmts, :(return $last_site))
    call = Expr(:call, sig.args[1], :rng, sig.args[2:end]...)
    return esc(Expr(:function, call, Expr(:block, stmts...)))
end

@model function primates(M, G, R)
    a ~ Normal(0.0, 1.0)
    bM ~ Normal(0.0, 0.5)
    bG ~ Normal(0.0, 0.5)
    sigma_sq ~ Exponential(1.0)
    mu = a .+ bM .* M .+ bG .* G
    SIGMA = R .* sigma_sq
    B ~ MvNormal(mu, SIGMA)
end

function nuts_program(rng, model, model_args, step_size, num_warmup, num_samples,
                      constraint, constrained_addresses, selection, inverse_mass_matrix)
    t, _, _ = ProbProg.generate(rng, constraint, model, model_args...; constrained_addresses)
    t, diagnostics, _, _, _ = ProbProg.mcmc(
        rng, t, model, model_args...;
        selection, algorithm = :NUTS, inverse_mass_matrix, step_size, max_tree_depth = 10,
        num_warmup, num_samples, adapt_step_size = true, adapt_mass_matrix = true,
    )
    return t, diagnostics
end

const SELECTION = ProbProg.select(
    ProbProg.Address(:a), ProbProg.Address(:bM), ProbProg.Address(:bG), ProbProg.Address(:sigma_sq),
)
const NUM_WARMUP = 1000
const NUM_SAMPLES = 3000
const MODEL_ARGS = Ref{Any}(nothing)
const PROGRAMS = Dict{Symbol, Any}()
const CPU_CLIENT = Ref{Any}(nothing)

function with_cpu(f)
    prev = Reactant.XLA.default_backend()
    CPU_CLIENT[] === nothing && (CPU_CLIENT[] = Reactant.XLA.client("cpu"))
    Reactant.set_default_backend(CPU_CLIENT[])
    try
        return f()
    finally
        Reactant.set_default_backend(prev)
    end
end

function constraint_inputs()
    init = DATA["init_params"]
    c = ProbProg.Constraint(
        :a => Float64.(init["a"]), :bM => Float64.(init["b"][1:1]), :bG => Float64.(init["b"][2:2]),
        :sigma_sq => Float64.(init["sigma_sq"]), :B => B,
    )
    addrs = ProbProg.extract_addresses(c)
    flat = reduce(vcat, [vec(c[a]) for a in addrs])
    return Reactant.to_rarray(reshape(flat, 1, :)), addrs
end

function compile_primates(; optimize_model::Bool)
    return with_cpu() do
        MODEL_ARGS[] === nothing &&
            (MODEL_ARGS[] = (Reactant.to_rarray(M), Reactant.to_rarray(G), Reactant.to_rarray(R)))
        flag = Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[]
        Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = !optimize_model
        try
            ct, addrs = constraint_inputs()
            fn, tt = ProbProg.with_trace() do
                @compile sync = true optimize = :probprog nuts_program(
                    ReactantRNG(Reactant.to_rarray(UInt64[1, 0])), primates, MODEL_ARGS[],
                    ConcreteRNumber(0.1), NUM_WARMUP, NUM_SAMPLES, ct, addrs, SELECTION,
                    Reactant.to_rarray(Matrix{Float64}(I, 4, 4)),
                )
            end
            (; fn, entries = ProbProg.filter_entries_by_selection(tt.entries, SELECTION))
        finally
            Reactant.Compiler.DEBUG_PROBPROG_DISABLE_OPT[] = flag
        end
    end
end

function run_primates(prog; seed = 2026)
    return with_cpu() do
        ct, addrs = constraint_inputs()
        elapsed = @elapsed begin
            t, _ = prog.fn(
                ReactantRNG(Reactant.to_rarray(UInt64[seed, 0])), primates, MODEL_ARGS[],
                ConcreteRNumber(0.1), NUM_WARMUP, NUM_SAMPLES, ct, addrs, SELECTION,
                Reactant.to_rarray(Matrix{Float64}(I, 4, 4)),
            )
        end
        tr = ProbProg.unflatten_trace(Array(t), 0.0, prog.entries, nothing)
        draws = Dict(k => vec(Float64.(tr.choices[k])) for k in (:a, :bM, :bG, :sigma_sq))
        (draws, elapsed)
    end
end

"""Compile the textbook model with compiler optimizations off and on, run each once, and
render every figure once off screen."""
function prepare_primates!()
    warm = Dict{Symbol, Any}()
    for (key, opt) in ((:off, false), (:on, true))
        PROGRAMS[key] = compile_primates(; optimize_model = opt)
        warm[key], _ = run_primates(PROGRAMS[key])
    end
    foreach(Makie.colorbuffer, (show_family_tree_matrix(), race_figure(1.0, 0.1),
        show_answer(warm[:off], warm[:on]), framework_bars()))
    return nothing
end

const TREE = JSON.parsefile(joinpath(@__DIR__, "data", "primates_tree_order.json"))
const ICON_DIR = joinpath(@__DIR__, "assets", "phylopic")
# Okabe-Ito colors, one per primate group.
const GROUP_COLOR = Dict("galago" => "#E69F00", "lemur" => "#CC79A7", "tarsier" => "#56B4E9",
    "ateles" => "#009E73", "gorilla" => "#A6761D", "macaca" => "#0072B2")
# Icon centers (rows): block midpoints, nudged apart around the three-row tarsier block.
const ICON_ROW = Dict("galago" => 7.5, "lemur" => 23.5, "tarsier" => 46.5, "ateles" => 66.0,
    "gorilla" => 87.5, "macaca" => 122.5)

"""A PhyloPic silhouette (CC0 or public domain, see assets/phylopic) in one color, rows top to bottom."""
function silhouette(name, color; stride = 8)
    c = Makie.to_color(color)
    img = Makie.FileIO.load(joinpath(ICON_DIR, "$name.png"))[1:stride:end, 1:stride:end]
    return [RGBAf(c.r, c.g, c.b, Makie.Colors.alpha(px)) for px in img]
end

"""The relatedness matrix in evolutionary-tree order, with each primate group outlined and labeled."""
function show_family_tree_matrix()
    o = Int.(TREE["order"])
    n = N_SPECIES
    fig = Figure(; size = (780, 560))
    ax = Axis(fig[1, 1]; title = "How related each pair of the 151 species is", titlesize = 18,
        aspect = DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    hm = heatmap!(ax, 1:n, 1:n, R[o, o]; colormap = :Greys, colorrange = (0, 1))
    box(a, b) = ([a, b, b, a, a], [a, a, b, b, a])
    lines!(ax, box(0.5, n + 0.5)...; color = :black, linewidth = 1)
    for g in TREE["clades"]
        first, last, name, color = g["first"], g["last"], g["icon"], GROUP_COLOR[g["icon"]]
        lines!(ax, box(first - 0.5, last + 0.5)...; color, linewidth = 2)
        lines!(ax, [-3.0, -3.0], [first - 0.2, last + 0.2]; color, linewidth = 3)
        img = silhouette(name, color)
        h = 13.0
        w = h * size(img, 2) / size(img, 1)
        y = ICON_ROW[name]
        image!(ax, (-6.0 - w)..(-6.0), (y - h / 2)..(y + h / 2), permutedims(img))
        text!(ax, -8.0 - w, y; text = g["label"], align = (:right, :center), fontsize = 13, color)
    end
    xlims!(ax, -78, n + 0.5)
    ylims!(ax, n + 0.5, 0.5)
    Colorbar(fig[1, 2], hm; label = "shared evolutionary history", height = Relative(0.75))
    return fig
end

"""NUTS on the textbook model (1,000 warm-up + 3,000 draws) with the program Setup compiled
with Impulse's compiler optimizations off or on. Returns the posterior draws."""
sample_textbook_model(; optimizations::Bool) = first(run_primates(PROGRAMS[optimizations ? :on : :off]))

function race_figure(t_off, t_on)
    fig = Figure(; size = (760, 260))
    ax = Axis(fig[1, 1]; xlabel = "seconds", yticks = (1:2, ["optimizations ON", "optimizations OFF"]),
        title = "Same textbook model, same answer", titlesize = 18)
    barplot!(ax, [1, 2], [t_on, t_off]; direction = :x,
        color = [Makie.wong_colors()[3], Makie.wong_colors()[6]],
        bar_labels = [@sprintf("%.2f s", t_on), @sprintf("%.2f s", t_off)])
    xlims!(ax, 0, 1.25 * t_off)
    return fig
end

function show_answer(draws_off, draws_on)
    fig = Figure(; size = (900, 400))
    ax = Axis(fig[1, 1]; title = "Same answer with optimizations off and on", titlesize = 18,
        xlabel = "effect of group size on brain size", yticksvisible = false, yticklabelsvisible = false)
    density!(ax, draws_off[:bG]; color = (Makie.wong_colors()[6], 0.45), label = "optimizations off")
    density!(ax, draws_on[:bG]; color = (Makie.wong_colors()[3], 0.45), label = "optimizations on")
    axislegend(ax; position = :rt)
    return fig
end

"""The textbook model in each framework, 1,000 warm-up + 3,000 draws, one CPU core."""
function framework_bars()
    names = ["Impulse", "NumPyro", "Stan", "Turing"]
    speedup = [1.0, 47.8, 101.0, 155.7]
    fig = Figure(; size = (820, 300))
    ax = Axis(fig[1, 1]; xscale = log10, xlabel = "run time relative to Impulse (log scale)",
        yticks = (1:4, reverse(names)), title = "Same textbook model in each framework (one CPU core)", titlesize = 18)
    barplot!(ax, 1:4, reverse(speedup); direction = :x, fillto = 0.5,
        color = reverse([Makie.wong_colors()[3], fill(:gray60, 3)...]),
        bar_labels = reverse(["1×", "48×", "101×", "156×"]))
    xlims!(ax, 0.5, 600)
    return fig
end

end
