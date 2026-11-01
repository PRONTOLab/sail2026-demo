"""
Oceananigans, an ocean model written in Julia for NVIDIA GPUs, compiled by Reactant: a front
between warm and cold water breaks into eddies (Oceananigans' baroclinic adjustment setup, with
the GB-25 time-stepping recipe).
"""
module Ocean

using CUDA, Reactant, Oceananigans, CairoMakie, Printf, Random
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
import Oceananigans.TimeSteppers: first_time_step!, time_step!

export prepare_ocean!, run_ocean!

const NX, NY, NZ = 96, 96, 8
const STEPS_PER_FRAME = 50
const DT = 20minutes
const MODEL = Ref{Any}(nothing)
const COMPILED = Dict{Symbol, Any}()

function build_model()
    grid = RectilinearGrid(ReactantState(); size = (NX, NY, NZ), halo = (6, 6, 3),
        x = (0, 1000kilometers), y = (-500kilometers, 500kilometers), z = (-1kilometers, 0),
        topology = (Periodic, Bounded, Bounded))
    model = HydrostaticFreeSurfaceModel(grid; coriolis = BetaPlane(latitude = -45),
        buoyancy = BuoyancyTracer(), tracers = :b,
        momentum_advection = WENOVectorInvariant(), tracer_advection = WENO(),
        free_surface = SplitExplicitFreeSurface(grid; substeps = 50))
    N², M², Δy = 4e-6, 1e-7, 50kilometers
    ramp(y) = min(max(0, y / Δy + 1 / 2), 1)
    Random.seed!(2026)
    set!(model, b = (x, y, z) -> N² * z + M² * Δy * ramp(y) + 1e-2 * M² * Δy * randn())
    model.clock.last_Δt = DT
    return model
end

function first_step!(model)
    first_time_step!(model, model.clock.last_Δt)
    return nothing
end

function loop!(model, n)
    Δt = model.clock.last_Δt + 0
    @trace track_numbers = false for _ in 1:n
        time_step!(model, Δt)
    end
    return nothing
end

"""Build the ocean model on Reactant and compile its time stepping."""
function prepare_ocean!()
    model = build_model()
    COMPILED[:first] = @compile sync = true raise = true first_step!(model)
    COMPILED[:loop] = @compile sync = true raise = true loop!(model, ConcreteRNumber(STEPS_PER_FRAME))
    MODEL[] = model
    return nothing
end

surface_temperature(model) = Array(interior(model.tracers.b, :, :, NZ))
days(model) = Reactant.to_number(model.clock.time) / day

"""Run the compiled ocean model for `frames × $(STEPS_PER_FRAME)` time steps, showing the sea
surface after each frame with `show(fig)`, then build a fresh model for the next run (~10 s,
reusing the compiled time stepping)."""
function run_ocean!(show; frames = 24)
    model = MODEL[]
    Reactant.to_number(model.clock.iteration) == 0 && COMPILED[:first](model)
    xs, ys = range(0, 1000; length = NX), range(-500, 500; length = NY)
    sst = Observable(surface_temperature(model))
    fig = Figure(; size = (640, 620))
    ax = Axis(fig[1, 1]; title = @sprintf("day %.1f", days(model)), titlesize = 20, aspect = 1,
        xlabel = "kilometers", ylabel = "kilometers")
    hm = heatmap!(ax, xs, ys, sst; colormap = :thermal, colorrange = (-0.0004, 0.0049))
    Colorbar(fig[1, 2], hm; label = "sea surface: cold → warm", ticksvisible = false, ticklabelsvisible = false)
    show(fig)
    t = 0.0
    n = ConcreteRNumber(STEPS_PER_FRAME)
    for _ in 1:frames
        t += @elapsed COMPILED[:loop](model, n)
        sst[] = surface_temperature(model)
        ax.title = @sprintf("day %.1f", days(model))
        show(fig)
    end
    frames > 0 || return nothing
    @printf "%.0f days of ocean, %d time steps: %.1f s on one GPU\n" frames * STEPS_PER_FRAME * DT / day frames * STEPS_PER_FRAME t
    MODEL[] = build_model()
    return nothing
end

end
