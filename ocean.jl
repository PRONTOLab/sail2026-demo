"""
Oceananigans, an ocean model written in Julia for NVIDIA GPUs, compiled by Reactant: a front
between warm and cold water breaks into eddies (Oceananigans' baroclinic adjustment setup, with
the GB-25 time-stepping recipe). The notebook builds the model in a visible cell with the same
code as `build_model`, which Setup uses to compile the time stepping once.
"""
module Ocean

using CUDA, Reactant, Oceananigans, CairoMakie, Printf
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
import Oceananigans.TimeSteppers: first_time_step!, time_step!

export prepare_ocean!

const NX, NY, NZ = 96, 96, 8
const STEPS_PER_FRAME = 50
const DT = 20minutes
const COMPILED = Dict{Symbol, Any}()

function build_model()
    grid = RectilinearGrid(ReactantState(); size = (96, 96, 8), halo = (6, 6, 3),
        x = (0, 1000kilometers), y = (-500kilometers, 500kilometers), z = (-1kilometers, 0),
        topology = (Periodic, Bounded, Bounded))
    model = HydrostaticFreeSurfaceModel(grid; coriolis = BetaPlane(latitude = -45),
        buoyancy = BuoyancyTracer(), tracers = :b,
        momentum_advection = WENOVectorInvariant(), tracer_advection = WENO(),
        free_surface = SplitExplicitFreeSurface(grid; substeps = 50))
    front(x, y, z) = 4e-6 * z + 5e-3 * clamp(y / 50kilometers + 1/2, 0, 1) + 5e-5 * randn()
    set!(model, b = front)
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
    model.clock.last_Δt = DT
    COMPILED[:first] = @compile sync = true raise = true first_step!(model)
    COMPILED[:loop] = @compile sync = true raise = true loop!(model, ConcreteRNumber(STEPS_PER_FRAME))
    return model
end

surface_temperature(model) = Array(interior(model.tracers.b, :, :, NZ))
days(model) = Reactant.to_number(model.clock.time) / day

"""Step `model` with the time stepping compiled during Setup, `$(STEPS_PER_FRAME)` steps per
frame, and record the sea surface after each frame into a looping GIF at `path`, holding the
first and last frames. Returns the number of time steps and the seconds spent stepping."""
function step_and_record!(model, path; days_total = 25)
    frames = round(Int, days_total * day / (STEPS_PER_FRAME * DT))
    model.clock.last_Δt = DT
    Reactant.to_number(model.clock.iteration) == 0 && COMPILED[:first](model)
    xs, ys = range(0, 1000; length = NX), range(-500, 500; length = NY)
    sst = Observable(surface_temperature(model))
    fig = Figure(; size = (640, 620))
    ax = Axis(fig[1, 1]; title = @sprintf("day %.1f", days(model)), titlesize = 20, aspect = 1,
        xlabel = "kilometers", ylabel = "kilometers")
    hm = heatmap!(ax, xs, ys, sst; colormap = :thermal, colorrange = (-0.0004, 0.0049))
    Colorbar(fig[1, 2], hm; label = "sea surface: cold → warm", ticksvisible = false, ticklabelsvisible = false)
    t = 0.0
    n = ConcreteRNumber(STEPS_PER_FRAME)
    CairoMakie.record(fig, path, 1:(frames + 12); framerate = 8) do i
        4 < i <= frames + 4 || return
        t += @elapsed COMPILED[:loop](model, n)
        sst[] = surface_temperature(model)
        ax.title = @sprintf("day %.1f", days(model))
    end
    return frames * STEPS_PER_FRAME, t
end

end
