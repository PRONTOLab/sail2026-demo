"""SkyMatch's star-formation model and its luminosity response to scatter, on a fixed synthetic catalog."""
module SkyMatchDemo

using SkyMatch, Enzyme, CairoMakie, StableRNGs, Random

export prepare_skymatch!, show_skymatch

const CASE = Ref{Any}(nothing)

function luminosities(theta, galaxies, params, noise, sfr2lir)
    varied = merge(params, (sigma_MS = sum(theta),))
    return sfr2lir .* SkyMatch.generate_sfr(galaxies.redshift, galaxies.mstar, varied, noise).SFR
end

"""Prepare 1,024 synthetic galaxies and differentiate with fixed random draws, on the CPU."""
function prepare_skymatch!()
    raw = SkyMatch.load_par_file(joinpath(pkgdir(SkyMatch), "data", "SIDES_from_original.par"))
    params = SkyMatch.parse_sfr_numeric_params(raw)
    rng, n = StableRNG(2026), 1024
    galaxies = (x = 4.8 .* rand(rng, n), y = 3.0 .* rand(rng, n),
        redshift = 0.1 .+ 3.9 .* rand(rng, n), mstar = 10.0 .^ (9.0 .+ 2.5 .* rand(rng, n)))
    draws = SkyMatch.make_simulation_noise(n; seed = 1234)
    noise = (; draws.q_uniform, draws.sb_uniform, draws.sfr_uniform)
    args = (galaxies, params, noise, Float64(raw["SFR2LIR"]))
    theta = [params.sigma_MS]
    lir = luminosities(theta, args...)
    derivative = Enzyme.autodiff(Forward, luminosities, Duplicated,
        Duplicated(theta, ones(1)), Const.(args)...)[1]

    # Check the derivative before the live demo; the same draws are used at each step.
    h = 1.0e-4
    fd = (luminosities(theta .+ h, args...) .- luminosities(theta .- h, args...)) ./ (2h)
    @assert all(isfinite, derivative)
    @assert all(abs.(derivative .- fd) .<= 2e-5 .* abs.(fd) .+ 1e-7 .* max.(lir, 1.0))
    CASE[] = (; galaxies, lir, derivative)
    return nothing
end

"""Show intrinsic luminosities and the local response to 0.01 dex more SFR scatter."""
function show_skymatch()
    case = CASE[]
    active = case.lir .> 0
    x, y = case.galaxies.x[active], case.galaxies.y[active]
    # 100 * (dL/dsigma * 0.01 dex) / L: the predicted percentage change.
    response = case.derivative[active] ./ case.lir[active]
    limit = maximum(abs, response)
    fig = Figure(; size = (1000, 480))
    ax1 = Axis(fig[1, 1]; title = "Infrared luminosity", xlabel = "sky offset (arcmin)",
        ylabel = "sky offset (arcmin)", aspect = DataAspect())
    ax2 = Axis(fig[1, 2]; title = "Response to more scatter", xlabel = "sky offset (arcmin)",
        ylabel = "sky offset (arcmin)", aspect = DataAspect())
    p1 = scatter!(ax1, x, y; color = log10.(case.lir[active]), colormap = :inferno, markersize = 6)
    p2 = scatter!(ax2, x, y; color = response, colormap = :balance,
        colorrange = (-limit, limit), markersize = 6)
    Colorbar(fig[2, 1], p1; vertical = false, label = "log10(luminosity / solar luminosity)")
    Colorbar(fig[2, 2], p2; vertical = false, label = "% change for +0.01 dex scatter (linear prediction)")
    Label(fig[0, 1:2], "SkyMatch: synthetic galaxies, fixed random draws"; fontsize = 22)
    return fig
end

end
