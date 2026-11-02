"""SkyMatch's star-formation model and its luminosity response to scatter, on a fixed synthetic catalog."""
module SkyMatchDemo

using SkyMatch, Enzyme, CairoMakie, StableRNGs, Random

export synthetic_galaxies, luminosities, prepare_skymatch!, show_skymatch

function luminosities(theta, galaxies, params, noise, sfr2lir)
    varied = merge(params, (sigma_MS = sum(theta),))
    return sfr2lir .* SkyMatch.generate_sfr(galaxies.redshift, galaxies.mstar, varied, noise).SFR
end

"""`n` synthetic galaxies with SkyMatch's model parameters and fixed random draws."""
function synthetic_galaxies(n)
    raw = SkyMatch.load_par_file(joinpath(pkgdir(SkyMatch), "data", "SIDES_from_original.par"))
    params = SkyMatch.parse_sfr_numeric_params(raw)
    rng = StableRNG(2026)
    galaxies = (x = 4.8 .* rand(rng, n), y = 3.0 .* rand(rng, n),
        redshift = 0.1 .+ 3.9 .* rand(rng, n), mstar = 10.0 .^ (9.0 .+ 2.5 .* rand(rng, n)))
    draws = SkyMatch.make_simulation_noise(n; seed = 1234)
    noise = (; draws.q_uniform, draws.sb_uniform, draws.sfr_uniform)
    return galaxies, params, noise, Float64(raw["SFR2LIR"])
end

"""Differentiate once before the live demo and check the derivative against finite differences."""
function prepare_skymatch!()
    args = synthetic_galaxies(1024)
    theta = [args[2].sigma_MS]
    lir = luminosities(theta, args...)
    derivative = Enzyme.autodiff(Forward, luminosities, Duplicated,
        Duplicated(theta, [1.0]), Const.(args)...)[1]
    h = 1.0e-4
    fd = (luminosities(theta .+ h, args...) .- luminosities(theta .- h, args...)) ./ (2h)
    @assert all(isfinite, derivative)
    @assert all(abs.(derivative .- fd) .<= 2e-5 .* abs.(fd) .+ 1e-7 .* max.(lir, 1.0))
    return args[1], lir, derivative
end

"""Show intrinsic luminosities and the local response to 0.01 dex more SFR scatter."""
function show_skymatch(galaxies, lir, derivative)
    active = lir .> 0
    x, y = galaxies.x[active], galaxies.y[active]
    # 100 * (dL/dsigma * 0.01 dex) / L: the predicted percentage change.
    response = derivative[active] ./ lir[active]
    limit = maximum(abs, response)
    fig = Figure(; size = (1000, 480))
    ax1 = Axis(fig[1, 1]; title = "Brightness", xlabel = "position on the sky",
        ylabel = "position on the sky", aspect = DataAspect())
    ax2 = Axis(fig[1, 2]; title = "Sensitivity to the parameter", xlabel = "position on the sky",
        ylabel = "position on the sky", aspect = DataAspect())
    p1 = scatter!(ax1, x, y; color = log10.(lir[active]), colormap = :inferno, markersize = 6)
    p2 = scatter!(ax2, x, y; color = response, colormap = :balance,
        colorrange = (-limit, limit), markersize = 6)
    Colorbar(fig[2, 1], p1; vertical = false, label = "brightness (log scale)")
    Colorbar(fig[2, 2], p2; vertical = false, label = "% change in brightness")
    Label(fig[0, 1:2], "SkyMatch: 1,024 galaxies"; fontsize = 22)
    return fig
end

end
