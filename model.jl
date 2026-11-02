using Comrade, VLBIFiles, VLBIImagePriors, VLBISkyModels, Distributions
using LinearAlgebra, StableRNGs
using Reactant, Enzyme

const ReactantEx = Comrade.ComradeBase.ReactantEx

uvd = VLBIFiles.load(
    VLBIFiles.UVData,
    joinpath(@__DIR__, "data", "SR1_M87_2017_096_lo_hops_netcal_StokesI.uvfits"),
)
dvis = extract_table(uvd, Visibilities(; time_average = VLBI.GapBasedScans()))
add_fractional_noise!(dvis, 0.02)

@sky function sky(grid; ftot, mimg, cprior)
    c ~ cprior
    σimg ~ VLBITruncated(VLBIGaussian(0.0, 0.5); lower = 0.0)
    fg ~ VLBIUniform(0.0, 1.0)
    rast = apply_fluctuations(CenteredLR(), mimg, σimg .* c.params)
    pimg = parent(rast)
    @. pimg = (ftot * (1 - fg)) * pimg
    m = ContinuousImage(rast, BSplinePulse{3}())
    g = modify(Gaussian(), Stretch(μas2rad(500.0), μas2rad(500.0)), Renormalize(ftot * fg))
    x, y = centroid(m)
    return shifted(m, -x, -y) + g
end

npix = 48
fovx = μas2rad(200.0)
fovy = μas2rad(200.0)
grid = imagepixels(fovx, fovy, npix, npix)
fwhmfac = 2 * sqrt(2 * log(2))
mpr = modify(Gaussian(), Stretch(μas2rad(60.0) ./ fwhmfac))
mimg_raw = intensitymap(mpr, grid)
mimg = mimg_raw ./ Comrade.flux(mimg_raw)
cprior = corr_image_prior(grid, dvis)
skym = sky(grid; ftot = 1.1, mimg, cprior)

@instrument function instrument()
    return @jones begin
        lg ~ ArrayPrior(IIDSitePrior(IntegSeg(), VLBIGaussian(0.0, 0.2)); LM = IIDSitePrior(IntegSeg(), VLBIGaussian(0.0, 1.0)))
        gp ~ ArrayPrior(IIDSitePrior(IntegSeg(), DiagonalVonMises(0.0, inv(π^2))); refant = SEFDReference(0.0), phase = true)
        return SingleStokesGain(exp(complex(lg, gp)))
    end
end
intmodel = instrument()

post = VLBIPosterior(skym, intmodel, dvis)
tpost = asflat(post)
postr = Comrade.prepare_device(VLBIPosterior(skym, intmodel, dvis; admode = nothing), ReactantEx())
tpostr = asflat(postr)
gimg = imagepixels(fovx, fovy, 128, 128)
