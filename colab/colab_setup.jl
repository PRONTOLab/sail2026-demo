# One-time setup of the SAIL 2026 demo on Google Colab (Julia runtime, GPU).
# Every source location can be overridden through an environment variable, which is how
# the Ubuntu 22.04 container test on hydra points it at local copies. While the demo repo
# is private, set ENV["SAIL_GITHUB_TOKEN"] to a GitHub token that can read it.

using Pkg, Downloads

const DEMO_SLUG = get(ENV, "SAIL_DEMO_SLUG", "PRONTOLab/sail2026-demo")
const DEMO_DIR = get(ENV, "SAIL_DEMO_DIR", "/content/sail2026-demo")
const REACTANT_URL = get(ENV, "SAIL_REACTANT_URL", "https://github.com/EnzymeAD/Reactant.jl.git")
const REACTANT_REV = get(ENV, "SAIL_REACTANT_REV", "impulse")
const LIB_URL = get(ENV, "SAIL_LIB_URL", "")
const LIB_ASSET = "libReactantExtra-sail2026.tar.gz"
const RELEASE_TAG = "v1"
const COMRADE_URL = "https://github.com/ptiede/Comrade.jl.git"
const TOKEN = get(ENV, "SAIL_GITHUB_TOKEN", "")

step(msg) = println("[", round(time() - T0; digits = 1), " s] ", msg)
const T0 = time()

auth_headers(accept) = isempty(TOKEN) ? ["Accept" => accept] :
    ["Accept" => accept, "Authorization" => "Bearer $TOKEN"]

"""Download the release asset through the GitHub API, which also works for private repos."""
function download_release_asset()
    api = "https://api.github.com/repos/$DEMO_SLUG/releases/tags/$RELEASE_TAG"
    meta = read(Downloads.download(api; headers = auth_headers("application/vnd.github+json")), String)
    block = match(Regex("\"url\":\\s*\"([^\"]+/releases/assets/\\d+)\"[^{}]*?\"name\":\\s*\"$(LIB_ASSET)\""), meta)
    block === nothing && error("release $RELEASE_TAG of $DEMO_SLUG has no asset $LIB_ASSET")
    return Downloads.download(String(block[1]); headers = auth_headers("application/octet-stream"))
end

if !isdir(DEMO_DIR)
    url = isempty(TOKEN) ? "https://github.com/$DEMO_SLUG.git" :
        "https://x-access-token:$TOKEN@github.com/$DEMO_SLUG.git"
    run(`git clone -q --depth 1 $url $DEMO_DIR`)
    run(`git -C $DEMO_DIR remote set-url origin https://github.com/$DEMO_SLUG.git`)
end
cd(DEMO_DIR)
step("demo files in $DEMO_DIR")

lib = joinpath(DEMO_DIR, "lib", "libReactantExtra.so")
if !isfile(lib)
    mkpath(dirname(lib))
    tarball = isempty(LIB_URL) ? download_release_asset() : Downloads.download(LIB_URL)
    run(`tar -xzf $tarball -C $(dirname(lib))`)
    rm(tarball)
end
step("research libReactantExtra ready ($(round(filesize(lib) / 2^30; digits = 2)) GiB)")

env = joinpath(DEMO_DIR, "env")
mkpath(env)
write(joinpath(env, "LocalPreferences.toml"), """
[Reactant_jll]
libReactantExtra_path = "$lib"
""")

comrade = joinpath(DEMO_DIR, "deps", "Comrade.jl")
if !isdir(comrade)
    run(`git clone -q --branch v0.11.33 --depth 1 $COMRADE_URL $comrade`)
    run(`git -C $comrade apply $(joinpath(DEMO_DIR, "colab", "comrade_sail2026.patch"))`)
end
step("Comrade v0.11.33 + patch")

Pkg.activate(env)
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
Pkg.add([
    PackageSpec(url = REACTANT_URL, rev = REACTANT_REV, subdir = "lib/ReactantCore"),
    PackageSpec(url = REACTANT_URL, rev = REACTANT_REV),
])
Pkg.develop(PackageSpec(path = comrade))
Pkg.add([
    PackageSpec(name = "Reactant_jll", version = "0.0.424"),
    PackageSpec(name = "VLBIFiles"),
    PackageSpec(name = "VLBIImagePriors", version = "0.10.16"),
    PackageSpec(name = "VLBISkyModels", version = "0.6.29"),
    PackageSpec(name = "Distributions"),
    PackageSpec(name = "StableRNGs"),
    PackageSpec(name = "Enzyme", version = "0.13.216"),
    PackageSpec(name = "Optimisers", version = "0.4.11"),
    PackageSpec(name = "CairoMakie", version = "0.15.16"),
    PackageSpec(name = "Oceananigans", version = "0.116.1"),
    PackageSpec(name = "CUDA", version = "6.2.2"),
    PackageSpec(name = "JSON"),
])
step("packages resolved")
include(joinpath(DEMO_DIR, "cuda_libs.jl"))
step("linked CUDA libraries from $(link_cuda_libs(DEMO_DIR))")
delete!(ENV, "JULIA_PKG_PRECOMPILE_AUTO")
Pkg.precompile()
step("packages precompiled; next: include(\"demo_utils.jl\"); prepare_demo!()")
