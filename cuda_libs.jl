# The research libReactantExtra.so loads cuDNN, NVRTC and NCCL from $ORIGIN/../nvidia/<pkg>/lib
# (the Bazel/pip layout baked into its RUNPATH). NVRTC and NCCL come from the released
# Reactant_jll CUDA artifact. cuDNN must be at least the version the library was built against
# (9.21), newer than the artifact's, so it comes from NVIDIA's PyPI wheel, which has that layout.

using Downloads, p7zip_jll

const CUDA_LIB_GROUPS = ["cuda_nvrtc" => r"^libnvrtc", "nccl" => r"^libnccl"]
const CUDA_LIBS_NEEDED = ["libnvrtc.so.12", "libnvrtc-builtins.so.12.9", "libnccl.so.2"]
const CUDNN_WHEEL = "https://files.pythonhosted.org/packages/5c/cf/47778414dd633ba93395f9d34c87283916c3163f57000c2aeef1f869c649/nvidia_cudnn_cu12-9.21.0.82-py3-none-manylinux_2_27_x86_64.whl"

"""Newest Reactant_jll CUDA artifact in the depot that has every library we need."""
function find_cuda_artifact()
    roots = filter(isdir, [joinpath(d, "artifacts") for d in DEPOT_PATH])
    dirs = [joinpath(r, a, "lib") for r in roots for a in readdir(r)]
    ok = filter(d -> isfile(joinpath(d, "libReactantExtra.so")) &&
        all(f -> isfile(joinpath(d, f)), CUDA_LIBS_NEEDED), dirs)
    isempty(ok) && error("no Reactant_jll CUDA artifact with $(CUDA_LIBS_NEEDED) in $(roots)")
    return argmax(mtime, ok)
end

"""Install cuDNN 9.21 from its wheel into `nvidia/cudnn/lib`, unless it is already there."""
function install_cudnn(demo_dir)
    dst = joinpath(demo_dir, "nvidia", "cudnn", "lib")
    lib = joinpath(dst, "libcudnn.so.9")
    isfile(lib) && !islink(lib) && return dst
    rm(dst; force = true, recursive = true)
    whl = Downloads.download(CUDNN_WHEEL)
    tmp = mktempdir()
    run(`$(p7zip()) x -y -bso0 -bsp0 -o$tmp $whl nvidia/cudnn/lib`)
    mkpath(dirname(dst))
    mv(joinpath(tmp, "nvidia", "cudnn", "lib"), dst)
    rm(whl)
    return dst
end

function link_cuda_libs(demo_dir)
    src = find_cuda_artifact()
    for (group, pattern) in CUDA_LIB_GROUPS
        dst = mkpath(joinpath(demo_dir, "nvidia", group, "lib"))
        for f in filter(f -> occursin(pattern, f), readdir(src))
            link = joinpath(dst, f)
            islink(link) && rm(link)
            symlink(joinpath(src, f), link)
        end
    end
    install_cudnn(demo_dir)
    return src
end
