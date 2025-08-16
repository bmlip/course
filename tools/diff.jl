source = joinpath(@__DIR__, "..")


function tryexpanduser(path)
	try
		expanduser(path)
	catch ex
		path
	end
end

tamepath = abspath ∘ tryexpanduser

pluto_state_cache = tamepath("pluto_state_cache")
gh_pages_dir = tamepath("gh_pages_dir")

if !isdir(pluto_state_cache) || readdir(pluto_state_cache) == []
    @warn "Running this script without a GitHub Actions PlutoSliderServer cache. Using a local cache in the temp dir."
    pluto_state_cache = joinpath(tempdir(), "bmlip cache")
end

cd(source)

using Pkg
Pkg.activate(joinpath(source, "pluto-slider-server-environment"))
Pkg.instantiate()

using PlutoNotebookComparison

include(joinpath(@__DIR__, "DramaBrokenLink.jl"))


sources_old = [
    PSSCache(pluto_state_cache)
    WebsiteDir(gh_pages_dir)
    WebsiteAddress("https://bmlip.github.io/course/")
    SafePreview()
]

sources_new = [
    PSSCache(pluto_state_cache)
    RunWithPlutoSliderServer(; Export_cache_dir=pluto_state_cache)
]

drama_checkers = [
    DramaBrokenLink()
    DramaRestartRequired()
    DramaNewError()
    DramaBrokenImport()
]

PlutoNotebookComparison.compare_PR(source;
    sources_old,
    sources_new,
    drama_checkers,
)




