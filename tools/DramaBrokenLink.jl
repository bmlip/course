

using Memoization
using URIs
import HTTP


# Set headers to mimic a real browser
const mock_browser_headers = [
    "User-Agent" => "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept" => "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language" => "en-US,en;q=0.9",
    "Accept-Encoding" => "gzip, deflate, br",
    "DNT" => "1",
    "Connection" => "keep-alive",
    "Upgrade-Insecure-Requests" => "1",
]


const impossible_domains = [
    "research.tue.nl",
    "doi.org",
]


struct DramaBrokenLink <: AbstractDrama end


"""
Check if a url is valid.

URLs can come in three types:

Some examples:

```
http://mathworld.wolfram.com/ModifiedBesselFunctionoftheSecondKind.html
https://en.wikipedia.org/wiki/Central_limit_theorem
https://en.wikipedia.org/wiki/Differential_entropy#Maximization_in_the_normal_distribution
http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf#page=250
https://en.wikipedia.org/wiki/Central_limit_theorem
https://reactivebayes.github.io/RxInfer.jl/stable/examples/basic_examples/Bayesian%20Linear%20Regression%20Tutorial/
https://github.com/bmlip/course/blob/main/assets/files/Jaynes-1990-straight-line-fitting-a-Bayesian-solution.pdf
https://bmlip.github.io/course/lectures/The%20Gaussian%20Distribution.html#natural-parameterization
#change-of-variable-derivation
#predictive-distribution
#matrix-calculus
```



The types:
- A web URL, like https://wikipedia.org/wiki/Central_limit_theorem
- No URL (just a fragment), like #change-of-variable-derivation
- A URL pointing to this course website. Like https://bmlip.github.io/course/lectures/The%20Gaussian%20Distribution.html#natural-parameterization In this case, we don't check the internet, but we check locally if that path will exist, given the new directory structure.

"""
@memoize function check_url(s::String)
    without_fragment = URI(s; fragment=@view("absent"[1:0]))
    
    if without_fragment.host in impossible_domains
        @info "The domain $(without_fragment.host) cannot be checked, skipping."
    elseif without_fragment.host == "bmlip.github.io"
        path_web = without_fragment.path
        path = replace(URIs.unescapeuri(path_web), r".html$" => ".jl")
        
        paths = split(path, "/")
        if length(paths) >= 3
            subpath = paths[3:end]
            
            source_path = normpath(joinpath(source, subpath...))
            @assert isfile(source_path) "The URL does not resolve:\n\n\t$s\n\ndoes not resolve. It refers to the local notebook file\n\n\t$source_path\n\nwhich was not found."
        end
    elseif without_fragment.host == ""
        if without_fragment.uri == ""
            # relative hash link, TODO
        else
            error("Checking relative URLs is not yet implemented. But Fons recommends to not use relative URLs, as they do not work consistently in all situations. The URL was: $s")
        end
    else
        response = HTTP.head(without_fragment; status_exception=false, cookies=false, headers=mock_browser_headers)
        if response.status ∉ 200:299
            # oh no!
            # let's check one last thing before erroring
            # this might be a "CloudFlare challenge", a sort of captcha that you need to fill in before you are allowed to access the page.
            response = HTTP.get(without_fragment; status_exception=false, headers=mock_browser_headers)
            is_challenge = occursin("challenge", String(response.body))
            @assert response.status ∈ 200:299 || is_challenge "The URL does not resolve:\n\n\t$s\n\nThe server responded with status $(response.status)."
        end
    end
end





const _htmlunescape_chars = Dict(
    "&lt;" => '<',
    "&gt;" => '>',
    "&quot;" => '"',
    "&amp;" => '&',
    "&#39;" => '\'',
)

function htmlunescape(s::AbstractString)
    # First replace &amp; that are part of other entities
    s = replace(s, "&amp;" => "&")
    
    # Then replace all known entities
    for (entity, char) in _htmlunescape_chars
        s = replace(s, entity => char)
    end
    
    # Handle numeric entities (&#34; or &#x22;)
    
    pat1 = r"&#(\d+);"
    pat2 = r"&#x([0-9a-fA-F]+);"
    
    s = replace(s, pat1 => m -> Char(parse(Int, match(pat1, m).captures[1])))
    s = replace(s, pat2 => m -> Char(parse(Int, match(pat2, m).captures[1], base=16)))
    
    return s
end





# you can test me by setting 
function PlutoNotebookComparison.check_drama(::DramaBrokenLink, di::DramaContext)
    tocheck = String[]
	for (cell_id, cell) in di.new_state["cell_results"]
		b = string(cell["output"]["body"])

		href_pattern = r"<a\s+[^>]*?href=['\"](.*?)['\"]"
	    for m in eachmatch(href_pattern, b)
	        href = m.captures[1]
            href_unescaped = htmlunescape(href)
            push!(tocheck, href_unescaped)
		end
	end
    
    Threads.@threads for u in tocheck
        check_url(u)
    end
end


PlutoNotebookComparison.should_check_drama(::DramaBrokenLink, di::DramaContext) = true


