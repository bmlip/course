function plotTrajectory(u_hat::Vector{Vector{Float64}}, o_hat::Vector{Vector{Float64}})
    ϕ = [o_hat_k[1] for o_hat_k in o_hat]
    x = [o_hat_k[2] for o_hat_k in o_hat]
    y = [o_hat_k[3] for o_hat_k in o_hat]

    Δϕ = [u_hat_k[1] for u_hat_k in u_hat]
    r  = [u_hat_k[2] for u_hat_k in u_hat]

    plot(x, y, color="black", marker="o", linewidth=2, alpha=0.8) # Observed locations
    for t=1:N
        arrow(x[t], y[t], 0.1*cos(ϕ[t]), 0.1*sin(ϕ[t]), color="green", width=0.02, head_width=0.1, alpha=0.8, zorder=1) # Observed orientation
    end

    ϕ_min = [ϕ_0; ϕ[1:end-1]] 
    Δx = r.*cos.(ϕ_min + Δϕ)
    Δy = r.*sin.(ϕ_min + Δϕ)
    
    for t=2:N
        arrow(x[t-1], y[t-1], Δx[t], Δy[t], color="red", width=0.02, head_width=0.1, alpha=0.8, zorder=0) # Proposed movement
    end

    scatter(0.0, 0.0, marker="*", color="orange", s=50)

    axis("square")
    grid("on")
    xlabel("x")
    ylabel("y")
end

"""
To find the index of the slide message in the schedule:
schedule = sumProductSchedule(u[2])
slide_msg_idx = findMessageIndex(schedule, :equ_s_1_1, 3)
"""
function findMessageIndex(schedule::Schedule, node_id::Symbol, outbound_interface_id::Union{Int64, Symbol})
    condensed_schedule = ForneyLab.condense(ForneyLab.flatten(schedule))
    for (idx, entry) in enumerate(condensed_schedule)
        node = entry.interface.node
        if (node.id == node_id) && (entry.interface == node.i[outbound_interface_id])
            return idx
        end
    end
    
    error("Message $node_id[$outbound_interface_id] could not be found in schedule")
end

function inspectSnippet(algo::String)
    println(algo[8:625])
    println("\n...\n")
    println(algo[end-89:end-13])
end