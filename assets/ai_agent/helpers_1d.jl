function plotTrajectory(u_hat::Vector{Float64}, o_hat::Vector{Float64})
    plot(1:N, o_hat, color="black", marker="o", linewidth=2, alpha=0.8) # Observed locations
    grid("on")
    xlabel("Time t")
    ylabel("Position o")
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
    println(algo[8:620])
    println("\n...\n")
    println(algo[end-89:end-13])
end