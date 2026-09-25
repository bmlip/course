function initializeWorld()
    Δϕ_lim = 0.25*pi # Turn rate limit per time unit
    r_lim = 1.0 # Distance limit per time unit

    ϕ_t_min = ϕ_0
    x_t_min = x_0
    y_t_min = y_0
    function execute(u_hat_t::Vector{Float64})
        Δϕ_t = u_hat_t[1] # Proposed turn rate
        Δϕ_t = clamp(Δϕ_t, -Δϕ_lim, Δϕ_lim) # Allowed turn rate

        r_t = u_hat_t[2] # Proposed distance
        r_t = clamp(r_t, -r_lim, r_lim) # Allowed distance

        # Compute next state
        ϕ_t = ϕ_t_min + Δϕ_t
        x_t = x_t_min + r_t*cos(ϕ_t)
        y_t = y_t_min + r_t*sin(ϕ_t)

        # Reset state for next step
        ϕ_t_min = ϕ_t
        x_t_min = x_t
        y_t_min = y_t
    end
    
    ϕ_t = ϕ_0    
    x_t = x_0
    y_t = y_0
    observe() = [ϕ_t, x_t, y_t]
            
    return (execute, observe)
end