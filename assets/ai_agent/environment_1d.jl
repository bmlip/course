function initializeWorld()
    Δs_lim = 1.0 # Distance limit per time unit

    s_t_min = s_0
    function execute(u_hat_t::Float64)
        Δs_t = clamp(u_hat_t, -Δs_lim, Δs_lim) # Allowed displacement

        # Compute next state
        s_t = s_t_min + Δs_t

        # Reset state for next step
        s_t_min = s_t
    end
    
    s_t = s_0
    observe() = s_t
            
    return (execute, observe)
end