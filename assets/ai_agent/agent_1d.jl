function initializeAgent()
    m_s_t_min = s_0 # Set initial brain state prior
    v_s_t_min = 0.01

    # Initialize messages and marginals dictionary
    messages = Vector{Message}(undef, 59)
    marginals = Dict{Symbol, ProbabilityDistribution}()
    function infer(u_t::Float64, o_t::Float64)
        data = Dict(:o_t       => o_t, 
                    :u_t       => u_t, 
                    :m_o       => zeros(T-1), 
                    :v_o       => sigma*ones(T-1),
                    :m_s_t_min => m_s_t_min,
                    :v_s_t_min => v_s_t_min)

        step!(data, marginals, messages)
    end

    marginals[:u_2] = ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0, v=tiny) # Register initial action
    act() = mode(marginals[:u_2]) # Choose the mode of the current control state as action

    function slide(slide_msg_idx=7)
        (m_s_t_min, V_s_t_min) = ForneyLab.unsafeMeanCov(messages[slide_msg_idx].dist) # Reset prior state statistics
    end

    return (infer, act, slide)    
end