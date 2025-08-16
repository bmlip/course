# Internal dynamic model
function g(s_t_min::Vector{Float64}, u_t::Vector{Float64}) # Transition function
    ϕ_t_min = s_t_min[1]
    x_t_min = s_t_min[2]
    y_t_min = s_t_min[3]

    Δϕ_t = u_t[1]
    r_t = u_t[2]

    # Compute next state
    ϕ_t = ϕ_t_min + Δϕ_t
    x_t = x_t_min + r_t*cos(ϕ_t)
    y_t = y_t_min + r_t*sin(ϕ_t)

    s_t = [ϕ_t, x_t, y_t]

    return s_t
end

function g_inv_in1(s_t::Vector{Float64}, u_t::Vector{Float64}) # Inverse transition function towards previous state
    ϕ_t = s_t[1]
    x_t = s_t[2]
    y_t = s_t[3]

    Δϕ_t = u_t[1]
    r_t = u_t[2]

    # Compute previous state
    ϕ_t_min = ϕ_t - Δϕ_t
    x_t_min = x_t - r_t*cos(ϕ_t)
    y_t_min = y_t - r_t*sin(ϕ_t)

    s_t_min = [ϕ_t_min, x_t_min, y_t_min]

    return s_t_min    
end

function g_inv_in2(s_t::Vector{Float64}, s_t_min::Vector{Float64}) # Inverse transition function towards control
    ϕ_t_min = s_t_min[1]
    x_t_min = s_t_min[2]
    y_t_min = s_t_min[3]

    ϕ_t = s_t[1]
    x_t = s_t[2]
    y_t = s_t[3]

    # Compute corresponding control
    Δϕ_t = ϕ_t - ϕ_t_min
    if Δϕ_t > 0.0
        Δϕ_t_tor = Δϕ_t - 2*pi
    else
        Δϕ_t_tor = Δϕ_t + 2*pi
    end
    if abs(Δϕ_t_tor) < abs(Δϕ_t)
        Δϕ_t = Δϕ_t_tor
    end
    Δx_t = x_t - x_t_min
    Δy_t = y_t - y_t_min
    r_t = sign(Δx_t*cos(ϕ_t) + Δy_t*sin(ϕ_t))*sqrt(Δx_t^2 + Δy_t^2)
    
    u_t = [Δϕ_t, r_t]

    return u_t
end

function initializeAgent()
    m_s_t_min = [ϕ_0, x_0, y_0] # Set initial brain state prior
    V_s_t_min = 0.01*diageye(3)

    # Initialize messages and marginals dictionary
    messages = Vector{Message}(undef, 59)
    marginals = Dict{Symbol, ProbabilityDistribution}()
    function infer(u_t::Vector{Float64}, o_t::Vector{Float64})
        x_t_c = m_s_t_min[2]
        y_t_c = m_s_t_min[3]
        θ_t = atan(y_t_c, x_t_c)
        if θ_t < 0
            m = [θ_t + pi, 0.0, 0.0] # Goal prior mean with orientation against polar angle
        else
            m = [θ_t - pi, 0.0, 0.0]
        end

        data = Dict(:o_t       => o_t, 
                    :u_t       => u_t, 
                    :m_o       => [m for k=2:T], 
                    :V_o       => [Sigma for k=2:T],
                    :m_s_t_min => m_s_t_min,
                    :V_s_t_min => V_s_t_min)

        step!(data, marginals, messages)
    end

    marginals[:u_2] = ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=zeros(2), v=tiny*diageye(2)) # Register initial action
    act() = mode(marginals[:u_2]) # Choose the mode of the current control state as action

    function slide(slide_msg_idx=7)
        (m_s_t_min, V_s_t_min) = ForneyLab.unsafeMeanCov(messages[slide_msg_idx].dist) # Reset prior state statistics
    end

    return (infer, act, slide)    
end