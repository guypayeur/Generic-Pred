module Lyaponuv

export lyaponuv_k, lyaponuv, lyaponuv_exp, lyaponuv_next, match_pairs, attractor,
    follow_points, compute_norms, column_norms, get_next

function lyaponuv_k(time_series, J, m, ref)
    X = attractor(time_series, m, J)
    norms = compute_norms(X)
    pairs = match_pairs(norms)
    y = follow_points(pairs, norms, ref)    
    return(norms,y)
end

function match_pairs(norms)
    M = size(norms)[1]
    pairs = Array(Int, M)
    for row in 1:M
        mn, idx = findmin(norms[row, :])
        pairs[row] = idx
    end
    return(pairs)
end

function attractor(time_series, m, J)
    N = length(time_series)
    M = N - (m - 1) * J  
    X = Array(Float64, m, M)
    i = 1
    for i=1:M
        X[:,i] = time_series[i:J:(i+(m-1)*J)]
    end
    return(X)
end

function follow_points(pairs, norms, ref)
    y = Array(Float64, ref)
    M = size(norms)[1]
    for i=0:ref-1
        agg = 0 
        count = 0
        for j=1:M
            jhat = pairs[j]+i
            jtrue = j+i

            if jhat <= M && jtrue <= M
                agg = agg + log(norms[jtrue, jhat])
               # agg = agg + log(vecnorm(X[:, jtrue] - X[:, jhat]))
                count = count + 1
            end
        end
        y[i+1] = agg/count # divide by delta-t also?
    end
    return(y)
end


function compute_norms(X)
    M = size(X)[2]
    norms = Array(Float64, M, M)
    for i=1:M
        norms[i,:] = column_norms(X, i)      
    end    
    return(norms)
end


function column_norms(X, i)
    M = size(X)[2]
    X_diff = X .- X[:, i]
    norm_vector = [vecnorm(X_diff[:, k]) for k=1:M]
    norm_vector[i] = 10^10
    return(norm_vector)
end


function lyaponuv_exp(series)
    nn = !isnan(series)
    A = ones(length(series), 2)
    A[:,1] = linspace(1, length(series), length(series))
    gradient = \(A, series)
    return(gradient[1])
end


function lyaponuv(time_series, J, m, ref)
    ts = lyaponuv_k(time_series, J, m, ref)[2]
    exponent = lyaponuv_exp(ts[isfinite(ts)])  ## only input those which are finite
    return(exponent)
end

function get_next(ts, m, M, norms, ref, J)
    
    attractor_array = attractor(ts, m, J)
    temp_norms = Array(Float64, M+1, M+1)
    temp_norms[1:M, 1:M] = norms
    col = column_norms(attractor_array, M+1)
    temp_norms[M+1, :] = col
    temp_norms[:, M+1] = col

    pairs=match_pairs(temp_norms)
    lyap_k_temp = follow_points(pairs, temp_norms, ref)
    return(lyaponuv_exp(lyap_k_temp))
end

function lyaponuv_next(time_series, J, m, ref, sample_size)
    ts_diff = time_series[2:end] - time_series[1:end-1]
    sigma = std(ts_diff)
    samples = randn(sample_size) * sigma + time_series[end]
    @time norms, lyap_k = lyaponuv_k(time_series, J, m, ref)
    true_exponent = lyaponuv_exp(lyap_k)
    exponents = Array(Float64, sample_size)
    M = size(norms)[1]
    tasks = Array(Future, sample_size)

    for i=1:sample_size
        s = samples[i]
        tasks[i] = @spawn get_next(vcat(time_series, s), m, M, norms, ref, J)
        #@time exponents[i] = get_next(vcat(time_series, s), m, M, norms, ref, J)        
        #@printf("process: %d\n", i) 
    end
    
    for i=1:sample_size
        exponents[i]=fetch(tasks[i])
    end

    diff = abs(exponents .-  true_exponent)    
    val, idx = findmin(diff)
    println("Next Value:", samples[idx])
    return(samples[idx])
end

end