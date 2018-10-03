for i=1:next_x_points
    #println("i:", i, ":")
    lyap_exp = lyaponuv(ts[end-sliding_window:end], J, m, r)
    #println("lyap_exp:", lyap_exp, "\n")
    tasks = Array(Future, sample_size)
    #println("tasks:", tasks, "\n")
    mu = mean(ts[end-sliding_window:end])
    #println("mu:", mu, "\n")

    diff = ts[end-sliding_window+1:end] - ts[end-sliding_window:end-1]
    mu = mean(diff)
    sigma = std(diff)

    sample_values = randn(sample_size) .* sigma .+ ts[end]

    for j=1:sample_size
        tempts=deepcopy(ts[end-sliding_window:end])
        append!(tempts, [sample_values[j]])
        tasks[j] = @spawn lyaponuv(tempts, J, m, r)
    end
    
    exponents = Array(Float64, sample_size)
    for j=1:sample_size
        exponents[j] = fetch(tasks[j])
    end
    #println("exponent:", mean(lyap_exp))
    
    exp_diff = abs(exponents .- lyap_exp)
    min_index = findmin(exp_diff)
    best_val = sample_values[min_index[2]]
    append!(ts, [best_val])
    println(i, "   best value:", best_val, "\t")
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
