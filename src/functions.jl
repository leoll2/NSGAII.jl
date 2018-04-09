function _nsga(::indiv{G,Ph,Y}, sense, lexico, popSize, nbGen, init, z, fdecode, fdecode!, fCV , pmut, fmut, fcross, seed, fplot, plotevery, refreshtime)::Vector{indiv{G,Ph,Y}} where {G,Ph,Y}

    popSize = max(popSize, length(seed))
    isodd(popSize) && (popSize += 1)
    P = Vector{indiv{G,Ph,Y}}(undef, 2*popSize)
    P[1:popSize-length(seed)] .= [create_indiv(init(), fdecode, z, fCV) for _=1:popSize-length(seed)]
    for i = 1:length(seed)
        P[popSize-length(seed)+i] = create_indiv(convert(G, seed[i]), fdecode, z, fCV)
    end
    for i=1:popSize
        P[popSize+i] = deepcopy(P[i])
    end
    fast_non_dominated_sort!(view(P, 1:popSize), sense, lexico)

    @showprogress refreshtime for gen = 1:nbGen

        for i = 1:2:popSize

            pa = tournament_selection(P)
            pb = tournament_selection(P)

            crossover!(pa, pb, fcross, P[popSize+i], P[popSize+i+1])

            rand() < pmut && mutate!(P[popSize+i], fmut)
            rand() < pmut && mutate!(P[popSize+i+1], fmut)

            eval!(P[popSize+i], fdecode!, z, fCV)
            eval!(P[popSize+i+1], fdecode!, z, fCV)
        end

        fast_non_dominated_sort!(P, sense, lexico)
        sort!(P, by = x->x.rank, alg=Base.Sort.QuickSort)

        let f::Int = 1
            ind = 0
            indnext = findlast(x->x.rank==f, P)
            while 0 < indnext <= popSize
                ind = indnext
                f += 1
                indnext = findlast(x->x.rank==f, P)
            end
            indnext == 0 && (indnext = length(P))
            crowding_distance_assignment!(view(P, ind+1:indnext))
            sort!(view(P, ind+1:indnext), by = x -> x.crowding, rev=true, alg=PartialQuickSort(popSize-ind))
        end

        gen % plotevery == 0 && fplot(P, gen)
    end

    fplot(P, nbGen)    #(default)
    #filter(x->x.rank==1, view(P, 1:popSize)) #comment if you want to plot all solutions (unfiltered)
    #fplot(view(P, 1:popSize), 0) #DEBUG
    view(P, 1:popSize)  #returns the first half of array (dominated and not)
end

function fast_non_dominated_sort!(pop::AbstractVector{T}, sense, lexico::Bool) where {T}
    n = length(pop)

    for p in pop
        empty!(p.dom_list)
        p.dom_count = 0
        p.rank = 0
    end

    @inbounds for i in 1:n
        for j in i+1:n
            if dominates(sense, pop[i], pop[j], lexico)
                push!(pop[i].dom_list, j)
                pop[j].dom_count += 1
            elseif dominates(sense, pop[j], pop[i], lexico)
                push!(pop[j].dom_list, i)
                pop[i].dom_count += 1
            end
        end
        if pop[i].dom_count == 0
            pop[i].rank = 1
        end
    end

    k = UInt16(2)
    @inbounds while any(==(k-one(UInt16)), (p.rank for p in pop)) #ugly workaround for #15276
        for p in pop
            if p.rank == k-one(UInt16)
                for q in p.dom_list
                    pop[q].dom_count -= one(UInt16)
                    if pop[q].dom_count == zero(UInt16)
                        pop[q].rank = k
                    end
                end
            end
        end
        k += one(UInt16)
    end
    nothing
end

function crowding_distance_assignment!(pop::AbstractVector{indiv{X,G,Y}}) where {X, G, Y}
    if length(first(pop).y) == 2
        sort!(pop, by=x->x.y[1])
        pop[1].y[1] == pop[end].y[1] && return #Don't waste time if all indivs are the same
        pop[1].crowding = pop[end].crowding = Inf
        @inbounds for i = 2:length(pop)-1
            pop[i].crowding = (pop[i+1].y[1]-pop[i-1].y[1]) / (pop[end].y[1]-pop[1].y[1])
            pop[i].crowding += (pop[i-1].y[2]-pop[i+1].y[2]) / (pop[1].y[2]-pop[end].y[2])
        end
    else
        for ind in pop
            ind.crowding = 0.
        end
        @inbounds for j = 1:length(first(pop).y) # Foreach objective
            let j = j #https://github.com/JuliaLang/julia/issues/15276
                sort!(pop, by = x-> x.y[j]) #sort by the objective value
            end
            pop[1].crowding = pop[end].crowding = Inf #Assign infinite value to extremas
            if pop[1].y[j] != pop[end].y[j]
                for i = 2:length(pop)-1
                    pop[i].crowding += (pop[i+1].y[j]-pop[i-1].y[j]) / (pop[end].y[j]-pop[1].y[j])
                end
            end
        end
    end
end

function tournament_selection(P)
    a, b = rand(1:length(P)÷2), rand(1:length(P)÷2)
    P[a] < P[b] ? P[a] : P[b]
end
