#offspring = Array{Array{indiv}}(1000)
#parents = Array{Array{indiv}}(1000)

function _nsga(::indiv{G,Ph,Y,C}, sense, lexico, popSize, nbGen, init, z, fdecode, fdecode!, fCV , pmut, fmut, fcross, seed, fplot, plotevery, refreshtime)::Vector{indiv{G,Ph,Y,C}} where {G,Ph,Y,C}

    popSize = max(popSize, length(seed))
    isodd(popSize) && (popSize += 1)
    P = Vector{indiv{G,Ph,Y,C}}(undef, 2*popSize)
    P[1:popSize-length(seed)] .= [create_indiv(init(), fdecode, z, fCV) for _=1:popSize-length(seed)]
    for i = 1:length(seed)
        P[popSize-length(seed)+i] = create_indiv(convert(G, seed[i]), fdecode, z, fCV)
    end
    for i=1:popSize
        P[popSize+i] = deepcopy(P[i])
    end

    if lexico && isa(P[1].y[1], AbstractArray)
        println("Lexico dominance, gross objects")
        const highestgp = 0     #highest power in the lexico order
        const lowestpower = first(eachindex(P[1].y[1])) #lowest power
        const ignoregp = false  #don't consider the power (used by dominates())
    elseif !lexico && isa(P[1].y[1], AbstractArray)
        println("Standard dominance, gross objects")
        const highestgp = 0
        const lowestpower = 0
        const ignoregp = true
    else
        println("Standard dominance, scalar objects")
        const highestgp = 1
        const lowestpower = 1
        const ignoregp = false
    end

    #this first fast_non_dominated_sort call is necessary for tournament selection
    fast_non_dominated_sort!(view(P, 1:popSize), sense, highestgp, 1, ignoregp)

    @showprogress refreshtime for gen = 1:nbGen

        for i = 1:2:popSize

            pa = tournament_selection(P)
            pb = tournament_selection(P)

            crossover!(pa, pb, fcross, P[popSize+i], P[popSize+i+1])

            rand() < pmut && mutate!(P[popSize+i], fmut)
            rand() < pmut && mutate!(P[popSize+i+1], fmut)

            eval!(P[popSize+i], fdecode!, z, fCV)
            eval!(P[popSize+i+1], fdecode!, z, fCV)
            #parents[gen] = deepcopy(P[1:popSize])
            #offspring[gen] = deepcopy(P[popSize+1:2*popSize])
        end

        ind = 0         #ind+1 is the first index we consider while iterating over the fronts
        indnext = 0     #indnext points to the last element of a front
        indmax = 2*popSize  #indmax is the last index we consider while iterating (last index of superfront)
        gp = highestgp  #power of the lexico order currently considered
        f = 1           #current front
        while (gp >= lowestpower)   #iterate over the lexico powers
            fast_non_dominated_sort!(view(P, ind+1:indmax), sense, gp, f, ignoregp)
            sort!(view(P, ind+1:indmax), by = x->x.rank, alg=Base.Sort.QuickSort)
            indnext = findlast(x->x.rank==f, view(P, 1:indmax))
            while 0 < indnext <= popSize
                ind = indnext
                f += 1
                indnext = findlast(x->x.rank==f, view(P, 1:indmax))
            end
            indnext == 0 && (indnext = indmax)
            indmax = indnext
            gp -= 1
        end
        crowding_distance_assignment!(view(P, ind+1:indnext))
        shuffle!(view(P, ind+1:indnext))    #prevents asymmetries in the choice
        sort!(view(P, ind+1:indnext), by = x -> x.crowding, rev=true, alg=PartialQuickSort(popSize-ind))

        gen == plotevery && println("Loading plot...")
        gen % plotevery == 0 && fplot(P, gen)
    end

    fplot(P, nbGen)
    view(P, 1:popSize)  #returns the first half of array (dominated and not)
end

function fast_non_dominated_sort!(pop::AbstractVector{T}, sense, gp::Int, firstrank::Int, ignoregp::Bool=false) where {T}
    n = length(pop)

    for p in pop
        empty!(p.dom_list)
        p.dom_count = 0
        p.rank = firstrank - 1
    end

    @inbounds for i in 1:n
        for j in i+1:n
            if dominates(sense, pop[i], pop[j], gp, ignoregp)
                push!(pop[i].dom_list, j)
                pop[j].dom_count += 1
            elseif dominates(sense, pop[j], pop[i], gp, ignoregp)
                push!(pop[j].dom_list, i)
                pop[i].dom_count += 1
            end
        end
        if pop[i].dom_count == 0
            pop[i].rank = firstrank
        end
    end

    k = UInt16(firstrank + 1)
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

function crowding_distance_assignment!(pop::AbstractVector{indiv{X,G,Y,C}}) where {X, G, Y, C}
    @assert (C <: Number || C <: AbstractArray)
    if length(first(pop).y) == 2
        sort!(pop, by=x->x.y[1])
        pop[1].y[1] == pop[end].y[1] && return
        if C <: Number
            pop[1].crowding = pop[end].crowding = Inf
        else
            pop[1].crowding[0] = pop[end].crowding[0] = Inf
        end
        @inbounds for i = 2:length(pop)-1
            pop[i].crowding = (pop[i+1].y[1]-pop[i-1].y[1]) / (pop[end].y[1]-pop[1].y[1])
            pop[i].crowding += (pop[i-1].y[2]-pop[i+1].y[2]) / (pop[1].y[2]-pop[end].y[2])
        end
    else
        for ind in pop      #zero out the crowding distance
            if C <: Number
                ind.crowding = 0.
            else
                for p in eachindex(ind.crowding)
                    ind.crowding[p] = 0.
                end
            end
        end
        @inbounds for j = 1:length(first(pop).y) # Foreach objective
            let j = j #https://github.com/JuliaLang/julia/issues/15276
                sort!(pop, by = x-> x.y[j]) #sort by the objective value
            end
            if C <: Number      #Assign infinite value to extremas
                pop[1].crowding = pop[end].crowding = Inf
            else
                pop[1].crowding[0] = pop[end].crowding[0] = Inf
            end
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
    if P[a] < P[b]
        return P[a]
    elseif P[b] < P[a]
        return P[b]
    else
        return rand(Bool) ? P[a] : P[b]
    end
end
