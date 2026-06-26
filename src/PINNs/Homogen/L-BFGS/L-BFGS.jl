#
# Método L-BFGS (Limited-Memory BFGS) Algoritmo Global
#
# Entradas:
# obj_fn::Function  -> Função objetivo
# df::Function      -> Gradiente da função objetivo
# x0::AbstractVector -> Ponto inicial
# m::Int            -> Tamanho limite da janela de memória (padrão: 10)
#
# Parâmetros opcionais:
# ϵ_g, ϵ_x, ϵ_f     -> Tolerâncias para gradiente, espaço e função
# max_iter          -> Número máximo de iterações permitidas
#
function L_BFGS!(obj_fn::F, rede::Rede{T}, historico::Vector{T}, historico_energia::Vector{T},
                 historico_avg::Vector{T}, nepoch::Int; m::Int=10, 
                 ϵ_g=1e-6, ϵ_x=1e-8, ϵ_f=1e-8, N_SHOW = 50, verbose = true) where {F<:Function, T<:AbstractFloat}

    # Transforma pesos e bias em vetor flat para o L-BFGS
    Θ_0 = vcat([vec(c.W) for c in rede.camadas]..., [vec(c.b) for c in rede.camadas]...)

    # Chama função objetivo e recupera termos de custo e gradiente 
    custo_prev, L_energia, L_avg, ∇L = obj_fn(Θ_0)
    
    # Inicialização das variáveis de estado
    Θ_n  = similar(Θ_0)
    k    = 0
    d = similar(Θ_0)
    s = similar(Θ_0)
    y = similar(Θ_0)

    # Já guardamos o custo aqui
    push!(historico, custo_prev)
    push!(historico_energia, L_energia)
    push!(historico_avg, L_avg) 

    # Inicia a fila de memória vazia
    # como um vetor de tuplas contendo os conjuntos (s, y, ρ)
    memoria = Tuple{Vector{T}, Vector{T}, T}[]
    
    # Loop principal
    @showprogress "Loop L-BFGS" for epoch = 1:nepoch
        
        # Passo inicial ou caso a memória tenha sido limpa
        if epoch == 0 || isempty(memoria)
            d .= -∇L

        else
            # Reconstrução implícita da direção via recursão
            r = Dois_Lacos(∇L, memoria) 
            d .= -r

        end
        
        # Chamada da rotina de busca em linha (LS_Wolfe)
        α = LS_Wolfe(obj_fn, Θ_0, d)
        
        # Atualização 
        Θ_n .= Θ_0 .+ α .* d
        
        # Avaliação dos termos da secante
        s .= Θ_n - Θ_0
        custo, L_energia, L_avg, ∇L_n = obj_fn(Θ_n)
        y .= ∇L_n .- ∇L

        sy = dot(s, y)

        # Já guardamos o custo aqui
        push!(historico, custo)
        push!(historico_energia, L_energia)
        push!(historico_avg, L_avg)

        # Impede estagnação
        if norm(s) < ϵ_x * (1.0 + norm(Θ_0)) || abs(custo - custo_prev) < ϵ_f * (1.0 + abs(custo_prev))
                println("Parou por estagnação na iteração ", k)
                Θ_0 .= Θ_n
                break
        end
        
        # Guarda na memória
        if sy > 0.0
            ρ = 1.0 / sy
            push!(memoria, (copy(s), copy(y), ρ))

            # Descarta a tupla mais antiga se exceder o limite 'm'
            if length(memoria) > m
                popfirst!(memoria)
            end
            
        end
        
        # Atualização para a próxima iteração
        Θ_0  .= Θ_n
        ∇L .= ∇L_n
        k  += 1
        custo_prev = custo

        # Mostra o resultado atual 
        if verbose && (epoch == 1 || epoch % max(1, epoch ÷ N_SHOW) == 0)
            println("Iteração ", epoch, "    energia = ", custo)
        end

        if norm(∇L) < ϵ_g
            println("Norma atingida na iteração: ", epoch)
        end

    end

    # Altera os valores da rede in-place para os encontrados pelo L-BFGS
    # Reconstrói os pesos e bias a partir do vetor flat
    offset = 0
    for c in rede.camadas
        n_W = length(c.W)
        c.W .= reshape(Θ_0[offset + 1 : offset + n_W], size(c.W))
        offset += n_W
    end
    for c in rede.camadas
        n_b = length(c.b)
        c.b .= reshape(Θ_0[offset + 1 : offset + n_b], size(c.b))
        offset += n_b
    end

end