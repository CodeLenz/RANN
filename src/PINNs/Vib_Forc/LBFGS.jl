function LBFGS(rede::Rede, treino::NamedTuple, nepoch::Int64; m = 10, conv = 1E-8, α0 = 1.0, otimizador = "LBFGS")

    # Intervalo para monitoramento
    intervalo_monitor = 1000

    # Copia rede.x para variável local
    x = copy(rede.x)

    # Aloca variáveis de monitoramento (Seguindo o padrão do AdamW)
    obj_treino = 0.0
    perda = zeros(3)
    vetor_obj_treino = zeros(nepoch)
    vetor_perda = [zeros(nepoch) for _ in 1:3]    

    # Aloca a resposta estimada para os pontos de teste
    u_test_pred = zeros(1, size(treino.t_teste, 2))

    # Aloca vetores gradiente
    G = zeros(length(x))
    G_new = Vector{Float64}(undef, length(x))
    
    # Cálculo inicial do Gradiente (Usando a assinatura nova: rede, treino, epoch, x)
    Enzyme.autodiff(
        Enzyme.Reverse,
        ObjetivoFloat,
        Const(rede),
        Const(treino),
        Const(1),
        Duplicated(x, G)
    )

    # Histórico para o cálculo da direção (Memória do L-BFGS)
    s_hist = Vector{Vector{Float64}}()
    y_hist = Vector{Vector{Float64}}()
    ρ_hist = Vector{Float64}()

    @showprogress "Otimizando com LBFGS..." for epoch in 1:nepoch

        # --- 1. Determinação da Direção de Descida (Recursão de dois loops) ---
        q = copy(G)
        α_coeffs = zeros(length(s_hist)) # renomeado para não confundir com o passo alpha
        for i in length(s_hist):-1:1
            α_coeffs[i] = ρ_hist[i] * dot(s_hist[i], q)
            q .-= α_coeffs[i] .* y_hist[i]
        end

        if !isempty(y_hist)
            γ = dot(s_hist[end], y_hist[end]) / dot(y_hist[end], y_hist[end])
        else
            γ = 1.0
        end
        
        r = (γ <= 0 ? 1.0 : γ) .* q

        for i in 1:length(s_hist)
            β = ρ_hist[i] * dot(y_hist[i], r)
            r .+= s_hist[i] .* (α_coeffs[i] - β)
        end

        p = -r  # Direção de busca

        # --- 2. Busca Linear (Line Search - Armijo) ---
        obj_atual = ObjetivoFloat(rede, treino, epoch, x)
        αk = α0
        c1 = 1E-4 
        
        # Backtracking
        while ObjetivoFloat(rede, treino, epoch, x + αk * p) > obj_atual + c1 * αk * dot(G, p)
            αk *= 0.5
            if αk < 1e-10 
                break
            end
        end
        
        # --- 3. Atualização e Cálculo do Novo Gradiente ---
        x_new = x + αk * p
        fill!(G_new, 0.0)

        Enzyme.autodiff(
            Enzyme.Reverse,
            ObjetivoFloat,
            Const(rede),
            Const(treino),
            Const(epoch),
            Duplicated(x_new, G_new)
        )
        
        # --- 4. Atualização do Histórico (Condição de Wolfe FR) ---
        s = x_new - x
        y = G_new - G
        ys = dot(y, s)
        
        if ys > 1E-10
            push!(s_hist, s); push!(y_hist, y); push!(ρ_hist, 1.0/ys)
            if length(s_hist) > m
                popfirst!(s_hist); popfirst!(y_hist); popfirst!(ρ_hist)
            end
        end

        # Atualiza variáveis para a próxima época
        x .= x_new
        G .= G_new

        # --- 5. Monitoramento e Logs ---
        obj_treino, perda = Objetivo(rede, treino, epoch, x)
        vetor_obj_treino[epoch] = obj_treino
        for i in 1:3
            vetor_perda[i][epoch] = perda[i]
        end

        # Critério de parada
        if obj_treino <= conv || norm(G) < 1E-9
            println("Convergência atingida na época $epoch")
            break
        end

        if (epoch % intervalo_monitor == 0) || (epoch == nepoch)
            u_test_pred = Resposta_Teste(rede, x, treino, vetor_obj_treino, vetor_perda, epoch, otimizador, intervalo_monitor)
        end
    end

    return x, vetor_obj_treino, u_test_pred
end