# Função de otimização da rede neural
# Método LBFGS
function LBFGS(obj_fn::Function, grad_fn!::Function, x::Vector{Float64}, rede::Rede, treino::NamedTuple,
               nepoch::Int64, prob::String; m = 10, conv = 1E-8, α0 = 1.0, otimizador = "LBFGS")

    # Intervalo para monitoramento
    intervalo_monitor = 1000

    # Aloca variáveis de monitoramento (Seguindo o padrão do AdamW)
    obj_treino = 0.0
    vetor_obj_treino = zeros(nepoch)

    # Define conjunto de pontos de treino para o L-BFGS 
    # Nesse caso, é o conjunto completo
    pontos_treino_fisica = treino.fisica

    # Aloca a resposta estimada para os pontos de teste
    u_test_pred = zeros(1, size(treino.teste, 2))

    # Aloca vetores gradiente
    G = zeros(length(x))
    G_new = Vector{Float64}(undef, length(x))
    
    # Cálculo inicial do gradiente com o Enzyme
    # Realiza a derivação automática com o Enzyme
    grad_fn!(G, x, pontos_treino_fisica)

    # Histórico para o cálculo da direção (Memória do L-BFGS)
    s_hist = Vector{Vector{Float64}}()
    y_hist = Vector{Vector{Float64}}()
    ρ_hist = Vector{Float64}()

    @showprogress "Otimizando com LBFGS..." for epoch in 1:nepoch

        # Determinação da Direção de Descida (Recursão de dois loops) 
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

        # 2. Busca Linear (Line Search - Armijo)
        obj_atual = obj_fn(x, pontos_treino_fisica)
        αk = α0
        c1 = 1E-4 
        
        # Backtracking
        while obj_fn(x + αk * p, pontos_treino_fisica) > obj_atual + c1 * αk * dot(G, p)
            αk *= 0.5
            if αk < 1e-10 
                break
            end
        end
        
        # Atualização e Cálculo do Novo Gradiente 
        # Realiza a derivação automática com o Enzyme
        x_new = x + αk * p
        grad_fn!(G_new, x_new, pontos_treino_fisica)
        
        # Atualização do Histórico (Condição de Wolfe FR) 
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

        # Monitoramento e Logs 
        obj_treino = obj_fn(x, pontos_treino_fisica)
        vetor_obj_treino[epoch] = obj_treino

        # Critério de parada
        if obj_treino <= conv || norm(G) < 1E-9
            println("Convergência atingida na época $epoch")
            break
        end

        if (epoch % intervalo_monitor == 0) || (epoch == nepoch)
            u_test_pred = Resposta_Teste(rede, x, treino, vetor_obj_treino, epoch, 
                                         prob, otimizador, intervalo_monitor)
        end
    end

    return x, vetor_obj_treino, u_test_pred

end