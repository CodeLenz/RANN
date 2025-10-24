# L-BFGS 
#     f:: função a otimizar
#    df:: derivada da função 
#    x0:: ponto inicial 
#     m:: número de iterações para trás para o ajuste da hessiana 
# niter:: número máximo de iterações
#   tol:: tolerância de parada pelo gradiente 
#    α0:: Passo inicial do backtracking (como H é escalonada, 1.0 é uma boa aproximação)
#
function LBFGS(rede::Rede, treino::Treino, x0::Vector, nepoch::Int64; m = 10, conv = 1E-8, α0 = 10.0, otimizador = "LBFGS")

    # Aloca objetivos
    obj_treino = 0.0
    perda_inicial_u = 0.0
    perda_inicial_du = 0.0
    perda_fisica = 0.0

    # Acessa os termos em Treino por apelidos
    t_inicial = treino.t_inicial
    u_inicial = treino.u_inicial
    du_inicial = treino.du_inicial
    t_fisica =  treino.t_fisica

    # Copia o vetor de entrada
    x = copy(x0)

    # Dimensão do vetor de entrada 
    n = rede.n_projeto

    # Obtém o número de dados de treino
    n_fisica = size(t_fisica, 2)

    # Aloca um array para monitorar o objetivo
    vetor_obj_treino = zeros(nepoch)
    vetor_perda_inicial_u = zeros(nepoch)
    vetor_perda_inicial_du = zeros(nepoch)
    vetor_perda_fisica = zeros(nepoch)

    # Aloca a resposta estimada para os pontos de teste
    u_test_pred = zeros(1, size(treino.u_an, 2))

    # Calcula o objetivo da rede para o treino 
    obj_treino = ObjetivoFloat(rede, treino, t_inicial, u_inicial, du_inicial, n_fisica, t_fisica, 0, x)

    # Aloca vetor gradiente - vazio, valores serão inputados posteriormente
    G = Vector{Float64}(undef,length(x))
    G_new = Vector{Float64}(undef,length(x))
    
    # Calcula o gradiente do objetivo em relação aos parâmetros da rede
    Enzyme.autodiff(
        Enzyme.Reverse,
        ObjetivoFloat,
        Const(rede),
        Const(treino),
        Const(t_inicial),
        Const(u_inicial),
        Const(du_inicial),
        Const(n_fisica),
        Const(t_fisica),
        Const(0),
        Duplicated(x, G)
    )

    # Vetores de vetores para guardar o histórico da otimização 
    # e fazer a atualização da "Hessiana"
    s_hist = Vector{Vector{Float64}}()
    y_hist = Vector{Vector{Float64}}()
    ρ_hist = Vector{Float64}()

    # Loop principal da otimização 
    @showprogress "Otimizando com LBFGS..." for epoch in 1:nepoch

        # Loop para obter a direção de descida, utilizando as direções anteriores
        q = copy(G)
        α = zeros(length(s_hist))
        for i in length(s_hist):-1:1
            α[i] = ρ_hist[i] * dot(s_hist[i], q)
            q .-= α[i] .* y_hist[i]
        end

        #
        # Escalanomento de H = (sᵀy)/(yᵀy)
        #
        if !isempty(y_hist)
            γ = dot(s_hist[end], y_hist[end]) / dot(y_hist[end], y_hist[end])
        else
            γ = 1.0
        end

        #
        # Evita que o escalonamento seja negativo
        #
        if  γ < 0.0
            println("Resetando o γ ... isso não deveria acontecer")
            γ = 1.0
        end

        # Escalona a direção 
        r = γ .* q

        # Atualiza β e r
        for i in 1:length(s_hist)
            β = ρ_hist[i] * dot(y_hist[i], r)
            r .+= s_hist[i] .* (α[i] - β)
        end

        # direção de descida
        p = - r  

        # Calcula o valor atual da função 
        obj_treino = ObjetivoFloat(rede, treino, t_inicial, u_inicial, du_inicial, n_fisica, t_fisica, 0, x)

        # Line-search usando o método de NR
        αk = LineSearch_NR(x, p, ObjetivoFloat, rede, treino, t_inicial, u_inicial, du_inicial, n_fisica, t_fisica)

        # Caso o NR falhe, utilizamos um Armijo backtracking
        if αk == -1  
            # Armijo Backtracking LS - Não tem garantia de que  γ seja estritamente 
            # positivo, o que pode dar problema no L-BFGS. O correto é garantir 
            # as condições completas de Wolff (redução do gradiente também)
            #
            αk = α0
            c = 0.1
        
            obj_treino_frente = ObjetivoFloat(rede, treino, t_inicial, u_inicial, du_inicial, n_fisica, t_fisica, 0, x + αk * p)
            
            while obj_treino_frente > obj_treino - c * αk * dot(G, p)
                αk *= 0.5
                if αk < 1e-8
                    break
                end
            end
        end
        
        # Atualiza as variáveis de projeto
        x_new = x + αk * p

        # Zera o vetor gradiente para novo cálculo
        fill!(G_new, 0.0)

        # Gradiente na nova posição 
        Enzyme.autodiff(
            Enzyme.Reverse,
            ObjetivoFloat,
            Const(rede),
            Const(treino),
            Const(t_inicial),
            Const(u_inicial),
            Const(du_inicial),
            Const(n_fisica),
            Const(t_fisica),
            Const(epoch),
            Duplicated(x_new, G_new)
        )
        
        # Atualiza s, y e ρ
        s = x_new - x
        y = G_new - G
        ρ = 1.0 / dot(y, s)

        # Atualiza histórico limitado
        if dot(y,s) > 1E-10
            push!(s_hist, s)
            push!(y_hist, y)
            push!(ρ_hist, ρ)
            if length(s_hist) > m
                popfirst!(s_hist)
                popfirst!(y_hist)
                popfirst!(ρ_hist)
            end
        end

        # Substitui variáveis para a próxima iteração
        x .= x_new
        G .= G_new

        # Calcula o objetivo da rede para o treino 
        obj_treino, perda_inicial_u, perda_inicial_du, perda_fisica  = 
            Objetivo(rede, treino, t_inicial, u_inicial, du_inicial, n_fisica, t_fisica, epoch, x)

        # Armazena o objetivo
        vetor_obj_treino[epoch] = obj_treino
        vetor_perda_inicial_u[epoch] = perda_inicial_u
        vetor_perda_inicial_du[epoch] = perda_inicial_du
        vetor_perda_fisica[epoch] = perda_fisica

        # Critério de parada por derivada
        if norm(G) < conv
            println("Convergência atingida em $epoch iterações.")
            break
        end

        # A cada 1000 epochs vamos monitorar o comportamento da rede 
        if (epoch % 1000 == 0) || (epoch == nepoch)

            # Obtém a resposta da rede neural para os pontos de teste
            u_test_pred = Deslocamento_Teste(rede, x, treino.u_an, treino.t_teste, vetor_obj_treino, vetor_perda_inicial_u,
                                             vetor_perda_inicial_du, vetor_perda_fisica, epoch, otimizador)

        end


    end

    # Retorna as variáveis otimizadas
    return x, vetor_obj_treino, u_test_pred

end