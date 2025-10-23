using LinearAlgebra

#
# Line-Search por Newton-Raphson
#
# alpha = arg min f(x + alpha*d)
#
# Utilizamos diferenças finitas centrais para calcular as derivadas 
# de f(x + alpha*d) em relação a alpha
#
#
function LineSearch_NR(x::Vector,d::Vector,f::Function,delta=1E-6,tol=1E-8; verbose=false)

    # Estimativa inicial do alpha
    alpha = 1.0

    # Loop para encontrar a raiz 
    for iter=1:100

        # Função no ponto atual 
        f0 = f(x .+ alpha*d)

        # Perturba para frente
        ff = f(x .+ (alpha+delta)*d)

        # Perturba para tras
        ft = f(x .+ (alpha-delta)*d)

        # Estimativa da primeira derivada
        d1 = (ff-ft)/(2*delta)

        # Estimativa da segunda derivada
        d2 = (ff -2*f0 + ft)/(delta^2)

        # Nova estimativa do alpha. Aqui temos que evitar a 
        # divisão por zero. Na verdade, isso é um problema não só 
        # numérico, mas também para a convergência do NR. Acho que o 
        # correto aqui seria devolver um erro e usar o Armijo neste caso
        if abs(d2)>=tol
           alphan = alpha - d1/d2
        else
           return -1
        end

        # Diferença na estimativa
        difalpha = abs(alphan-alpha)

        # Testa se atingimos a tolerância 
        if  difalpha < tol 
            verbose && println("LineSearch_NR:: atingimos a tolerância $(difalpha) em $(iter) iterações")
            alpha = alphan
            break
        end

        # Atualiza a estimativa do alpha 
        alpha = alphan  

    end

    # Retorna a estimativa do alpha 
    return alpha

end

#
# L-BFGS 
#
#     f:: função a otimizar
#    df:: derivada da função 
#    x0:: ponto inicial 
#     m:: número de iterações para trás para o ajuste da hessiana 
# niter:: número máximo de iterações
#   tol:: tolerância de parada pelo gradiente 
#    α0:: Passo inicial do backtracking (como H é escalonada, 1.0 é uma boa aproximação)
#
function LBFGS(f::Function, df::Function, x0::Vector; m=10, niter=50, tol=1e-5, α0=10.0)

    # Copia o vetor de entrada
    x = copy(x0)

    # Dimensão do vetor de entrada 
    n = length(x)
    
    # Calcula o gradiente do objetivo em relação aos parâmetros da rede
    g = df(x)

    # Vetores de vetores para guardar o histórico da otimização 
    # e fazer a atualização da "Hessiana"
    s_hist = Vector{Vector{Float64}}()
    y_hist = Vector{Vector{Float64}}()
    ρ_hist = Vector{Float64}()

    # Loop principal da otimização 
    for k in 1:niter

        # Loop para obter a direção de descida, utilizando as direções anteriores
        q = copy(g)
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
        if  γ<0.0
            println("Resetando o gamma ... isso não deveria acontecer")
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
        p = -r  

        # Calcula o valor atual da função 
        fx = f(x)

        @show fx

        # Line-search usando o método de NR
        αk = LineSearch_NR(x,p,f)

        # Caso o NR falhe, utilizamos um Armijo backtracking
        if αk==-1  
            # Armijo Backtracking LS - Não tem garantia de que  γ seja estritamente 
            # positivo, o que pode dar problema no L-BFGS. O correto é garantir 
            # as condições completas de Wolff (redução do gradiente também)
            #
            αk = α0
            c = 0.1
            while f(x + αk * p) > fx - c * αk * dot(g, p)
                αk *= 0.5
                if αk < 1e-8
                    break
                end
            end
        end
        
        # Atualiza as variáveis de projeto
        x_new = x + αk * p

        # Gradiente na nova posição 
        g_new = df(x_new)
        
        # Atualiza s, y e ρ
        s = x_new - x
        y = g_new - g
        ρ = 1.0 / dot(y, s)

        # Atualiza histórico limitado
        if dot(y,s)>1E-10
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
        g .= g_new

        # Critério de parada por derivada
        if norm(g) < tol
            println("Convergência atingida em $k iterações.")
            break
        end
    end

    # Retorna as variáveis otimizadas
    return x
end


#
# Mínimo em 1,1
#
function Rosenbroock(x) 
      100*(x[2]-x[1]^2)^2 + (1-x[1])^2
end

function dRosenbrook(x)
    [(-400*x[1]*(x[2]-x[1]^2))-2*(1-x[1]) ; 
     200*(x[2]-x[1]^2) ]
end

# Chama 
LBFGS(Rosenbroock, dRosenbrook, [0.0; 2.0])   