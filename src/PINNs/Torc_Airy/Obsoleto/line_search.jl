#
# Line-Search por Newton-Raphson
#
# alpha = arg min f(x + alpha*d)
#
# Utilizamos diferenças finitas centrais para calcular as derivadas 
# de f(x + alpha*d) em relação a alpha
#
#
function LineSearch_NR(x::Vector,d::Vector,f::Function, rede::Rede, treino::Treino, t_inicial::Vector{Float64}, 
                       u_inicial::Vector{Float64}, du_inicial::Vector{Float64}, n_fisica::Int64, 
                       t_fisica::Matrix{Float64}, delta=1E-6,tol=1E-8; verbose=false)
                       
    # Estimativa inicial do alpha
    alpha = 1.0

    # Loop para encontrar a raiz 
    for iter=1:100

        # Função no ponto atual 
        f0 = ObjetivoFloat(rede, treino, t_inicial, u_inicial, du_inicial, n_fisica, t_fisica, 0, x .+ alpha*d)

        # Perturba para frente
        ff = ObjetivoFloat(rede, treino, t_inicial, u_inicial, du_inicial, n_fisica, t_fisica, 0, x .+ (alpha+delta)*d)

        # Perturba para tras
        ft = ObjetivoFloat(rede, treino, t_inicial, u_inicial, du_inicial, n_fisica, t_fisica, 0, x .+ (alpha-delta)*d)

        # Estimativa da primeira derivada
        d1 = (ff - ft) / (2 * delta)

        # Estimativa da segunda derivada
        d2 = (ff -2 * f0 + ft) / (delta^2)

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