#
# Line search com garantia de curvatura para usar com Quasi-Newton
#
#  f(x)::R   -> Função objetivo
# df(x)::R^n -> gradiente do objetivo
# x::R^n     -> ponto atual 
# d::R^n     -> direção de descida
# α_1::R     -> passo inicial 
# c_1 e c_2  -> ctes para as condições de Wolfe
#
# α_max::R   -> valor limite para o passo 
# δ          -> "tolerância" das bordas do intervalo de busca (para evitar problemas numéricos)
# max_iter   -> número máximo de iterações tanto desta rotina quanto dos refinos de intervalo 
# tol_α      -> tolerância de estagnação do intervalo de refino
#
function LS_Wolfe(f::Function, Θ::Vector{T}, d::Vector{T}; 
                  α_max=10.0, α_1=1.0, c1=1e-4, c2=0.9, δ=0.1, max_iter=50, 
                  tol_α=1e-8) where T<:AbstractFloat

    # Ponto inicial (atual)
    α_0 = 0.0
    f_0, _, _, ∇L_0 = f(Θ)
    m_0 = dot(∇L_0, d)

    # Teste meio bobo, mas pode nos ajudar
    if m_0 >= 0
        return zero(T)
    end
    
    # Variáveis de atualização das iterações
    α_prev = α_0
    α_i    = α_1
    f_prev = f_0
        
    # Loop pelas iterações 
    for iter in 1:max_iter

        # Se passamos do passo limite estipulado, devolvemos o limite refinado para garantir Wolfe
        if α_i > α_max
            Θ_max = Θ .+ α_max .* d
            f_max = f(Θ_max)[1]
            return Refinamento(f, Θ, d, α_prev, α_max, f_prev, f_max, c1, c2, f_0, m_0, δ, max_iter, tol_α)
        end

        # Próximo passo 
        Θ_i = Θ .+ α_i .* d
        f_i, _, _, ∇L_i = f(Θ_i)

        # Violação de Armijo confina o passo válido no intervalo anterior
        if (f_i > f_0 + c1 * α_i * m_0) || (f_i >= f_prev && iter > 1)
            return Refinamento(f, Θ, d, α_prev, α_i, f_prev, f_i, c1, c2, f_0, m_0, δ, max_iter, tol_α)
        end
        
        # Nova inclinação 
        m_i = dot(∇L_i, d)
        
        # Condições 1 e 2 satisfeitas durante a expansão
        if abs(m_i) <= -c2 * m_0
            return α_i
        end
        
        # Inclinação ficou positiva: passamos do mínimo e temos que refinar 
        # a estimativa de mínimo no intervalo anterior
        if m_i >= 0.0
            return Refinamento(f, Θ, d, α_i, α_prev, f_i, f_prev, c1, c2, f_0, m_0, δ, max_iter, tol_α)
        end
        
        # Extrapolação para ampliar o intervalo de busca
        # pois ainda podemos descer mais
        # Nocedal-Wright e Moré-Thuente sugerem entre 2 e 3
        α_prev = α_i
        f_prev = f_i
        α_i = min(α_i * 3.0, α_max) 

    end
    
    # Retorna 0.0 para indicar erro 
    return 0.0

end