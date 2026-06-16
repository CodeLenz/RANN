#
# Refina o valor do passo ótimo por meio de interpolação polinomial
#
#
#  f(x)::R   -> Função objetivo
# df(x)::R^n -> gradiente do objetivo
# x::R^n     -> ponto atual 
# d::R^n     -> direção de descida
#
# α_A        -> Limite inferior do passo 
# α_B        -> Limite superior do passo 
# f_A e f_B  -> Valores da função objetivo nos limites do intervalo
# f_0        -> valor da função quando α=0
# m_0        -> inclinação quando α=0 
# c_1 e c_2  -> ctes para as condições de Wolfe
# δ          -> "tolerância" das bordas do intervalo de busca (para evitar problemas numéricos)
# max_iter   -> número máximo de iterações tanto desta rotina quanto dos refinos de intervalo 
# tol_α      -> tolerância de estagnação do intervalo para evitar loops infinitos
#
function Refinamento(f::Function, Θ::Vector{T}, d::Vector{T}, 
                     α_A::T, α_B::T, f_A::T, f_B::T, c1::T, c2::T,
                     f_0::T, m_0::T, δ=0.1, max_iter = 50, 
                     tol_α=1e-8) where {T<:AbstractFloat}
  
    # Valor da função e inclinação no começo do intervalo 
    _, _, _, ∇L_A = f(Θ .+ α_A .* d)
    m_A = dot(∇L_A, d)
    
    # Valor da função e inclinação no final do intervalo
    m_B = nothing
    
    # Loop pelas iterações (refino)
    for iter in 1:max_iter

        # Prevenção de loop infinito 
        if abs(α_B - α_A) < tol_α * (1.0 + abs(α_A))
            return α_A
        end

        # Se não temos a inclinação em B (direita do intervalo)
        # usamos interpolação quadrática. Do contrário tentamos
        # usar a cúbica 
        if m_B === nothing
            α_j = Interp_quadratica(α_A, α_B, f_A, f_B, m_A, δ)
        else
            α_j = Interp_cubica(α_A, α_B, f_A, f_B, m_A, m_B, δ)
        end
        
        # Avança o ponto e o valor da função 
        Θ_j = Θ .+ α_j .* d
        f_j, _, _, ∇L_j = f(Θ_j)
        
        # Primeiro teste é por Armijo (primeira condição de Wolfe)
        # Se o valor novo está acima da estimativa linear ou se 
        # piorou em relação ao inicial, ajustamos o limite direito 
        # do intervalo. Como começamos sem saber a inclinação do lado 
        # direito, mantemos m_B desconhecido
        if (f_j > f_0 + c1 * α_j * m_0) || (f_j >= f_A)

            # Move o lado direito do intervalo para o ponto atual (j)
            α_B = α_j
            f_B = f_j
            m_B = nothing 

        else

            # O ponto melhorou. Podemos calcular a inclinação neste ponto 
            m_j = dot(∇L_j, d)
            
            # E também podemos ver se a inclinação melhorou (segunda condição de Wolfe)
            if abs(m_j) <= -c2 * m_0

                # Deu tudo certo, aceitamos o passo 
                return α_j 
            end
            
            # Se o gradiente mudar de sinal, então passamos do mínimo. Neste caso precisamos 
            # inverter o nosso intervalo
            if m_j * (α_B - α_A) >= 0.0
                α_B = α_A
                f_B = f_A
                m_B = m_A
            end
            
            # Avança o limite inferior para forçar um passo maior na próxima iteração
            α_A = α_j
            f_A = f_j
            m_A = m_j

        end
    end
    
    # Retorna a estimativa de passo 
    return α_A 
end