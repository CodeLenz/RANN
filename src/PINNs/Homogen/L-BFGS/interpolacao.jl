#
# Interpolação quadrática do objetivo
#
# Intervalo inicial [α_A, α_B] 
# f_A e  f_B são os valores da função nos extremos do intervalo 
# m_A é a inclinação da função no início do intervalo
# δ é a "folga" em relação às bordas do intervalo (se a raiz estiver muito perto 
# das bordas retornamos o ponto médio.)
#
function Interp_quadratica(α_A::T, α_B::T, f_A::T, f_B::T, m_A::T, δ::T) where T

    # Intervalo inicial
    h = α_B - α_A

    # Varição do objetivo inicial 
    delta_f = f_B - f_A
    
    # Denominador 
    denom = 2.0 * (delta_f - m_A * h)
    
    # Evita divisão por zero e devolve o meio do intervalo
    if abs(denom) < 1e-10
        return (α_A + α_B) / 2.0
    end
    
    # Estimativa do mínimo da função quadrática
    α_q = α_A - (m_A * h^2) / denom
    
    # Se a raiz estiver muito perto das bordas, podemos ter problemas numéricos.
    # Neste caso, devolvemos o ponto médio também
    if α_q <= min(α_A, α_B) + δ * abs(h) || α_q >= max(α_A, α_B) - δ * abs(h)
        return (α_A + α_B) / 2.0
    end
    
    # Retorna a estimativa de mínimo 
    return α_q
    
end