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
    denom = T(2.0) * (delta_f - m_A * h)

    # Evita divisão por zero e devolve o meio do intervalo
    if abs(denom) < T(1e-10)
        return (α_A + α_B) / T(2.0)
    end
    
    # Estimativa do mínimo da função quadrática
    α_q = α_A - (m_A * h^2) / denom
    
    # Se a raiz estiver muito perto das bordas, podemos ter problemas numéricos.
    # Neste caso, devolvemos o ponto médio também
    if α_q <= min(α_A, α_B) + δ * abs(h) || α_q >= max(α_A, α_B) - δ * abs(h)
        return (α_A + α_B) / T(2.0)
    end
    
    # Retorna a estimativa de mínimo
    return α_q

end

#
# Interpolação cúbica
#
# Intervalo inicial [α_A, α_B] 
# f_A e  f_B são os valores da função nos extremos do intervalo 
# m_A e m_B são as inclinações da função nos extremos do intervalor
# δ é a "folga" em relação às bordas do intervalo (se a raiz estiver muito perto 
# das bordas retornamos o ponto médio.)
#
function Interp_cubica(α_A::T, α_B::T, f_A::T, f_B::T, m_A::T, m_B::T, δ::T) where T

    # Calcula os coeficientes para a aproximação cúbica
    h = α_B - α_A
    G = (f_B - f_A) / h
    θ = m_A + m_B - T(3.0) * G
    D = θ^2 - m_A * m_B

    # Se o determinante for negativo usamos a interpolação quadrática
    if D < zero(T)
        return Interp_quadratica(α_A, α_B, f_A, f_B, m_A, δ)
    end

    # Agora podemos calcular os próximos termos da solução cúbica
    w = sign(h) * sqrt(D)
    denom = m_B - m_A + T(2.0) * w

    # Se o denominador for muito pequeno devolvemos o ponto médio
    if abs(denom) < T(1e-10)
        return (α_A + α_B) / T(2.0)
    end

    # Estimativa do ponto de mínimo
    α_c = α_B - h * (m_B + w - θ) / denom

    # Se a estimativa estiver muito perto das bordas devolvemos o ponto médio
    if α_c <= min(α_A, α_B) + δ * abs(h) || α_c >= max(α_A, α_B) - δ * abs(h)
        return (α_A + α_B) / T(2.0)
    end
    
    # Retorna a estimativa de mínimo
    return α_c

end