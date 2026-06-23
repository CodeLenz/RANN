# -----------------------------------------------------------------------------
#  Rotina para devolver E, ν para um ponto (y_1,y_2)
#  em uma fibra de raio r0 em uma matriz 
# -----------------------------------------------------------------------------
function Propriedades_Material_Circular(y1::T, y2::T, params::NamedTuple) where {T<:AbstractFloat}

    # Calcula a distância em relação ao centro
    d = sqrt((y1 - params.yc1)^2 + (y2 - params.yc2)^2)
    
    # Se a distância for menor do que o raio da inclusão 
    # usamos as propriedades da fibra
    if d <= params.r0 
       E = params.E_f
       ν = params.ν_f
    else 
       E = params.E_m
       ν = params.ν_m
    end

    # Retorna as propriedades
    return E, ν

end

# -----------------------------------------------------------------------------
#  Rotina para devolver E, ν para um ponto (y_1,y_2)
#  em uma fibra retangular de altura hf e largura unitária em uma matriz 
#  A fibra se encontra simétrica em relação ao centro da célula
# -----------------------------------------------------------------------------
function Propriedades_Material_Retangular(y1::T, y2::T, params::NamedTuple) where {T<:AbstractFloat}
    
    # Se o ponto estiver entre (0.5 - hf/2) e (0.5 + hf/2) usamos as propriedades da fibra
    if y2 >= (0.5 - params.hf/2) && y2 <= (0.5 + params.hf/2)
       E = params.E_f
       ν = params.ν_f
    else 
       E = params.E_m
       ν = params.ν_m
    end

    # Retorna as propriedades
    return E, ν

end

# -----------------------------------------------------------------------------
#  \C para EPT
# -----------------------------------------------------------------------------
function Matriz_Constitutiva(E::T, ν::T) where {T<:AbstractFloat}

    # termo comum 
    fac = E / (T(1.0) - ν^2)

    # Matriz EPT
    C = fac * [T(1.0)  ν      T(0.0);
               ν      T(1.0)  T(0.0);
               T(0.0)  T(0.0)  (T(1.0) - ν)/T(2.0)]

    # Devolve a matriz
    return C

end