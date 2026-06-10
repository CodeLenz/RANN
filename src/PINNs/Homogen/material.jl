# -----------------------------------------------------------------------------
#  Rotina para devolver E, ν para um ponto (y_1,y_2)
#  em uma fibra de raio r0 em uma matriz 
# -----------------------------------------------------------------------------
function Propriedades_Material(y1::T, y2::T, params::NamedTuple) where {T<:AbstractFloat}

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

# -----------------------------------------------------------------------------
#  C pela regra das misturas
# -----------------------------------------------------------------------------
function Calcula_Tensor_Regra_Mistura(params::NamedTuple)

    # C da matriz
    C_m = Matriz_Constitutiva(params.E_m, params.ν_m)

    # C da fibra
    C_f = Matriz_Constitutiva(params.E_f, params.ν_f)

    # Fração volumétrica da fibra
    V_f = π * params.r0^2

    # Fração volumétrica da matriz
    V_m = 1.0 - V_f

    # Regra das misturas
    CH = V_m * C_m + V_f * C_f

    return CH

end