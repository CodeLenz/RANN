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

# -----------------------------------------------------------------------------
# Problema de Fibra retangular:
# Calcula E e ν efetivos por regra da mistura
# Estima os coeficientes obtidos pela homogeneização e compara
# -----------------------------------------------------------------------------
function Compara_Retangular(mat_params::NamedTuple, CH::Matrix{T}) where {T<:AbstractFloat}

   # Calcula E1 e ν12 efetivos por regra da mistura
   E1 = mat_params.E_m * (1 - mat_params.hf) + mat_params.E_f * mat_params.hf
   ν12 = mat_params.ν_m * (1 - mat_params.hf) + mat_params.ν_f * mat_params.hf

   # Calcula E2 através da fórmula de Chamis
   E2 = mat_params.E_m / (1 - sqrt(mat_params.hf) * (1 - mat_params.E_m / mat_params.E_f)) 

   # Calcula coeficientes homogeneizados a partir do tensor C calculado
   E1H = (CH[1, 1] * CH[2, 2] - CH[1, 2]^2) / CH[2, 2]
   E2H = (CH[1, 1] * CH[2, 2] - CH[1, 2]^2) / CH[1, 1]
   ν12H = CH[1, 2] / CH[2, 2]

   # Mostra resultados
   println("Coeficientes por regra das misturas:\n E1: $E1; \n E2: $E2; \n ν12: $ν12 \n")
   println("Coeficientes por homogeneização:\n E1: $E1H; \n E2: $E2H; \n ν12: $ν12H")

   # Salva resultados em arquivo de texto
   writedlm("Resultados/compara_retangular.txt", [
       "E1"   E1;
       "E2"   E2;
       "nu12" ν12;
       "E1H"  E1H;
       "E2H"  E2H;
       "nu12H" ν12H
   ])

end