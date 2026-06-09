# Propriedades do material compósito na célula unitária

# Struct que define as propriedades do material
# Material 1 é a inclusão - E1, ν1
# Material 2 é a matriz - E2, ν2
# A inclusão tem raio R com centro em [0.5, 0.5]
struct PropMaterial

    E1 = 10.0
    E2 = 1.0
    ν1 = 0.3
    ν2 = 0.3
    R = 0.1
    y_c = [0.5 0.5]

end

# Retorna o tensor constitutivo C na célula
function C(y::Vector{T}, mat::PropMaterial) where T

    # Define o módulo de elasticidade e o coeficiente de Poisson efetivos do ponto a partir  
    # de uma interpolação entre E1 e E2, através de uma função α(y)

    # Calcula α(y)
    α = func_α(y, mat)

    # Calcula o E e ν efetivo no ponto
    E = mat.E2 + (mat.E1 - mat.E2) * α
    ν = mat.ν2 + (mat.ν1 - mat.ν2) * α

    # Calcula e retorna o tensor C
    k = E / (1 - ν^2) 
    return k * [1 ν 0;
                ν 1 0;
                0 0 (1 - ν) / 2]

end

# Calcula a função α(y) para a interpolação do módulo de elasticidade
# Neste caso, a fórmula está definida para uma célula unitária 
# com inclusão circular de R com centro em [0.5, 0.5]
# δ controla a espessura da zona de transição
function func_α(y::Vector{T}, mat::PropMaterial; δ = 0.01) where T

    return 0.5 * ( 1 + tanh( ( mat.R - norm(y .- mat.y_c) ) / δ ) )
    
end

# Derivadas analíticas de C(y) em relação a y1 e y2
# Usa regra da cadeia: dC/dy_k = (∂C/∂E)(dE/dy_k) + (∂C/∂ν)(dν/dy_k)
# Retorna: (dC_dy1, dC_dy2) — duas matrizes 3×3
function dC_dy(y::AbstractVector{T}, mat::MaterialParams) where T

    dy1 = y[1] - mat.y_c[1]
    dy2 = y[2] - mat.y_c[2]
    dist = norm(y .- mat.y_c)

    # Evita divisão por zero no centro da inclusão
    if dist < 1e-12
        dC_dy1 = zeros(3, 3)
        dC_dy2 = zeros(3, 3)
        return dC_dy1, dC_dy2
    end

    # α e sua derivada em relação à distância
    t    = tanh( (mat.R - dist) / mat.delta )
    dα_dd = -0.5 * (1.0 - t^2) / mat.delta   # dα/d(dist)

    # Derivadas da distância em relação a y1 e y2
    dd_dy1 = dy1 / dist
    dd_dy2 = dy2 / dist

    # Derivadas de α em relação a y1 e y2
    dα_dy1 = dα_dd * dd_dy1
    dα_dy2 = dα_dd * dd_dy2

    # Propriedades em y
    α      = 0.5 * (1.0 + t)
    E      = mat.E2  + (mat.E1  - mat.E2)  * α
    ν      = mat.ν2 + (mat.nu1 - mat.ν2) * α
    dE_dα  = mat.E1  - mat.E2
    dν_dα = mat.ν1 - mat.ν2

    dE_dy1  = dE_dα  * dα_dy1
    dE_dy2  = dE_dα  * dα_dy2
    dν_dy1 = dν_dα * dα_dy1
    dν_dy2 = dν_dα * dα_dy2

    # C = E/(1-ν²) * M(ν)  onde M depende de ν
    # Fator escalar e suas derivadas
    denom    = 1.0 - ν^2
    fac      = E / denom
    dfac_dE  = 1.0 / denom
    dfac_dν = E * 2.0 * ν / denom^2    # d(E/(1-ν²))/dν

    dfac_dy1 = dfac_dE * dE_dy1 + dfac_dν * dν_dy1
    dfac_dy2 = dfac_dE * dE_dy2 + dfac_dν * dν_dy2

    # M(ν) = [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]
    # dM/dν = [[0, 1, 0], [1, 0, 0], [0, 0, -1/2]]
    M = [1.0 ν        0.0     ;
         ν   1.0      0.0     ;
         0.0 0.0 (1 - ν) / 2.0]
    dM_dν = [0.0 1.0  0.0;
              1.0 0.0  0.0;
              0.0 0.0 -0.5]

    # dC/dy_k = dfac/dy_k * M + fac * dM/dν * dν/dy_k
    dC_dy1 = dfac_dy1 .* M .+ fac .* dM_dν .* dν_dy1
    dC_dy2 = dfac_dy2 .* M .+ fac .* dM_dν .* dν_dy2

    return dC_dy1, dC_dy2

end