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

    return 0.5 * ( 1 + tanh( ( mat.R - norm(y - mat.y_c) ) / δ ) )
    
end