# -----------------------------------------------------------------------------
#  Calcula as deformações para a parte do pós-processamento do C^H
# -----------------------------------------------------------------------------
function Calcula_Deformacoes_pos_treino(y1::T, y2::T, ε_macro::Matrix{T}, rede::Rede{T},
                                        N_modos::Int, As::Vector{Matrix{T}}, Z_buffers::Vector{Matrix{T}}; h=T(1e-5)) where {T<:AbstractFloat}

    # Perturba em y1
    Forward_Rede_InPlace!(rede, reshape(Camada_Periodica(y1 + h, y2, N_modos), :, 1), Z_buffers, As)
    u_x_mais  = vec(As[end])
    Forward_Rede_InPlace!(rede, reshape(Camada_Periodica(y1 - h, y2, N_modos), :, 1), Z_buffers, As)
    u_x_menos = vec(As[end])
    du_dy1 = (u_x_mais .- u_x_menos) ./ (T(2.0) * h)
    
    # Perturba em y2
    Forward_Rede_InPlace!(rede, reshape(Camada_Periodica(y1, y2 + h, N_modos), :, 1), Z_buffers, As)
    u_y_mais  = vec(As[end])
    Forward_Rede_InPlace!(rede, reshape(Camada_Periodica(y1, y2 - h, N_modos), :, 1), Z_buffers, As)
    u_y_menos = vec(As[end])
    du_dy2 = (u_y_mais .- u_y_menos) ./ (T(2.0) * h)
    
    # Calcula as deformações
    ε11 = ε_macro[1,1] + du_dy1[1]
    ε22 = ε_macro[2,2] + du_dy2[2]
    ε12 = ε_macro[1,2] + T(0.5) * (du_dy2[1] + du_dy1[2])
    
    # Retorna no formato de Voigt
    return [ε11, ε22, ε12]

end

# -----------------------------------------------------------------------------
#  Integração final de propriedades
# -----------------------------------------------------------------------------
function Calcula_Tensor_Homogeneizado(redes::Vector{Rede{T}}, modos::Vector{Matrix{T}}, 
                                      N_modos::Int, mat_params::NamedTuple, N_eval::Int=50) where {T<:AbstractFloat}

    # Aloca a matriz 
    CH = zeros(T, 3, 3)

    # Gera N_eval pontos entre 0 e 1 para calcular a integral 
    # na célula 
    ys  = range(T(0.0), T(1.0), length=N_eval)
    pts = [(y1, y2) for y1 in ys for y2 in ys]

    # Número de pontos total 
    N_pts = length(pts)

    # Deformações 
    deformacoes = [Vector{Vector{T}}(undef, N_pts) for _ in 1:3]

    # Pré-aloca o histórico de ativações para o backward
    As = [Matrix{T}(undef, size(redes[1].camadas[1].W, 2), N_pts * 5)]
    for c in redes[1].camadas
        push!(As, Matrix{T}(undef, size(c.W, 1), N_pts * 5))
    end

    # Pré-aloca os rascunhos de Z apenas para usar no forward
    Z_buffers = [Matrix{T}(undef, size(c.W, 1), N_pts * 5) for c in redes[1].camadas]
    
    # Loop pelas redes (k) e pelos pontos (p)
    for k in 1:3

        for p in 1:N_pts

            # Coordenadas do ponto 
            y1, y2 = pts[p][1], pts[p][2]

            # Deformação do ponto 
            ε_vec = Calcula_Deformacoes_pos_treino(y1, y2, modos[k], redes[k], N_modos, As, Z_buffers)

            # Monta em formato de Voigt
            deformacoes[k][p] = [ε_vec[1], ε_vec[2], T(2.0)*ε_vec[3]]

        end

    end

    # Aloca as tensões 
    tensoes = [Vector{T}(undef, 3) for _ in 1:3]

    # Loop pelos pontos (p)
    for p in 1:N_pts

        # Recupera coordenadas do ponto 
        y1, y2 = pts[p][1], pts[p][2]

        # Propriedades do material no ponto 
        E, ν = Propriedades_Material(y1, y2, mat_params)

        # Matriz constitutiva
        C_local = Matriz_Constitutiva(E, ν)
        
        # Tensões para cada rede 
        for i in 1:3
            tensoes[i] = C_local * deformacoes[i][p]
        end
        
        # Acumula a matriz homogeneizada
        for i in 1:3, j in 1:3
            CH[i, j] += dot(deformacoes[i][p], tensoes[j])
        end

    end
    
    # Calcula a média 
    CH ./= N_pts

    # E retorna o valor médio 
    return CH

end