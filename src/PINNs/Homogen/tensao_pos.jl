# -----------------------------------------------------------------------------
#  Calcula as deformações para a parte do pós-processamento do C^H
# -----------------------------------------------------------------------------
function Calcula_Deformacoes_pos_treino(X::Matrix{T}, ε_macro::Matrix{T}, rede::Rede{T}, W::Vector{Matrix{T}},
                                        b::Vector{Vector{T}}, As::Vector{Matrix{T}}, Z_buffers::Vector{Matrix{T}},
                                        h::T) where {T<:AbstractFloat}

    # Roda o forward para todos os pontos de pertubação
    Forward_Rede_InPlace!(W, b, rede, X, Z_buffers, As)

    # Extraí os deslocamentos vetorizados
    # Perturba em y1
    u_x_mais  = As[end][:, 1:4:end]
    u_x_menos = As[end][:, 2:4:end]
    du_dy1 = (u_x_mais .- u_x_menos) ./ (T(2.0) * h)
    
    # Perturba em y2
    u_y_mais  = As[end][:, 3:4:end]
    u_y_menos = As[end][:, 4:4:end]
    du_dy2 = (u_y_mais .- u_y_menos) ./ (T(2.0) * h)
    
    # Calcula as deformações
    ε11 = ε_macro[1,1] .+ du_dy1[1, :]
    ε22 = ε_macro[2,2] .+ du_dy2[2, :]
    ε12 = ε_macro[1,2] .+ T(0.5) .* (du_dy2[1, :] .+ du_dy1[2, :])
    
    # Monta em formato de Voigt
    return [ε11, ε22, ε12]

end

# -----------------------------------------------------------------------------
#  Integração final de propriedades
# -----------------------------------------------------------------------------
function Calcula_Tensor_Homogeneizado(redes::Vector{Rede{T}}, modos::Vector{Matrix{T}}, 
                                      N_modos::Int, prob::String, mat_params::NamedTuple, N_eval::Int=50;
                                      h=T(1e-5)) where {T<:AbstractFloat}

    # Aloca a matriz 
    CH = zeros(T, 3, 3)

    # Gera N_eval pontos entre 0 e 1 para calcular a integral 
    # na célula 
    ys  = range(T(0.0), T(1.0), length=N_eval)
    pts = [(y1, y2) for y1 in ys for y2 in ys]

    # Número de pontos total 
    N_pts = length(pts)

    # Inicializa malha de pontos para integral (pré-alocada; evita hcat(x...) com
    # milhões de argumentos, que trava na especialização do splat)
    X_all = Matrix{T}(undef, 4 * N_modos, N_pts * 4)

    # Deformações
    deformacoes = [Vector{Vector{T}}(undef, N_pts) for _ in 1:3]

    # Loop pelos pontos para calcular as deformações
    for p in 1:N_pts

            # Coordenadas do ponto
            y1, y2 = pts[p][1], pts[p][2]

            # Lista com o stencil
            coords = ((y1+h, y2), (y1-h, y2), (y1, y2+h), (y1, y2-h))

            # Para cada ponto aplicamos a camada periódica
            for (j, pt) in enumerate(coords)
                X_all[:, 4*(p-1) + j] .= Camada_Periodica(pt[1], pt[2], N_modos)
            end

    end

    # Pré-aloca o histórico de ativações para o backward
    As = [Matrix{T}(undef, size(redes[1].camadas[1].W, 2), N_pts * 4)]
    for c in redes[1].camadas
        push!(As, Matrix{T}(undef, size(c.W, 1), N_pts * 4))
    end

    # Pré-aloca os rascunhos de Z apenas para usar no forward
    Z_buffers = [Matrix{T}(undef, size(c.W, 1), N_pts * 4) for c in redes[1].camadas]

    # Loop pelas redes (k)
    for k in 1:3

            # Copia os pesos e bias atuais da rede para chamada da função
            W = [copy(c.W) for c in redes[k].camadas]
            b = [copy(c.b) for c in redes[k].camadas]

            # Deformação do ponto 
            ε_vec = Calcula_Deformacoes_pos_treino(X_all, modos[k], redes[k], W, b, As, Z_buffers, h)

            # Monta em formato de Voigt
            for p in 1:N_pts
                deformacoes[k][p] = [ε_vec[1][p], ε_vec[2][p], T(2.0)*ε_vec[3][p]]
            end

    end

    # Aloca as tensões 
    tensoes = [Vector{T}(undef, 3) for _ in 1:3]

    # Loop pelos pontos (p)
    for p in 1:N_pts

        # Recupera coordenadas do ponto 
        y1, y2 = pts[p][1], pts[p][2]

        # Propriedades do material no ponto 
        props_simbolo = Symbol("Propriedades_Material_"*prob)
        E, ν = getfield(Main, props_simbolo)(y1, y2, mat_params)

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
