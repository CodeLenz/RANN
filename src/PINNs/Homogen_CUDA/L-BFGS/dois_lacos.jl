#
# 
#                            Recursão em Dois Laços
# 
#
# Calcula o produto
#
#     r  ≈  (H^k)^{-1} · q_in
#
# em que q_in é tipicamente o vetor gradiente ∇f(x^k) e a memória contém os
# m pares {(s^i, y^i, ρ^i)} mais recentes. O algoritmo vem do desdobramento
# recursivo da forma produto da atualização BFGS, que ao ser aplicada
# sucessivamente produz duas varreduras :
#
#   • Laço retroativo: aplica V_{k-1}^T · V_{k-2}^T · ... · V_{k-m}^T sobre q,
#     acumulando os escalares τ_i que serão reutilizados adiante.
#
#   • Laço progressivo: aplica V_{k-m} · ... · V_{k-2} · V_{k-1} e adiciona
#     os termos aditivos τ_i · s^i recuperados da etapa anterior.
#
#
function Dois_Lacos(q_in::AbstractVector{T}, memoria::Vector) where {T<:AbstractFloat}

    # O laço retroativo sobrescreve q iterativamente. Sem a cópia o vetor
    # gradiente externo a rotina seria destruído (sobrescrito).
    q     = copy(q_in)
    m_len = length(memoria)

    # Aloca r (mesmo tipo de q: CPU ou GPU)
    r = zero(q)

    # Os escalares τ_i são CALCULADOS no laço retroativo e USADOS no laço
    # progressivo. O armazenamento é necesário pois quando o laço
    # progressivo precisa deles, o vetor q já sofreu m modificações sucessivas e
    # os τ_i não podem mais ser reconstruídos a posteriori. Este vetor auxiliar
    # de tamanho m é o único "custo extra" da recursão em relação a uma
    # aplicação matricial direta.
    τ_arr = zeros(T, m_len)

    #
    # Laço retroativo
    # 
    # Percorre a memória do par mais recente ao mais antigo. Cada iteração
    # implementa a ação de um único V_i^T = (I - ρ_i · y^i · (s^i)^T)
    # sobre o vetor corrente, usando 
    #
    #     V_i^T · q  =  q - ρ_i · ((s^i)^T q) · y^i  =  q - τ_i · y^i
    #
    # Ao final da varredura, q corresponde ao vetor
    # sobre o qual incidiria a aproximação inicial (H_k^0)^{-1} em uma
    # expansão matricial não truncada.
    #
    for i in m_len:-1:1

        # Recupera da memória
        s_i, y_i, ρ_i = memoria[i]

        # Calcula o produto 
        τ_i = ρ_i * dot(s_i, q)

        # Guarda na memória 
        τ_arr[i] = τ_i      
        
        # Atualiza o vetor 
        q .= q .- τ_i * y_i
        
    end

    # 
    # Escalonamento central: fator de Shanno-Phua
    # 
    # Em vez de uma matriz inicial estática, adota-se a diagonal 
    # (H_k^0)^{-1} = γ_k · I, com o fator escalar
    #
    #     γ_k = (s^{k-1})^T y^{k-1} / (y^{k-1})^T y^{k-1}
    #
    # Esta expressão é o quociente de Rayleigh da matriz Hessiana inversa
    # avaliado na direção y^{k-1}, ou seja, uma estimativa local do
    # autovalor dominante de (H^k)^{-1} ao longo da última direção percorrida.
    #
    # Isso  decorre da aproximação secante
    # s^{k-1} ≈ (H^k)^{-1} y^{k-1}.
    #
    # Usamos apenas o par mais recente (e não uma média sobre a memória) pois
    # a Hessiana muda ao longo das iterações, e a informação 
    # local é tipicamente a mais representativa da curvatura atual.
    #
    s_recent, y_recent, _ = memoria[end]
    yy = norm(y_recent)^2

    # Se ||y^{k-1}||² ≈ 0, estamos próximos de um ponto
    # estacionário ou em estagnação severa, então o quociente seria mal-condicionado
    # (ou até 0/0 ). Por segurança, γ_k = 1 retorna ao comportamento
    # de steepest descent escalonado pela identidade.
    if yy < T(1e-10)
        γ_k = one(T)
    else
        γ_k = dot(s_recent, y_recent) / yy
    end

    # Início da reconstrução progressiva
    # r = γ_k · q corresponde ao produto(H_k^0)^{-1} · q já calculado
    r = γ_k * q

    # 
    # Laço progressivo
    # 
    # Varredura agora em ordem cronológica (do par mais antigo ao
    # mais recente). Cada iteração executa DUAS operações simultâneas que se
    # fundem em uma única atualização
    #
    #   (a) Aplicação de V_i = (I - ρ_i · s^i · (y^i)^T) sobre r:
    #          V_i · r  =  r - ρ_i · ((y^i)^T r) · s^i  =  r - β_i · s^i
    #
    #   (b) Adição do termo projetado τ_i · s^i que ficou "pendente" do laço
    #       retroativo (vindo da parcela ρ_k · s^k (s^k)^T da forma produto).
    #
    # Como ambas as contribuições são múltiplos de s^i
    #
    #       r  ←  r + s^i · (τ_i - β_i)
    #
    # Pontos a observar:
    #   - τ_i é REUTILIZADO (leitura do cache construído no laço 1).
    #   - β_i é CALCULADO a cada passagem, pois depende do estado corrente
    #     de r, que está sendo alterado a cada iteração
    #   - τ_i usa o produto (s^i)^T · q (dovetor de trabalho do laço 1)
    #   - β_i usa (y^i)^T · r (do vetor de trabalho do laço 2). 
    #   - Os mesmos ρ_i aparecem em ambos, mas contraídos contra vetores diferentes.
    #
    for i in 1:m_len

        # Recupera da memória
        s_i, y_i, ρ_i = memoria[i]

        # Calcula β
        τ_i = τ_arr[i]
        β_i = ρ_i * dot(y_i, r)

        # Acumula em r
        r .= r .+ s_i * (τ_i - β_i)
    end

    # Devolve r ≈ (H^k)^{-1} · q_in 
    return r
    
end