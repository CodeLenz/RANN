# Função que calcula as labels estimadas pelo modelo para os dados de teste 
# e compara com o esperando, calculando a acurácia da rede neural
function acuracia(rede::Rede, treino::Treino, pesos::Vector{Vector{T}}, bias::Vector{Vector{T}}) where T

    # Aloca contador da quantidade de acertos de labels
    cont_acertos = 0.0

    # Aloca vetor para guardar as labels estimadas pela rede neural
    labels_teste_calc = zeros(Int64, length(treino.labels_teste))
    
    # Calcula a saida da rede para as entradas de teste
    for col=1:size(treino.entradas_teste,2)
 
       # Forward loop pelos dados de entrada; gera o logit e depois a saída final pela função Softmax
       logits1 =  RNA(rede, pesos, bias, treino.entradas_teste[:,col])
       s1 = Softmax(logits1) 
       
       # Descobre a posição de valor máximo, que é o label estimado pela rede e aloca no vetor de labels
       # Desconta 1 da posição devido começar em 0 (0 está na posição 1 do vetor, 1 está na posição 2 do vetor, ...)
       _, pos = findmax(s1)
       labels_teste_calc[col] = pos - 1

       # Compara a label da rede neural com o dado verdadeiro
       if labels_teste_calc[col] == treino.labels_teste[col]

            # Atualiza o contador de acertos
            cont_acertos = cont_acertos + 1

       end

    end

    # Calcula a acurácia em percentual
    acuracia = 100.0 * cont_acertos / length(labels_teste_calc)

    # Retorna a acurácia e o vetor de labels calculado
    return acuracia, labels_teste_calc

end