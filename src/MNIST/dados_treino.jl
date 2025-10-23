# Struct Dados_Treino
# Contém os dados para treino da rede neural: entradas e saídas
struct Treino

    # Matriz de entradas para treino
    entradas_treino::Matrix

    # Matriz de saídas para treino
    saidas_esperadas_treino::Matrix

    # Matriz de entradas para teste 
    entradas_teste::Matrix

    # Matriz de saídas para teste
    saidas_esperadas_teste::Matrix

    # Labels de teste
    labels_teste::Vector
    
    # Função que inicializa todas as variáveis na struct
    function Treino(topologia::Vector{Int64})

        # Lê os dados de treino
        imagens_treino_unint = read_images("data/train-images-idx3-ubyte.gz")
        labels_treino_unint = read_labels("data/train-labels-idx1-ubyte.gz")

        # Lê os dados de teste
        imagens_teste_unint = read_images("data/t10k-images-idx3-ubyte.gz")
        labels_teste_unint = read_labels("data/t10k-labels-idx1-ubyte.gz")

        # Converte labels para Int64
        labels_treino = Int64.(labels_treino_unint)
        labels_teste = Int64.(labels_teste_unint)

        # Define matriz de saídas - cada coluna tem 10 posições, correspondendo ao número de 0 a 9
        saidas_esperadas_treino = zeros(Float64, 10, length(labels_treino))
        saidas_esperadas_teste = zeros(Float64, 10, length(labels_teste))

        # Transpõe o vetor de labels - esses dados são as saídas
        # Formato "one-hot" - todos as linhas zero exceto na posição da label correta, que é 1
        for i in eachindex(labels_treino)

            # Soma 1 devido ao 0 ser a primeira label
            saidas_esperadas_treino[labels_treino[i]+1, i] = 1.0

        end

        # Replica para o teste
        for i in eachindex(labels_teste)

            saidas_esperadas_teste[labels_teste[i]+1, i] = 1.0

        end

        # Converte os vetores de imagens para Float64 e normaliza 
        # Dados originais estão entre [0, 255], passamos para [0, 1]
        imagens_treino = Float64.(imagens_treino_unint) ./ 255.0
        imagens_teste = Float64.(imagens_teste_unint) ./ 255.0

        # Modifica dimensão das imagens de arrays de 3 dimensões (28 x 28 x length)
        # para matrizes "flat" de 784 (28^2) x length
        # Cada coluna é um vetor de entradas
        entradas_treino = reshape(imagens_treino, :, size(imagens_treino, 3)) 
        entradas_teste = reshape(imagens_teste, :, size(imagens_teste, 3))

        # Teste de consistência
        # Valida o número de entradas em relação a topologia
        if size(entradas_treino, 1) != topologia[1]

            error("Número de entradas digitadas está incorreto")

        end

        # Valida o número de saídas em relação a topologia
        if size(saidas_esperadas_treino, 1) != topologia[end]

            error("Número de saídas digitadas está incorreto")

        end

        # Returna os dados
        new(entradas_treino, saidas_esperadas_treino, entradas_teste, saidas_esperadas_teste, labels_teste)

    end

end

# Leitura dos arquivos com o banco de dados MNist
# Função auxiliar para leitura de big-endian UInt32
function read_uint32(io)
    bytes = read(io, 4)
    return reinterpret(UInt32, reverse(bytes))[1]
end

# Função para leitura dos arquivos de labels
function read_labels(filename::String)
    io = GzipDecompressorStream(open(filename))
    magic = read_uint32(io)
    magic != 0x00000801 && error("Número mágico inválido em arquivo de labels")
    n = Int(read_uint32(io))
    labels = Vector{UInt8}(undef, n)
    read!(io, labels)
    close(io)
    return labels
end

# Função para leitura dos arquivos de imagens
function read_images(filename::String)
    io = GzipDecompressorStream(open(filename))
    magic = read_uint32(io)
    magic != 0x00000803 && error("Número mágico inválido em arquivo de imagens")
    n = Int(read_uint32(io))
    rows = Int(read_uint32(io))
    cols = Int(read_uint32(io))
    total = n * rows * cols
    data = Vector{UInt8}(undef, total)
    read!(io, data)
    close(io)
    images = reshape(data, (rows, cols, n))  # 28×28×60000
    return images
end

