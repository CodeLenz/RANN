using CodecZlib # Leitura de arquivos comprimidos (.gz)
using Plots # Geração de gráficos
using StatsBase # Função Sample

# Função auxiliar para leitura de big-endian UInt32
function read_uint32(io)
    bytes = read(io, 4)
    return reinterpret(UInt32, reverse(bytes))[1]
end

# Função para leitura dos arquivos de labels
function read_labels(filename::String)
    io = GzipDecompressorStream(open(filename))
    magic = read_uint32(io)
    magic != 0x00000801 && error("Invalid magic number in label file")
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
    magic != 0x00000803 && error("Invalid magic number in image file")
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

# Lê os dados de treino
imagens_treino_unint = read_images("train-images-idx3-ubyte.gz")
labels_treino_unint = read_labels("train-labels-idx1-ubyte.gz")

# Lê os dados de teste
imagens_teste_unint = read_images("t10k-images-idx3-ubyte.gz")
labels_teste_unint = read_labels("t10k-labels-idx1-ubyte.gz")

# Converte labels para Float64
labels_treino = Float64.(labels_treino_unint)
labels_teste = Float64.(labels_teste_unint)

# Converte para Float64 e normaliza os vetores de imagens
# Dados originais estão entre [0, 255], passamos para [0, 1]
imagens_treino = Float64.(imagens_treino_unint) ./ 255.0
imagens_teste = Float64.(imagens_teste_unint) ./ 255.0

# Modifica dimensão das imagens de arrays de 3 dimensões (28 x 28 x length)
# para matrizes "flat" de 784 (28^2) x length
# Cada coluna é um vetor de entradas
imagens_treino_flat = reshape(imagens_treino, :, size(imagens_treino, 3)) 
imagens_teste_flat = reshape(imagens_teste, :, size(imagens_teste, 3)) 

# Gera um gráfico de um amostral de imagens para análise

# Cria vetor vazio para armazenar as imagens
imgs = []

# Gera gráfico
plot()

# Seleciona amostral de 16 imagens, sem replacement (sempre valores únicos)
idx = sample(1:size(imagens_treino, 3), 16; replace=false)

# Loop pelas amostras
for i in idx

    # Seleciona matriz da imagem e inverte orientação das colunas
    # Necessário pois heatmap plota os valores de baixo para cima
    img = reverse(imagens_treino[:, :, i], dims=2)

    # Extraí o label
    label = Int(labels_treino[i])

    # Gera o gráfico de mapa de calor cinzento
    p = heatmap(img', c=:grays, axis=nothing, ticks=nothing, title="[$i]: $label")

    # Incluí o mapa de calor dentro do vetor de imagens
    push!(imgs, p)

end

# Gera o gráfico com todas as imagens em formato de grid 4 x 4
plot!(imgs..., layout = (4, 4), size=(600, 600))