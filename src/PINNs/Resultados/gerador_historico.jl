using DelimitedFiles
using Plots

# Cria o gráfico base
plot()

# Le todos os arquivos teste_rede*.txt do diretório e coloca no gráfico
arquivos = filter(contains("teste_rede_"), readdir())

for arquivo in arquivos

    println("Lendo $arquivo")

    # Le os dados 
    dados = readdlm(arquivo)

    # Adiciona no gráfico
    display(plot!(dados',label=""))

end