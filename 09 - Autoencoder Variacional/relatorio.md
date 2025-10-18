# Tarefa

- Conjunto de dados a ser utilizado MNIST (ao invés do Fashion MNIST):

    ``` python
    train_data  = datasets.MNIST(root="data", train=True, download=True)     # carrega conjunto de treinamento
    test_data   = datasets.MNIST(root="data", train=False, download=True)    # carrega conjunto de teste
    ```

1. Implemente uma versão convolucional do VAE.
2. Treine a versão convolucional do VAE com os valores diferentes para o parâmetro $\beta$ (ex.: $\beta=\in\{0, 0.1, 100\}$).
    - Mostre o termo de reconstrução (MSE), o termo de regularização (KL) e a função de perda por época (usando `curvas_de_treinamento`).
    - Mostre as reconstruções (usando `mostrar_reconstrucoes`)
    - Mostre o espaço latente (usando visualizar_espaco_latente e `visualizar_grade_latente`)
    - Gere amostras sintéticas (usando `gerar_amostras_aleatorias`)

**Entregáveis**:
1. Notebook `.ipynb`.
2. Relatório `.pdf`:
    - Reporte e comente os resultados no relatório.
    - Incluir gráficos gerados.