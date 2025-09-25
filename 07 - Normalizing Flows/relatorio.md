# Tarefa

1. Replique o experimeto com o conjunto de dados `make_circles` do scikit-learn:
    ``` python
    data, _ = datasets.make_circles(n_samples=10000, noise=0.02, random_state=42)
    ```

- Treine o mesmo modelo por um mesmo número de épocas (ex.: 100) com 4  taxas de aprendizado (`lr`) diferentes: $0.1, 0.01, 0.001, 0.0001$.
    - Para cada modelo, gere 1000 amostras e produza `plotar_heatmap` (densidade + amostras).
    - Capture a perda por época.
    - Construa em um único gráfico: época x perda para os 4 valores.
    - Analise as curvas.
    - Comente sobre a geração dos dados sintéticos desses modelos.

2. Utilizando o seguinte conjunto de dados:

    ``` python
    X1, _  = datasets.make_circles(n_samples=10000, noise=0.02, random_state=42)
    X2, _ = datasets.make_moons(10000, noise=0.05)
    X3, _ = datasets.make_blobs(10000, cluster_std=0.05, center_box=(-3,3), centers=np.array([[0,0], [2, -1], [3, 0]]))
    X2[:, 0] += 1
    X2[:, 1] += 1.5
    data = np.vstack((X1, X2, X3))
    ```

- Fixe `lr = 1e-4` e treine 3 arquiteturas RelNVP:
    - 2 camadas de acoplamento (baseline)
    - 4 camadas de acoplamento
    - 6 camadas de acoplamento
 - Para cada modelo, gere 1000 amostras e produza `plotar_heatmap` (densidade + amostras).
- Capture a perda por época.
- Construa em um único gráfico: época x perda para os 3 valores.

- Analise as curvas.
- Comente sobre a geração dos dados sintéticos desses modelos.

**Entregáveis**:
1. Notebook `.ipynb`.
2. Relatório `.pdf`:

    - Reporte e comente os resultados no relatório.

    - Incluir gráficos gerados.