# Tarefa

- Realize experimentos com 3  valores diferentes de NUM_STEPS e EMBEDDING_SIZE.

- Mostre as figuras com os processos direto e reverso de difusão.

**Entregáveis**:
1. Notebook `.ipynb`.
2. Relatório `.pdf`:
    - Reporte e comente os resultados no relatório.
    - Incluir gráficos gerados.


O objetivo do modelo é aproximar a distribuição "swiss roll", representada pelos pontos laranjas no gráfico abaixo.

![alt text](img/swiss_roll.png)

Será explorada a seguinte matriz de valores

NUM_STEPS | EMBEDDING_SIZE |
|----------:|---------------:|
| 50  | 16 |
| 50  | 32 |
| 50  | 2 |
| 500 | 2 |
| 500 | 16 |
| 500 | 32 |
| 1000 | 2 |
| 1000 | 16 |
| 1000 | 32 |

# T = 50
![alt text](img/direto_50.png)

# T = 500
![alt text](img/direto_500.png)
## Embedding size = 32
![alt text](img/loss_500_32.png)


# T = 1000
![alt text](img/direto_1000.png)
##

