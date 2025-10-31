# Tarefa

- Realize experimentos com 3  valores diferentes de NUM_STEPS e EMBEDDING_SIZE.

- Mostre as figuras com os processos direto e reverso de difusão.

O objetivo do modelo é aproximar a distribuição "swiss roll", representada pelos pontos laranjas no gráfico abaixo.

![alt text](img/swiss_roll.png)

Será explorada a seguinte matriz de hiperparâmetros:

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
# Resultados

Com os experimentos realizados, foi possível perceber a importância dos embeddings temporais para o aprendizado dos modelos de difusão. Os modelos com embeddings de tamanho 2 se mantiveram acima de 0.5 de perda durante todo o treinamento e, ao fim, não foram capazes de gerar uma distribuição próxima da esperada. Já os modelos com embedding size 16 e 32 apresentaram loss similar e boas reconstruções para 50, 500 e 1000 passos.

## T = 50
![alt text](img/direto_50.png)
### Embedding size = 2
![alt text](img/loss_50_2.png)
![alt text](img/inverso_50_2.png)
### Embedding size = 16
![alt text](img/loss_50_16.png)
![alt text](img/inverso_50_16.png)
### Embedding size = 32
![alt text](img/loss_50_32.png)
![alt text](img/inverso_50_32.png)

## T = 500
![alt text](img/direto_500.png)
### Embedding size = 2
![alt text](img/loss_500_2.png)
![alt text](img/inverso_500_2.png)
### Embedding size = 16
![alt text](img/loss_500_16.png)
![alt text](img/inverso_500_16.png)
### Embedding size = 32
![alt text](img/loss_500_32.png)
![alt text](img/inverso_500_32.png)


## T = 1000
![alt text](img/direto_1000.png)
### Embedding size = 2
![alt text](img/loss_1000_2.png)
![alt text](img/inverso_1000_2.png)
### Embedding size = 16
![alt text](img/loss_1000_16.png)
![alt text](img/inverso_1000_16.png)
### Embedding size = 32
![alt text](img/loss_1000_32.png)
![alt text](img/inverso_1000_32.png)
