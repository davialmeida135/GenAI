# Tarefa

1. Replique o experimeto com o conjunto de dados `MNIST` do Pytorch:
    ``` python
    train_full  = datasets.MNIST(root="data", train=True, download=True) 
    test_ds     = datasets.MNIST(root="data", train=False, download=True)
    ```

  e os seguintes Autoencoders: convolucional, esparso e por remoção de ruído. Para cada modelo teste 3 valores distintos para a dimensionalidade do espaço latente.

- **Capture a perda (MSE) por época.** 
- **Mostre as reconstruções**

**Entregáveis**:
1. Notebook `.ipynb`.
2. Relatório `.pdf`:
    - Reporte e comente os resultados no relatório.
    - Incluir gráficos gerados.


# Autoencoder determinístico convolucional

## z = 2
![alt text](img/gen_conv_2.png)

![alt text](img/loss_conv_2.png)
## z = 32

![alt text](img/gen_conv_32.png)

![alt text](img/loss_conv_2.png)
## z = 64
![alt text](img/gen_conv_64.png)

![alt text](img/gen_conv_64.png)
# Autoencoder determinístico esparso

## z = 64
![alt text](img/gen_sparse_64.png)

![alt text](img/loss_sparse_64.png)
## z = 300
![alt text](img/gen_sparse_300.png)

![alt text](img/loss_sparse_300.png)
## z = 600
![alt text](img/gen_sparse_600.png)

![alt text](img/loss_sparse_600.png)

# Autoencoder determinístico de remoção de ruído

## z = 2
![alt text](img/gen_drop_2.png)

![alt text](img/loss_drop_2.png)
## z = 32
![alt text](img/gen_drop_32.png)

![alt text](img/loss_drop_32.png)
## z = 64
![alt text](img/gen_drop_64.png)

![alt text](img/loss_drop_64.png)