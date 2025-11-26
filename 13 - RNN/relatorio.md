# Tarefa

1. Use uma [Gated recurrent units (GRUs)](https://en.wikipedia.org/wiki/Gated_recurrent_unit) e uma RNN clássica para replicar os experimentos realizados com a LSTM:
- [GRU em Pytorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [RNN clássica em Pytorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)

2. Gere receitas com os modelos variando o valor do hiperparâmetro temperatura.

3. Usando *perplexity*, compare os resultados obtidos pelos modelos: GRU vs. RNN clássica vs. LSTM.

**Entregáveis**:
1. Notebook `.ipynb`
2. Relatório `.pdf`

## RNN Clássica
### Implementação
```python
RNN
```
### Prompts utilizados
- "recipe for roasted vegetables"
    - Temp = 0  
    - Temp = 0.5
    - Temp = 1
    - Temp = 2

- "recipe for mac and cheese"

## GRU
### Implementação
```python
GRU
```
### Prompts utilizados
- "recipe for roasted vegetables"
    - Temp = 0
    - Temp = 0.5
    - Temp = 1
    - Temp = 2

- "recipe for mac and cheese"

## Perplexity
