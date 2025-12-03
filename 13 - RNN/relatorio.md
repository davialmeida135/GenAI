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
self.embedding = nn.Embedding(
    vocab_size,
    embedding_dim,
    padding_idx=0
)
self.rnn = nn.RNN(
    input_size=embedding_dim,
    hidden_size=hidden_size,
    batch_first=True
)
self.fc = nn.Linear(
    hidden_size,
    vocab_size
)
```

### Prompts utilizados
- "recipe for roasted vegetables"
    - Temp = 0: `in a large bowl , combine the sugar and salt`
    - Temp = 0.2: `in a bowl whisk together the flour , the baking`
    - Temp = 1: `preheat the oven to 350°f . 5 . bake for`

- "recipe for mac and cheese"
    - Temp = 0: `preheat oven to 350°f . butter a 9 -`
    - Temp = 0.2: `preheat oven to 350°f . butter and flour 9`
    - Temp = 1: `stir anise ( but sauce can be kept overed`

## GRU
### Implementação
```python
self.embedding = nn.Embedding(
    vocab_size,
    embedding_dim,
    padding_idx=0
)
self.gru = nn.GRU(
    input_size=embedding_dim,
    hidden_size=hidden_size,
    batch_first=True
)
self.fc = nn.Linear(
    hidden_size,
    vocab_size
)
```
### Prompts utilizados
- "recipe for roasted vegetables"
    - Temp = 0: `preheat oven to 450°f . toss potatoes with 1 /`
    - Temp = 0.2: `preheat oven to 450°f . toss potatoes with 1 /`
    - Temp = 1: `fill a large saucepan with 1 water and drain well`

- "recipe for mac and cheese"
    - Temp = 0: `combine all ingredients in a cocktail shaker and shake`
    - Temp = 0.2: `combine all ingredients in a cocktail shaker and shake`
    - Temp = 1: `line the bottom rack with two 1 - inch`

## LSTM
### Implementação
```python
self.embedding = nn.Embedding(
    vocab_size,
    embedding_dim,
    padding_idx=0
)
self.lstm = nn.LSTM(
    input_size=embedding_dim,
    hidden_size=hidden_size,
    batch_first=True
)
self.fc = nn.Linear(
    hidden_size,
    vocab_size
)
```
### Prompts utilizados
- "recipe for roasted vegetables"
    - Temp = 0: `preheat oven to 350°f . butter a large baking sheet`
    - Temp = 0.2: `preheat oven to 350°f . rinse turkey inside and out`
    - Temp = 1: `preheat oven to 475°f . brush each 4 slightly inside`

- "recipe for mac and cheese"
    - Temp = 0: `1 . preheat the oven to 350°f . in`
    - Temp = 0.2: `1 . preheat the oven to 350°f . in`
    - Temp = 1: `1 . preheat the oven to 425°f . slowly`

## Perplexity

- RNN: 13.10
- LSTM: 10.31
- GRU: 10.31
