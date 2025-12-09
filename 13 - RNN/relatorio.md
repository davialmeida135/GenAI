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
Para o primeiro prompt, as gerações com baixas temperaturas apresentaram uma boa estrutura sintática, mas semânticamente não fizeram sentido. A geração com temperatura alta acabou por ser a que mas fez sentido no contexto da query, apesar de ter um número '5' solto no meio da frase.
- "recipe for roasted vegetables"
    - Temp = 0: `in a large bowl , combine the sugar and salt`
    - Temp = 0.2: `in a bowl whisk together the flour , the baking`
    - Temp = 1: `preheat the oven to 350°f . 5 . bake for`

Já para este prompt, o resultado foi inverso. As respostas de temperatura mais baixa foram as que mais fizeram sentido para o contexto. O texto gerado com temperatura 1 não fez o menor sentido para a receita proposta.

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
A GRU se mostrou bem melhor no primeiro prompt, com todas as respostas, mesmo que diferentes, fazendo algum sentido para a receita sugerida.
- "recipe for roasted vegetables"
    - Temp = 0: `preheat oven to 450°f . toss potatoes with 1 /`
    - Temp = 0.2: `preheat oven to 450°f . toss potatoes with 1 /`
    - Temp = 1: `fill a large saucepan with 1 water and drain well`

Já para esta receita, as respostas da GRU não fizeram o menor sentido em nenhum dos testes.
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

| Modelo | Perplexity | Melhoria vs RNN |
|--------|-----------|-----------------|
| **RNN Clássica** | 13.10 | - (baseline) |
| **GRU** | 10.31 | 21.3% melhor |
| **LSTM** | 10.31 | 21.3% melhor |

Embora GRU e LSTM tenham perplexity idêntica, a qualidade percebida varia:
- **LSTM:** Gerações mais estruturadas (numeração, passos lógicos)
- **GRU:** Gerações funcionais, mas ocasionalmente comete erros semânticos, como "cocktail shaker"
