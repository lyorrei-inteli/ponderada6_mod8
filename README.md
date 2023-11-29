# Documentação do Projeto Perceptron para Portas Lógicas

## Objetivo

Este projeto foi desenvolvido como uma atividade de sala de aula com o objetivo de implementar um perceptron que pode ser treinado para simular o comportamento das seguintes portas lógicas: AND, OR, NAND e XOR. O foco principal é demonstrar a aplicação do algoritmo de aprendizado do perceptron em problemas lógicos simples.

## Como Instalar e Rodar o Sistema

### Requisitos
- Python 3.x
- Biblioteca NumPy

### Instalação
```bash
python3 -m venv .venv
```

Dê source na venv para ativar o ambiente virtual e instale as dependências:
```bash
source .venv/bin/activate or .venv\Scripts\activate
```

```bash
 pip install numpy
```

### Execução

Para executar o script, utilize o comando:
```bash
python3 perceptron.py
```
Este comando irá rodar o script `perceptron.py`, que contém a implementação do perceptron e testes para as portas lógicas AND, OR, NAND e XOR.

## Padrão de Qualidade e Explicação do Código

### Implementação do Perceptron

O perceptron foi implementado com as seguintes características principais:

- **Pesos (Weights) e Bias:**
  ```python
  self.weights = np.zeros(2)
  self.bias = 0
  ```
  Inicialmente, os pesos são configurados para zero, e o bias também é inicializado como zero. Eles são ajustados ao longo do processo de treinamento.

- **Função de Ativação:**
  ```python
  def activation_function(self, x):
      return 1 if x >= self.threshold else 0
  ```
  Esta é uma função degrau que determina a saída do perceptron. Se a soma ponderada das entradas for maior ou igual ao limiar, a saída será 1; caso contrário, será 0.

- **Método de Predição:**
  ```python
  def predict(self, inputs):
      linear_output = np.dot(inputs, self.weights) + self.bias
      y_predicted = self.activation_function(linear_output)
      return y_predicted
  ```
  Calcula a saída do perceptron dadas as entradas. A soma ponderada das entradas e dos pesos, mais o bias, é passada para a função de ativação.

- **Método de Treinamento:**
  ```python
  for x, y_true in zip(X, y):
      y_pred = self.predict(x)
      error = y_true - y_pred
      self.weights += error * self.learning_rate * x
      self.bias += error * self.learning_rate
  ```
  Durante o treinamento, o perceptron ajusta seus pesos e bias com base no erro (diferença entre a saída prevista e a saída real). O erro é usado para fazer ajustes proporcionais aos pesos e ao bias, guiados pela taxa de aprendizado.

### Treinamento para as Portas Lógicas
Para cada porta lógica, o perceptron é treinado com um conjunto de dados específico que representa a verdade lógica da porta. Após o treinamento, o perceptron é testado com os mesmos dados para verificar se aprendeu corretamente a função lógica. Os resultados, como mencionado anteriormente, são precisos para AND, OR e NAND, mas não para XOR, destacando uma limitação importante do modelo de perceptron simples.

## Resultados das Portas Lógicas

### AND, OR e NAND Gates

Para as portas lógicas AND, OR e NAND, o perceptron treinado conseguiu reproduzir o comportamento esperado. Os resultados obtidos foram:

- **AND Gate**
  - Input: [0 0], Expected: 0, Predicted: 0
  - Input: [0 1], Expected: 0, Predicted: 0
  - Input: [1 0], Expected: 0, Predicted: 0
  - Input: [1 1], Expected: 1, Predicted: 1

- **OR Gate**
  - Input: [0 0], Expected: 0, Predicted: 0
  - Input: [0 1], Expected: 1, Predicted: 1
  - Input: [1 0], Expected: 1, Predicted: 1
  - Input: [1 1], Expected: 1, Predicted: 1

- **NAND Gate**
  - Input: [0 0], Expected: 1, Predicted: 1
  - Input: [0 1], Expected: 1, Predicted: 1
  - Input: [1 0], Expected: 1, Predicted: 1
  - Input: [1 1], Expected: 0, Predicted: 0

Estes resultados indicam que o perceptron foi capaz de aprender corretamente as funções lógicas básicas, demonstrando a eficácia do algoritmo de aprendizado.

### XOR Gate
No entanto, para a porta XOR, o perceptron não conseguiu reproduzir o comportamento esperado. Os resultados foram:

- **XOR Gate**
  - Input: [0 0], Expected: 0, Predicted: 1
  - Input: [0 1], Expected: 1, Predicted: 1
  - Input: [1 0], Expected: 1, Predicted: 0
  - Input: [1 1], Expected: 0, Predicted: 0

Este resultado é um exemplo clássico das limitações de um perceptron simples. A porta XOR é um problema não linearmente separável, o que significa que não é possível traçar uma única linha reta (ou hiperplano, em dimensões maiores) que separe perfeitamente as saídas 0 e 1. Isso demonstra a necessidade de redes neurais mais complexas para resolver certos tipos de problemas.

## Demonstração
https://youtu.be/oXLmb27hax0