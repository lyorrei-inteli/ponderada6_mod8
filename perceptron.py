import numpy as np

class Perceptron:
    # Inicialização do Perceptron com taxa de aprendizado, número de iterações e limiar
    def __init__(self, learning_rate=0.1, n_iterations=100, threshold=0.5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.weights = np.zeros(2)
        self.bias = 0

    # Função de ativação: função degrau
    def activation_function(self, x):
        return 1 if x >= self.threshold else 0

    # Faz a predição baseada nas entradas
    def predict(self, inputs):
        # Calcula a soma ponderada das entradas
        linear_output = np.dot(inputs, self.weights) + self.bias
        # Aplica a função degrau para determinar a saída
        y_predicted = self.activation_function(linear_output)
        return y_predicted

    # Treina o perceptron ajustando pesos e viés
    def train(self, X, y):
        for _ in range(self.n_iterations):
            for x, y_true in zip(X, y):
                y_pred = self.predict(x)
                error = y_true - y_pred
                self.weights += error * self.learning_rate * x
                self.bias += error * self.learning_rate

# Função para testar o perceptron
def test_perceptron(perceptron, X, y, logic_gate):
    print(f"Testing {logic_gate} Gate")
    for x, y_true in zip(X, y):
        y_pred = perceptron.predict(x)
        print(f"Input: {x}, Expected: {y_true}, Predicted: {y_pred}")

# Exemplo de uso
if __name__ == "__main__":
    '''
        ------AND------
    '''
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1]) 

    # Treinando o Perceptron
    perceptron_and = Perceptron()
    perceptron_and.train(X_and, y_and)
    test_perceptron(perceptron_and, X_and, y_and, "AND")

    '''
        ------OR------
    '''

    # Dados para a porta OR
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1]) 

    # Treinando o Perceptron
    perceptron_or = Perceptron()
    perceptron_or.train(X_or, y_or)
    test_perceptron(perceptron_or, X_or, y_or, "OR")

    '''
        ------NAND------
    '''
    # Dados para a porta NAND
    X_nand = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_nand = np.array([1, 1, 1, 0]) 

    # Treinando o Perceptron
    perceptron_nand = Perceptron()
    perceptron_nand.train(X_nand, y_nand)
    test_perceptron(perceptron_nand, X_nand, y_nand, "NAND")

    '''
        ------XOR------
    '''

    # Dados para a porta XOR
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0]) 

    # Treinando o Perceptron
    perceptron_xor = Perceptron()
    perceptron_xor.train(X_xor, y_xor)
    test_perceptron(perceptron_xor, X_xor, y_xor, "XOR")