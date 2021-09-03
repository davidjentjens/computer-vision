import numpy as np

class BaseLayer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

class FCLayer(BaseLayer):
    def __init__(self, input_size, output_size) -> None:
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_error = np.dot(output_gradient, self.weights.T)
        weights_error = np.dot(self.input.T, output_gradient)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_gradient

        return input_error

class ActLayer(BaseLayer):
    def __init__(self, activation_function, activation_prime):
        self.activation_function = activation_function
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return self.activation_prime(self.input) * output_gradient