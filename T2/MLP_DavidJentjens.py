from network_layers import *
from activation_functions import *
from error_functions import *

class Network:
    def __init__(self):
        self.layers = []
        self.error_function = mse
        self.error_function_prime = mse_prime

    def layer(self, layer):
        self.layers.append(layer)

    def use(self, error_function, error_function_prime):
        self.error_function = error_function
        self.error_function_prime = error_function_prime

    def fit(self, x_train, y_train, epochs=10, mini_batch=1000, learning_rate=0.1):
        error_list = []
        x_batches = np.array( [x_train[i * mini_batch:(i + 1) * mini_batch] for i in range((len(x_train) + mini_batch - 1) // mini_batch )] )
        y_batches = np.array( [y_train[i * mini_batch:(i + 1) * mini_batch] for i in range((len(y_train) + mini_batch - 1) // mini_batch )] )
        
        print(x_train.shape)
        print(x_batches.shape)

        for epoch in range(epochs):
            error = 0
            for x, y in zip(x_batches, y_batches):
                print(x.shape)
                # Forward propagation
                output = x
                for layer in self.layers:
                    output = layer.forward(output)
                    
                # Prediction error calculation
                error += self.error_function(y, output)
                
                # Backward propagation
                grad = self.error_function_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)
            print('Epoch=%d/%d, Error=%f' % (epoch+1, epochs, error))
            error_list.append(error)
        return error_list

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
            
        return output