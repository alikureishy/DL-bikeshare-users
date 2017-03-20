'''
Created on Mar 20, 2017

@author: safdar
'''
import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        self.activation_function = lambda x : (1 / (1 + np.exp(-x)))  # Sigmoid
    
    def train(self, inputs_list, targets_list):
        # Convert X list to 2d array
        X = np.array(inputs_list, ndmin=2)
        Y = targets_list #np.array(targets_list, ndmin=2).T
        
        hidden_grad = np.zeros(self.weights_input_to_hidden.shape)
        output_grad = np.zeros(self.weights_hidden_to_output.shape)
        
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output).T # signals into final output layer
        final_outputs = final_inputs # no activation function. use for regression.
        
        #### Implement the backward pass here ####
        ### Backward pass ###
        # TODO: Output error - Replace this value with your calculations.
        output_errors = Y - final_outputs # Output layer error is the difference between desired target and actual output.
        output_errors_term = output_errors[0][0] * 1 # activation function is f(x) = x. So, differentiation = 1
        output_grad += output_errors_term * hidden_outputs.T
        
        # TODO: Backpropagated error - Replace these values with your calculations.
        hidden_errors = np.dot(output_errors_term, self.weights_hidden_to_output) # errors propagated to the hidden layer
        hidden_errors_term = hidden_errors * (hidden_outputs * (1 - hidden_outputs)).T
        
        hidden_grad += (hidden_errors_term * X).T # hidden layer gradients
        
        # TODO: Update the weights - Replace these values with your calculations.
        self.weights_hidden_to_output += self.lr * output_grad # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * hidden_grad # update input-to-hidden weights with gradient descent step
 
        
    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2)
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(inputs, self.weights_input_to_hidden) # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output).T # signals into final output layer
        final_outputs = final_inputs # no activation function. use for regression.
        
        return final_outputs