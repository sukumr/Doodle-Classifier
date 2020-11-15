import numpy

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

class Neural_Network:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learn_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learn_rate

        # weight matrix input to hidden
        self.weights_ih = numpy.random.uniform(
            -1, 1, size=(self.hidden_nodes, self.input_nodes)
        )

        # weight matrix hidden to output 
        self.weights_ho = numpy.random.uniform(
            -1, 1, size=(self.output_nodes, self.hidden_nodes)
        )

        # bias hidden nodes
        self.bias_h = numpy.random.uniform(-1, 1, size=(self.hidden_nodes, 1))

        # bias output nodes
        self.bias_o = numpy.random.uniform(-1, 1, size=(self.output_nodes, 1))


    def predict(self, input_arr):
        # converting list to a column matrix
        inputs = numpy.array(input_arr).reshape(len(input_arr), 1)
        # matrix multiplication 
        hidden = numpy.dot(self.weights_ih, inputs)
        # adding bias
        hidden = numpy.add(hidden, self.bias_h)
        # activation
        hidden_output = sigmoid(hidden)

        # matrix multiplication 
        outputs = numpy.dot(self.weights_ho, hidden_output)
        # adding bias
        outputs = numpy.add(outputs, self.bias_o)   
        # activation       
        outputs = sigmoid(outputs)

        return outputs.reshape(1, len(outputs))


    def train(self, input_arr, target_arr):
        inputs = numpy.array(input_arr).reshape(len(input_arr), 1)
        hidden = numpy.dot(self.weights_ih, inputs) 
        hidden = numpy.add(hidden, self.bias_h)
        hidden_output = sigmoid(hidden)

        outputs = numpy.dot(self.weights_ho, hidden_output)
        outputs = numpy.add(outputs, self.bias_o)     
        outputs = sigmoid(outputs)
        
        targets = numpy.array(target_arr).reshape(len(target_arr), 1)

        # error = target - guess
        output_errors = numpy.subtract(targets, outputs)
        gradients = self.learning_rate * numpy.multiply(output_errors, numpy.multiply(outputs, numpy.subtract(1, outputs)))
        weights_ho_deltas = numpy.dot(gradients, numpy.transpose(hidden_output)) 

        # new weight matrix hidden to output 
        self.weights_ho = numpy.add(weights_ho_deltas, self.weights_ho) 
        # new biases at output nodes
        self.bias_o = numpy.add(gradients, self.bias_o)                 

        hidden_errors = numpy.dot(numpy.transpose(self.weights_ho), output_errors)
        hidden_gradient = self.learning_rate * numpy.multiply(hidden_errors, numpy.multiply(hidden_output, numpy.subtract(1, hidden_output)))
        weights_ih_deltas = numpy.multiply(hidden_gradient, numpy.transpose(inputs))

        # new weight matrix input to hidden 
        self.weights_ih = numpy.add(weights_ih_deltas, self.weights_ih)  
        # new biases at hidden nodes
        self.bias_h = numpy.add(hidden_gradient, self.bias_h)           
