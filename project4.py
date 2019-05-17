"""
Tom Parker
Usage: python3 project4.py DATASET.csv

Contains a neural network implementation with forward propagation for prediction and
back propagation for learning weights and repeatedly makes neural networks learning how to classify
based on the given data.

Pass the program a .csv file as the dataset.
This dataset must be formatted such that the first line contains information such that "target"
refers to the output layer and anything else refers to the input layer. The form is:
    x1,x2,...,xn,target1,target2...,targetm. The x can be something else or nothing. The number
    label is also unnecessary.
The following lines contain examples of inputs with their correct outputs. For example,
a 3-bit incrementer, the second line is, "0,0,0,0,0,1". The input is of size 3 and so is
the output, so the total length is 6. If the input is "0,0,0", then the output is "0,0,1".

The program takes this dataset and learns how to classify as best it can based on the input,
ultimately producing a neural network object that as best as it can classifies information of the
given format. 
"""

import csv, sys, random, math

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
            true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here

class NeuralNetwork():
    """ A neural network """

    def __init__(self, layers):
        """ Initializes the neural network graph such that it is the right shape and contains
            random weights """

        self.layers = layers
        self.graph = [] # List of 2d lists, where each 2d list is the matrix corresponding to the
                        # synapses (edges) between a given layer. So equivalent to [[[]][[]]]
                        # So like each list has w i,j
        for i in range(len(layers) - 1): # For each synapse layer (area between two layers)
            self.graph.append([]) # Add synapse layer (matrix)

            for row in range(layers[i] + 1): # for each sending neuron plus the dummy value
                self.graph[i].append([]) # Add row to matrix

                for col in range(layers[i + 1]): # for each receiving neuron
                    self.graph[i][row].append([]) # Add column to matrix
                    self.graph[i][row][col] = random.uniform(0, 2) # Add random value
                

                # print("pre: ", self.graph[i])
                # self.graph[i][row].insert(0, []) # Add spot for dummy value
                # self.graph[i][row][0] = random.uniform(0, 2)
                # print("post: ", self.graph[i])

    def __str__(self):
        """ Representation of the neural network for printing """
        output = ""
        for i, layer in enumerate(self.graph):
            output += ("\n\nSynapse Matrix {}:".format(i))
            for r, row in enumerate(layer):
                output += ("\nNeuron {}: ".format(r))
                for col in row:
                    output += ('\n          {}'.format(col))
        return output

    ############################################################################
    #                                                                          #
    #  REMEMBER THAT LAST LAYER NEEDS TO HAVE DUMMY VAL REMOVED WHEN RELEVANT  #
    #  Also should be noted that I may have accounted for dummy values twice   #
    #                                                                          #
    ############################################################################



    def forward_propagate(self, training):
        """ Propagate the inputs forward to compute the outputs """
        # for i, neuron in enumerate(self.graph[0]):
        # so each step: for each node in second layer, add together wi * xi for each input 
        #for r, row in enumerate(self.graph[0]):
            # print(row)
            # print(training[r][0])
            # result = dot_product(row, training[r][0])
            # print(result)
        
        DUMMY_VALUE = 1.0

        output = []
        for example in training: # For each training set
            inflow = example[0] # Start
            #print("inf: ", inflow)

            for i, synapse in enumerate(self.graph): # For each synapse
                results = []
                #print("i: ", i)
                for c in range(self.layers[i+1]): # For each in-neuron
                    #print("c: ", c)
                    result = 0
                    #print("Ex: ", example[0])
                    for r, row in enumerate(self.graph[i]):   # For each out-neuron to that in-neuron
                        # print("r: ", r)
                        # print(row[c])
                        # print(inflow)
                        # print(inflow[r])
                        #print()
                        result += row[c] * inflow[r]   # Multiply the weight and the input value
                    if i > 0: # If not the first layer, apply the sigmoid function
                        result = logistic(result)
                    results.append(result)
                    #print("res: ", results)
                inflow = [DUMMY_VALUE] + results
                #print("inf: ", inflow)

            #print()
            #print("Yeet: ", inflow[1:])
            #print()
            output.append(inflow[1:]) # Cut off the dummy value if on the output layer
            # print(output)
            # quit()

        return output # A list of lists, where each sublist is of size layers[-1], containing each
                      # of the output values

        #return inflow[1:] # Return output values without the dummy value
            #print(results)

            # results = []
            # for c in range(self.layers[1]): # For each in-neuron
            #     result = 0
            #     #print("Ex: ", example[0])
            #     for r, row in enumerate(self.graph[0]):   # For each out-neuron to that in-neuron
            #         result += row[c] * example[0][r]   # Add the xi * ai
            #         print(row[c])
            #         print(example[0][r])
            #         print()
            #     results.append(result)
            # #print(results)

            # for synapses in self.graph[1:]: # Starting from the second set of synapses



    # def predict_class():
    #     pass

    def back_propagation_learning(self, training):
        # RELEVANCE OF REMOVING DUMMY VAL IN LAST LAYER OCCURS HERE
        # for synapse in self.graph:
        #     for row in synapse:
        #         for col in row:
        #             for i in range(5): # Make 1000 later
        forward_result = self.forward_propagate(training)

        # Propagate deltas backward from output layer to input layer





def main():
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs] # so this is a list of tuples of lists

    # Check out the data:
    for example in training:
        print(example)

    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    nn = NeuralNetwork([3, 6, 3])
    print(nn)
    forward = nn.forward_propagate(training)
    print()
    print(forward)
    nn.back_propagation_learning(training)

if __name__ == "__main__":
    main()
