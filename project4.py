"""
Tom Parker
PLEASE DOCUMENT HERE

Usage: python3 project4.py DATASET.csv
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
        for example in training:  #***** 
            inflow = example[0]
        #inflow = training[0][0]

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

        return output
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
        pass
        # RELEVANCE OF REMOVING DUMMY VAL IN LAST LAYER OCCURS HERE



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
    # nn.back_propagation_learning(training)

if __name__ == "__main__":
    main()
