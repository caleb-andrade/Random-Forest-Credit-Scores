# -*- coding: utf-8 -*-
"""
Created on Sat Dec 05 10:33:36 2015

@author: Caleb Andrade
"""

import random
import math


def main():
    
    # read data
    data = readFile('crx.data.txt')
    
    # create forest
    forest = Forest(data, 15)
    
    # trees accuracies
    forest.treeAccuracy()
    
    # test accuracy
    forest.forestAccuracy()
    
    # query an instance
    x = ['b', 56.75, 12.25, 'u', 'g', 'm', 'v', 1.25, 't', 't', 4, 't', 'g', '?', '?', '?']
    
    y = convertInstance(x, forest.getThreshold())

    print "\nQuery instance: ", forest.forestQuery(y)

              
# map of the possible attribute values, for numerical values: [True, False]
ATT_VALUES = {0:['a', 'b'],
              1:[True, False],
              2:[True, False],
              3:['u', 'y', 'l', 't'],
              4:['g', 'p', 'gg'],
              5:['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff'],
              6:['v','h','bb','j','n','z','dd','ff','o'],
              7:[True, False],
              8:['t','f'],
              9:['t','f'],
              10:[True, False],
              11:['t','f'],
              12:['g','p','s'],
              13:[True, False],
              14:[True, False],
              15:['+', '-']}
              
#*****************************************************************************
# HELPER FUNCTIONS  
#*****************************************************************************
def readFile(filename):
    """
    Read file to construct a list of instances (data set)
    """
    data = []
    with open(filename) as f: 
        while True:
            line = f.readline()
            if line == '':
                break
            instance = []
            attribute = ''
            
            # create a list of attributes for current line read
            for c in line:
                if c == ',':
                    instance.append(attribute)
                    attribute = ''
                    continue
                if c == '+' or c == '-':
                    instance.append(c)
                    attribute = ''
                    break
                attribute += c
            
            # parsing numbers to correct format
            for idx in [1, 2, 7]:
                if instance[idx] != '?':
                    instance[idx] = float(instance[idx])
            for idx in [10, 13, 14]:
                if instance[idx] != '?':
                    instance[idx] = int(instance[idx])
            
            data.append(instance)
            
    return data


def convertInstance(instance, threshold):
    """
    Convert numerical values of an instance to boolean.
    
    Input: an instance, a map of threshold values computed for the
    particular numerical attributes of the training data set.
    Output: an instance with numerical values changed to boolean.
    """
    instance_copy = list(instance)
    for num_att in [1, 2, 7, 10, 13, 14]:
        if instance_copy[num_att] == '?':
            instance_copy[num_att] = random.choice([True, False])
        elif instance_copy[num_att] < threshold[num_att]:
            instance_copy[num_att] = True
        else:
            instance_copy[num_att] = False
            
    return instance_copy
            
        
def copy(subset):
    """
    Make a deep copy of a subset of training set.
    """
    copy_subset = []
    for instance in subset:
        copy_subset.append(list(instance))
        
    return copy_subset


def signCounter(subset, attribute, value):
    """
    Count the number of '+'/'-' of those instances of a subset that, 
    for a specific attribute, have a given value.
    
    Input: training data subset, attribute index, attribute value.
    Output: number of '+', number of '-'
    """
    sign_count = {'+':0, '-':0}
    for instance in subset:
        if instance[attribute] == value:
            sign_count[instance[-1]] += 1
    
    return sign_count['+'], sign_count['-']
        

def priorDist(subset):
    """
    Compute the prior empirical distribution of classification
    in a subset of the training data.
    
    Input: training data subset.
    Output: P('+'), P('-')
    """
    plus = 0
    for instance in subset:
        if instance[-1] == '+':
            plus += 1
            
    plus_fraction = float(plus) / len(subset)
    
    return plus_fraction, (1 - plus_fraction)
    

def candidThreshold(subset, num_att):
    """
    Compute a list of candidate thresholds when an attribute is 
    numeric.
    
    Input: data set, index of numerical attribute.
    """
    candidates = []
    temp = list(subset)
    # sort list of instances according to numeric attribute
    temp.sort(key = lambda instance:instance[num_att])
    # initialize first sign and index
    sign = temp[0][-1]
    # loop all instances, except for the first
    for idx in range(1, len(temp)):
        # avoid missing values
        if temp[idx][num_att] != '?':
            # if there is a switch in the sign, take the average of values
            if temp[idx][-1] != sign:
                average = (temp[idx][num_att] + temp[idx-1][num_att]) / 2
                candidates.append(average)
                # update sign and index
                sign = temp[idx][-1]

    return candidates                


def entropy(probabilities):
    """
    Compute the entropy over a set of probabilities.
    """
    entropy_sum = 0
    for p in probabilities:
        if p != 0:
            entropy_sum += -p*math.log(p, 2)
    
    return entropy_sum
    

def condEntropy(subset, attribute, value):
    """
    Compute entropy of Y = '+' conditioned to X = value,
    where X represents an attribute, in a subset of training set.
    
    Input: training set, attribute's index, attribute's value.
    Output: conditional entropy as a float number.
    """
    # count the number of '+' and '-'
    plus, minus = signCounter(subset, attribute, value)
    total = plus + minus
    
    return entropy([float(plus) / total, float(minus) / total])


def attValues(subset, attribute):
    """
    Compute the set of values that an attribute take in a subset
    of a training data set.
    """
    values = set([])
    for instance in subset:
        values.add(instance[attribute])
        
    return values
    
    
def expGain(subset, attribute):
    """
    Compute expected gain conditioning on a given attribute.
    """
    # initialize with entropy of prior distribution
    gain = entropy(priorDist(subset))
    # loop values of attribute in subset
    for value in attValues(subset, attribute):
        weigth = sum(signCounter(subset, attribute, value))
        # substract P(value)*entropy(Y|X = value)
        gain -= (float(weigth)/len(subset))*condEntropy(subset, attribute, value)
    
    return gain    
    

def missingValue(subset, attribute, instance):
    """
    Estimate a missing value for an instance's attribute in a subset
    of the training data, using the values' empirical distribution.
    """
    values = []
    # loop all instances (items)
    for item in subset:
        # do not take into account other missing values
        if item[attribute] != '?':
            values.append(item[attribute])
    # sample value uniformly from values
    return random.sample(values, 1)[0]
    
    
def completeValues(subset):
    """
    Sweep data once and if there is any missing value, complete
    data using missingValue method.
    """
    for instance in subset:
        for idx in range(len(instance)):
            if instance[idx] == '?':
                instance[idx] = missingValue(subset, idx, instance)
    
                
def setThreshold(subset, num_att):
    """
    Set threshold for a numeric attribute in subset.
    """
    best_gain = 0
    # make a copy of subset
    copy_subset = copy(subset)
    # get candidate thresholds
    candidates = candidThreshold(subset, num_att)
    best = float('inf')
    # loop threshold candidates
    for candidate in candidates:
        # set boolean values for num_att depending on candidate
        for idx in range(len(subset)):
            if subset[idx][num_att] < candidate:
                copy_subset[idx][num_att] = True
            else:
                copy_subset[idx][num_att] = False
        # compute expected gain for such candidate
        gain = expGain(copy_subset, num_att)
        # pick best candidate so far
        if gain > best_gain:
            best_gain = gain
            best = candidate
    
    return best
    

def numToBool(subset, num_att):
    """
    Convert numerical attribute to boolean attribute.
    Note: Mutates subset, numerical values to boolean values.
    
    Input: subset of data, index of numerical attribute.
    Output: threshold for this particular numerical attribute and subset.
    """
    threshold = setThreshold(subset, num_att)
    
    for instance in subset:
        # ignore missing values
        if instance[num_att] == '?':
            continue
        if instance[num_att] < threshold:
            instance[num_att] = True
        else:
            instance[num_att] = False
    
    return threshold


def testSign(subset):
    """
    Verifies classification of all instances in subset.
    """
    sign = subset[0][-1]
    for instance in subset[1:]:
        if instance[-1] != sign:
            return '+/-'
    
    return sign
    

def getInstances(subset, attribute, value):
    """
    Return instances in subset that have 'value' for attribute.
    """
    instances = []
    for instance in subset:
        if instance[attribute] == value:
            instances.append(instance)
    
    return instances
    

def mostCommon(subset):
    """
    Compute the most common classification in subset.
    """
    sign_count = {'+':0, '-':0}
    for instance in subset:
        sign_count[instance[-1]] += 1
    
    if sign_count['+'] > sign_count['-']:
        return '+'
    else:
        return '-'

#*****************************************************************************
# NODE CLASS OBJECT, borrowed from tree_searcher.py, Homework 2  
#*****************************************************************************
class Node:
    """
    Representation of a generic decision tree.
    Each node holds
    1. An attribute (internal nodes)
    2. A sign (leaf nodes)
    3. A parent's branch value (except the root)
    4. A node type (internal / leaf)
    5. list of child nodes.
    """
    def __init__(self, attribute, sign, value, node_type, children=[]):
        self.attribute = attribute
        self.sign = sign
        self.value = value
        self.node_type = node_type
        self.children = children
    
    def setAttribute(self, attribute):
        """
        Set an attribute for this three node.
        """
        self.attribute = attribute
        
    def setSign(self, sign):
        """
        Set classification sign for this three node.
        """
        self.sign = sign        
           
    def setValue(self, value):
        """
        Set value for this three node.
        """
        self.value = value
        
    def setNodeType(self, node_type):
        """
        Set a node_type for this three node.
        """
        self.node_type = node_type
  	
    def setChildren(self, child_nodes):
        """
        Set the children of this tree node.
        """
	if not self.children:
	    self.children = []
	for child in child_nodes:
	    self.children.append(child)
     
    def addChildren(self, child):
        """
        Add children to this node.
        """
	if not self.children:
	    self.children = []	    
	self.children.append(child)
        
    def getAttribute(self):
        return self.attribute
        
    def getSign(self):
        return self.sign
        
    def getValue(self):
        return self.value
        
    def getNodeType(self):
        return self.node_type    
    
    def getChildren(self):
        return self.children
    
    def toString(self, depth):
        """
        As string.
        """
        string = "\n" + "     "*depth + "Depth: " + str(depth)
        if self.value != None:
            string += "\n" + "     "*depth + "Branch value: " + str(self.value)
        if self.attribute != None:
            string += "\n" + "     "*depth + "Attribute   : " + str(self.attribute)
        string += "\n" + "     "*depth + "Node Type   : " + str(self.node_type)
        if self.sign != None:
            string += "\n" + "     "*depth + "Sign        : " + str(self.sign)
        
        return string
   

def printTree(node, depth = 0):
    """
    Tree as string.
    """
    print node.toString(depth)
    for child in node.getChildren():
        printTree(child, depth + 1)
        

def query(node, instance):
    """
    Recursively query for an instance classification in a decision tree.
    """
    if node.getNodeType() == 'leaf':
        return node.getSign()
    else:
        value = instance[node.getAttribute()]
    for child in node.getChildren():
        child_value = child.getValue()
        if child_value == value:
            return query(child, instance)
        elif type(child_value) != type(value):
            raise Exception ("Different type of values")
    return None
    

def testAccuracy(decision_tree, training_set):
    """
    Tests accuracy of a decision tree on a training set.
    """
    hits = 0
    for instance in training_set:
        output = query(decision_tree, instance)
        if output == instance[-1]:
            hits += 1
    return round(100*float(hits)/len(training_set), 1)

#*****************************************************************************
# DECISION TREE: ID3 ALGORITHM  
#*****************************************************************************
def decisionTree(subset, attributes, att_values, branch_value = None, best_split = True):
    """
    ID3 algorithm implementation. Returns a tree (Node object).
    
    Input: training data set, list of attributes indices, dictionary
    of attributes values, if node is not the root branch_value is the
    value of the edge that connects it with its parent. Best_split is a boolean
    if on, will select best split attribute, otherwise will select it randomly.
    Output: a decision tree. To perform queries use the function 'query'.
    """
    # initialize root and check "classification purity" of subset
    root = Node(None, None, branch_value, None)
    sign = testSign(subset)
    
    # test if all instances have same classification
    if sign != '+/-':
        root.setSign(sign)
        root.setNodeType('leaf')
        return root
    
    if len(attributes) == 0:
        # return the most common classification sign in subset
        sign = mostCommon(subset)
        root.setSign(sign)
        root.setNodeType('leaf')
        return root
    
    if best_split:
        # find the attribute that produces the best split, highest info gain.
        best_gain = -1
        best = None
        copy_subset = copy(subset)
        for att in attributes:
            gain = expGain(copy_subset, att)
            # pick best candidate so far
            if gain > best_gain:
                best_gain = gain
                best = att
    else:
        # select the attribute to split randomly
        best = random.choice(attributes)
    
    # set the decision attribute for root
    root.setAttribute(best)   
    root.setNodeType('internal')
    for value in att_values[best]:
        subset_value = getInstances(subset, best, value)
        if subset_value == []:
            sign = mostCommon(subset_value)
            root.addChildren(Node(None, sign, value, 'leaf'))
        else:
            child_att = list(attributes)
            child_att.remove(best)
            child = decisionTree(subset_value, child_att, att_values, value) 
            root.addChildren(child)
            
    return root


#*****************************************************************************
# DECISION TREE: ID3 ALGORITHM  
#*****************************************************************************
class Forest(object):
    """
    Create a number of decision trees, specified by the user,
    using the ID3 algorithm to query instance classification 
    predictions. Partition the training set as follows:
        
        * 2/3 to train the forest.
        * 1/3 for validation.
    
    Out of those two thirds, for each decision tree, the data
    is splitted likewise: 2/3 to train the decision tree, another
    1/3 for validation. The data is sampled randomly for every
    tree. When a query is performed in the forest, the output
    is the majority vote of all the decision trees created.
    """
    def __init__(self, data_set, num_trees):
        """
        Input: set of pre-processed data, that is, no numerical
        values nor missing values; number of trees in the forest.
        """
        data = copy(data_set) # make a copy of data set
#        random.shuffle(data) # shuffle data
        self.threshold = {} # keep track of thresholds for numerical attributes
        
        # convert numerical values to boolean
        for num_att in [1, 2, 7, 10, 13, 14]: # loop indices of numerical attributes
            self.threshold[num_att] = numToBool(data, num_att)
    
        # If there are missing values, fill them!
        completeValues(data)    
    
        cut = int( 2*float( len(data) ) / 3 ) # index at 2/3 of data
        
        self.training = copy(data[:cut])
        self.validation = copy(data[cut:])
        self.num_trees = num_trees
        self.decision_trees = []
        self.tree_accuracy = []
        
        subcut = int( 2*float( len(self.training) ) / 3 )
        
        # initialize decision trees
        for idx in range(num_trees):
            # sample randomly 2/3 of training data for each tree
            temp = copy(self.training)
            random.shuffle(temp)
            # create decision tree
#            tree = decisionTree(temp[:subcut], range(15), ATT_VALUES)
            tree = decisionTree(temp[:subcut], range(15), ATT_VALUES, None, False)
            self.decision_trees.append(tree)
            # test individual tree accuracy
            accuracy = testAccuracy(tree, temp[subcut:])
            self.tree_accuracy.append(accuracy)
            
    def forestQuery(self, instance):
        """
        Query an instance on each of the trees of forest.
        Return the majority vote classification.
        """
        sign = {'+':0, '-':0}
        idx = 1
        for tree in self.decision_trees:
            output = query(tree, instance)
            sign[output] += 1
            idx += 1
 
        # consensus
        if sign['+'] > sign['-']:
            return '+'
        else:
            return '-'
            
    def forestAccuracy(self):
        """
        Tests forrest accuracy on the initial validation set.
        """
        hits = 0
        for instance in self.validation:
            output = self.forestQuery(instance)
            if output == instance[-1]:
                hits += 1
        
        accuracy = round(100*float(hits)/len(self.validation), 1)
        print "\nForest accuracy: ", accuracy, "%"
        
        return accuracy
        
    def getThreshold(self):
        """
        Return list of thresholds.
        """
        return self.threshold
        
    def treeAccuracy(self):
        """
        Prints the individual accuracies of each tree, on its own validation
        set.
        """
        idx = 1
        for accuracy in self.tree_accuracy:
            print idx, ") Tree accuracy: ", accuracy, "%"
            idx += 1
            
        
if __name__ == '__main__':
    main()
