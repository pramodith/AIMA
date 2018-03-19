from __future__ import division

import numpy as np
from collections import Counter
import time
import random


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.

        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.

        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.

        Args:
            feature (list(int)): vector for feature.

        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if isinstance(self.class_label,int) :
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.

    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.

    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if class_index == -1:
        classes = map(int, out[:, class_index])
        features = out[:, :class_index]
        return features, classes

    elif class_index == 0:
        classes = map(int, out[:, class_index])
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the provided data.

    Tree is built fully starting from the root.

    Returns:
        The root node of the decision tree.
    """

    class_0=DecisionNode(None,None,None,0)
    class_1=DecisionNode(None,None,None,1)
    a4=DecisionNode(class_0,class_1,lambda x:x[3]==1)
    a5=DecisionNode(class_1,class_0,lambda x:x[3]==1)
    a3=DecisionNode(a5,a4,lambda x:x[2]==1)
    a1=DecisionNode(class_1,a3,lambda x:x[0]==1)


    return a1


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.

    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        A two dimensional array representing the confusion matrix.
    """

    true_positive=true_negative=false_positive=false_negative=0
    for i in range(0,len(classifier_output)):
        if classifier_output[i]==true_labels[i]:
            if classifier_output[i]==1:
                true_positive+=1
            else:
                true_negative+=1
        else:
            if classifier_output[i]==1:
                false_positive+=1
            else:
                false_negative+=1

    return [[true_positive, false_negative],[false_positive, true_negative]]

def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.

    Precision is measured as:
        true_positive/ (true_positive + false_positive)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The precision of the classifier output.
    """
    #print "class labels"+str(classifier_output)
    #print "actual output"+str(true_labels)
    true_positive=true_negative=false_positive=false_negative=0
    try:
        for i in range(0,len(classifier_output)):
            if classifier_output[i]==true_labels[i]:
                if classifier_output[i]==1:
                    true_positive+=1
                else:
                    true_negative+=1
            else:
                if classifier_output[i]==1:
                    false_positive+=1
                else:
                    false_negative+=1
        return float(true_positive)/ (true_positive + false_positive)
    except Exception as e:
        print classifier_output

def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.

    Recall is measured as:
        true_positive/ (true_positive + false_negative)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The recall of the classifier output.
    """
    true_positive=true_negative=false_positive=false_negative=0
    for i in range(0,len(classifier_output)):
        if classifier_output[i]==true_labels[i]:
            if classifier_output[i]==1:
                true_positive+=1
            else:
                true_negative+=1
        else:
            if classifier_output[i]==1:
                false_positive+=1
            else:
                false_negative+=1
    return  float(true_positive)/ (true_positive + false_negative)


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.

    Accuracy is measured as:
        correct_classifications / total_number_examples

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The accuracy of the classifier output.
    """
    true_positive=true_negative=false_positive=false_negative=0
    for i in range(0,len(classifier_output)):
        if classifier_output[i]==true_labels[i]:
            if classifier_output[i]==1:
                true_positive+=1
            else:
                true_negative+=1
        else:
            if classifier_output[i]==1:
                false_positive+=1
            else:
                false_negative+=1
    return float(true_negative+true_positive)/len(classifier_output)

def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.

    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.

    Returns:
        Floating point number representing the gini impurity.
    """
    positive=negative=0.0
    for value in class_vector:
        if value==1:
            positive+=1
        else:
            negative+=1
    return (1-(positive/len(class_vector))**2-(negative/len(class_vector))**2)

def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    den=0

    prev_impurity=gini_impurity(previous_classes)
    present_impurity=0.0
    for partition in current_classes:
        num=len(partition)
        positive=negative=0.0
        den+=num
        for val in partition:
            if val==1:
                positive+=1
            else:
                negative+=1
        if num==0:
            num=1
        present_impurity+=num*(1-(positive/num)**2-(negative/num)**2)
    if den==0:
        den=1
    return prev_impurity-(present_impurity/den)


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float("inf")):
        """Create a decision tree with a set depth limit.

        Starts with an empty root.

        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """
        #checking if we have reached a pure node ie a node with just one class in it
        if 0 in classes and 1 not in classes:
            return DecisionNode(None,None,None,0)
        elif 1 in classes and 0 not in classes:
            return DecisionNode(None,None,None,1)
        #checking if we have reached the depth limit then assign the node to the majority class
        elif depth==self.depth_limit:
            positive=0
            negative=0
            for x in classes:
                if x==1:
                    positive+=1
                else:
                    negative+=1
            if positive>negative:
                return DecisionNode(None,None,None,1)
            else:
                return DecisionNode(None,None,None,0)

        else:
            #otherwise for each feature store all of it's values and the class it corresponds to this is to easily calculate the right split point
            index_map=[]
            try:
                index_map=[[] for x in range(len(features[0]))]
                for i in range(len(features[0])):
                    for j in range(len(features)):
                        index_map[i].append([features[j][i],classes[j]])
            except Exception as e:
                pass
            #sorting the feature,class pair based on the value of the feature
            for i in range(len(index_map)):
                index_map[i].sort(key=lambda x:x[0])


            previous_classes=classes[:]
            split_point=0
            split_index=0
            best_gain=float("-inf")
            present_gain=None
            #store the class corresponding to each index in the index_map
            fixed_class_order=[[] for x in range(len(index_map))]
            for cnt,attr in enumerate(index_map):
                for i in range(len(attr)):
                    try:
                        fixed_class_order[cnt].append(index_map[cnt][i][1])
                    except Exception as e:
                        pass

            class_left=[]
            class_right=[]
            best_split=(0,index_map[0][0])

            #find the best split point
            for cnt,attr in enumerate(index_map):
                #for each attribute of a feature
                for i in range(len(attr)-1):
                    #ensuring that both the split points are not the same
                    if attr[i][1]!=attr[i+1][1] and attr[i][0]!=attr[i+1][0]:
                        split_point=(attr[i][0]+attr[i+1][0])/2
                        present_gain=gini_gain(previous_classes,[fixed_class_order[cnt][0:i+1],fixed_class_order[cnt][i+1:]])
                        if present_gain>=best_gain:
                            best_gain=present_gain
                            best_split=(cnt,split_point)


            node=DecisionNode(None,None,lambda x:x[best_split[0]]<best_split[1])
            features_left=[]
            features_right=[]
            best_class_left=[]
            best_class_right=[]
            #identifying all the left and right nodes after identifying the split point
            for x in range(len(features)):
                if features[x][best_split[0]]<best_split[1]:
                    features_left.append(features[x])
                    best_class_left.append(classes[x])
                if features[x][best_split[0]]>=best_split[1]:
                    features_right.append(features[x])
                    best_class_right.append(classes[x])

            #if one of the left or right splits is empty we have reached a leaf node.
            if len(features_left)==0 or len(features_right)==0:
                if len(features_left)==0:
                    if best_class_right.count(0)>best_class_right.count(1):
                        return DecisionNode(None,None,None,0)
                    elif best_class_right.count(0)<best_class_right.count(1):
                        return DecisionNode(None,None,None,1)
                    else:
                        return DecisionNode(None,None,None,random.randint(0,1))
                if len(features_right)==0:
                    if best_class_left.count(0)>best_class_left.count(1):
                        return DecisionNode(None,None,None,0)
                    elif best_class_left.count(0)<best_class_left.count(1):
                        return DecisionNode(None,None,None,1)
                    else:
                        return DecisionNode(None,None,None,random.randint(0,1))
            else:
                node.left=self.__build_tree__(features_left,best_class_left,depth+1)
                node.right=self.__build_tree__(features_right,best_class_right,depth+1)
            return node



    def classify(self, features):
        """Use the fitted tree to classify a list of example features.

        Args:
            features (list(list(int)): List of features.

        Return:
            A list of class labels.
        """
        class_labels=[]
        for cnt,feature in enumerate(features):
            class_labels.append(self.root.decide(feature))
        return class_labels

#cross validation
def generate_k_folds(dataset, k):
    """Split dataset into folds.

    Randomly split data into k equal subsets.

    Fold is a tuple (training_set, test_set).
    Set is a tuple (examples, classes).

    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.

    Returns:
        List of folds.
    """
    tree=DecisionTree()
    total_accuracy=0
    sets=[]
    cnt=0
    dupl_dataset=[]
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if i==0:
                dupl_dataset.append([])
                for data in dataset[i][j]:
                    dupl_dataset[j].append(data)
            else:
                dupl_dataset[j].append(dataset[i][j])
        cnt+=1
    random.shuffle(dupl_dataset)
    for i in range(0,k):
        sets.append(dupl_dataset[int(i*len(dupl_dataset)/k):int((i+1)*len(dupl_dataset)/k)])
    attributes=[]
    classes=[]
    cnt=0
    for s in sets:
        for i in range(len(s)):
            attributes.append([])
            for j in range(len(s[i])):
                if j<len(s[i])-1:
                    attributes[cnt].append(s[i][j])
                else :
                    classes.append(s[i][j])
            cnt+=1
    folds=[]
    for i in range(0,k):
        features=attributes[0:i*int(len(dupl_dataset)/k)]+attributes[(i+1)*int(len(dupl_dataset)/k):]
        tree.fit(features,classes[0:i*int(len(dupl_dataset)/k)]+classes[(i+1)*int(len(dupl_dataset)/k):])
        train=(features,classes[0:i*int(len(dupl_dataset)/k)]+classes[(i+1)*int(len(dupl_dataset)/k):])
        test=(attributes[i*int(len(dupl_dataset)/k):(i+1)*int(len(dupl_dataset)/k)],classes[i*int(len(dupl_dataset)/k):(i+1)*int(len(dupl_dataset)/k)])
        folds.append((train,test))
    return folds

class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.

         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.

            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """
        number_of_samples=int(self.example_subsample_rate*len(features))
        number_of_attrs=int(self.attr_subsample_rate*len(features[0]))
        attribute_index=[i for i in range(0,len(features[0]))]
        for times in range(0,self.num_trees):
            dt=[]
            cl=[]
            for i in range(0,number_of_samples):
                index=random.randint(0,len(features)-1)
                x=features[index].tolist()
                dt.append(x)
                cl.append(classes[index])
            random.shuffle(attribute_index)
            for row in dt:
                del_el=[]
                for ind in range(0,len(features[0])-number_of_attrs):
                    try:
                        del_el.append(row[attribute_index[ind]])
                    except Exception as e:
                        pass
                for l in range(0,len(del_el)):
                    row.remove(del_el[l])
            self.trees.append(DecisionTree(self.depth_limit))
            self.trees[times].fit(dt,cl)

    def classify(self, features):
        """Classify a list of features based on the trained random forest.

        Args:
            features (list(list(int)): List of features.
        """
        class_labels=[]
        pred_cnt=[{0:0,1:0} for x in range(0,len(features))]
        #creating a voting mechansim where the vote of each tree in the random forest is stored and the majority vote wins.
        for tree in self.trees:
            if tree.root:
                class_labels=[]
                for cnt,feature in enumerate(features):
                    class_labels.append(tree.root.decide(feature))
                for ind,label in enumerate(class_labels):
                    try:
                        pred_cnt[ind][label]+=1
                    except Exception as e:
                        pass
        for i in range(0,len(pred_cnt)):
            if pred_cnt[i][0]>pred_cnt[i][1]:
                class_labels[i]=0
            else:
                class_labels[i]=1
        return class_labels

