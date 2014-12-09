import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import math,operator


class CarFeaturizer():
    def create_features(self, data, training=False):

    # Combination of the following calculations and column drops gave the best result
        data.loc[:,'MPY'] = data.VehOdo/ (data.VehYear - data.VehYear.min())
        data.loc[:,'priceDiff1'] = data.MMRAcquisitionRetailAvera - data.MMRAcquisitionAuctionAver
        data.loc[:,'priceDiff3'] = data.MMRCurrentAuctionCleanPri - data.MMRAcquisitionAuctionClea


        data = data.drop('SubModel', axis=1)
        data = data.drop('Trim', axis=1)
        data = data.drop('Transmission', axis=1)
        data = data.drop('PurchDate', axis=1)
        data = data.drop('Color', axis=1)

        #use the size category
        sizeType = pd.get_dummies(data.loc[:,'Size'])
        data = pd.concat([data, sizeType], axis=1)
        data = data.drop('Size', axis=1)

        data = data.drop('Auction', axis=1)

        TopThreeType = pd.get_dummies(data.loc[:,'TopThreeAmericanName'])
        data = pd.concat([data, TopThreeType], axis=1)
        data = data.drop('TopThreeAmericanName', axis=1)

        NationalityType = pd.get_dummies(data.loc[:,'Nationality'])
        data = pd.concat([data, NationalityType], axis=1)
        data = data.drop('Nationality', axis=1)

        WheelType = pd.get_dummies(data.loc[:,'WheelType'])
        data = pd.concat([data, WheelType], axis=1)
        data = data.drop('WheelType', axis=1)

        top_make = set( data.Make.value_counts().index[:10])
        data.loc[:,'Newmake'] = data.Make.map(lambda make: make if make in top_make else 0.0)
        makeType = pd.get_dummies(data.loc[:,'Newmake'])
        data = pd.concat([data, makeType], axis=1)
        data = data.drop('Make', axis=1)

        # use top ten Models
        top_models = set (data.Model.value_counts().index[:50])
        data.loc[:,'NewModel'] = data.Model.map(lambda model: model if model in top_models else 0.0)
        modelType = pd.get_dummies(data.loc[:,'NewModel'])
        data = pd.concat([data, modelType], axis=1)
        data = data.drop('Model', axis=1)

        return data

def train_model(data):
    data_values = data.values.tolist()
    model = create_decision_tree(data_values, trace=0)
    return model

def create_submission(model, transformer):
  submission_test = pd.read_csv('test_equal_interval.csv')

  get_refid = pd.read_csv('test.csv')
  predictions = pd.Series(model.predict(transformer.create_features(submission_test)))

  print predictions.value_counts()
  submission = pd.DataFrame({'RefId': get_refid.RefId, 'IsBadBuy': predictions})
  cols = submission.columns.tolist()
  cols = cols[-1:] + cols[:-1]

  submission = submission[cols]

  submission.to_csv('submission_midterm.csv', index=False)

def entropy(instances, class_index=0, attribute_name=None, value_name=None):
    '''Calculate the entropy of attribute in position attribute_index for the list of instances.'''
    num_instances = len(instances)
    if num_instances <= 1:
        return 0
    value_counts = defaultdict(int)
    for instance in instances:
        value_counts[instance[class_index]] += 1
    num_values = len(value_counts)
    if num_values <= 1:
        return 0
    attribute_entropy = 0.0
    n = float(num_instances)

    for value in value_counts:
        value_probability = value_counts[value] / n
        child_entropy = value_probability * math.log(value_probability, num_values)
        attribute_entropy -= child_entropy

    return attribute_entropy


def information_gain(instances, parent_index, class_index=0, attribute_name=False):
    '''Return the information gain of splitting the instances based on the attribute parent_index'''
    parent_entropy = entropy(instances, class_index, attribute_name)
    child_instances = defaultdict(list)
    for instance in instances:
        child_instances[instance[parent_index]].append(instance)
    children_entropy = 0.0
    n = float(len(instances))
    for child_value in child_instances:
        child_probability = len(child_instances[child_value]) / n
        children_entropy += child_probability * entropy(child_instances[child_value], class_index, attribute_name, child_value)
    return parent_entropy - children_entropy

def split_instances(instances, attribute_index):
    partitions = defaultdict(list)
    for instance in instances:
        partitions[instance[attribute_index]].append(instance)
    return partitions

def majority_value(instances, class_index=0):
    '''Return the most frequent value of class_index in instances'''
    class_counts = Counter([instance[class_index] for instance in instances])
    return class_counts.most_common(1)[0][0]


def choose_best_attribute_index(instances, candidate_attribute_indexes, class_index=0):
    '''Return the index of the attribute that will provide the greatest information gain
    if instances were partitioned based on that attribute'''
    gains_and_indexes = sorted([(information_gain(instances, i), i) for i in candidate_attribute_indexes],
                               reverse=True)
    return gains_and_indexes[0][1]

def create_decision_tree(instances, candidate_attribute_indexes=None, default_class=0, trace=0):
    '''Returns a new decision tree trained on a list of instances.

    The tree is constructed by recursively selecting and splitting instances based on
    the highest information_gain of the candidate_attribute_indexes.

    The class label is found in position class_index.

    The default_class is the majority value for the current node's parent in the tree.
    A positive (int) trace value will generate trace information with increasing levels of indentation.
    '''

    # if no candidate_attribute_indexes are provided, assume that we will use all but the target_attribute_index

    # Class index here is IsBadBuy

    if candidate_attribute_indexes is None:
        candidate_attribute_indexes = range(len(instances[0]))
        candidate_attribute_indexes.remove(0)


    class_labels_and_counts = Counter([instance[0] for instance in instances])

    # If the dataset is empty or the candidate attributes list is empty, return the default value
    if not instances or not candidate_attribute_indexes:
        if trace:
            print '{}Using default class {}'.format('< ' * trace, default_class)
        return default_class

    # If all the instances have the same class label, return that class label
    if len(class_labels_and_counts) == 1:
        class_label = class_labels_and_counts.most_common(1)[0][0]
        if trace:
            print '{}All {} instances have label {}'.format('< ' * trace, len(instances), class_label)
        return class_label
    else:
        default_class = majority_value(instances)
        best_index = choose_best_attribute_index(instances, candidate_attribute_indexes, 0)

        if trace:
            print '{}Creating tree node for attribute index {}'.format('> ' * trace, best_index)

        # Create a new decision tree node with the best attribute index and an empty dictionary object (for now)
        tree = {best_index:{}}

        # Create a new decision tree sub-node (branch) for each of the values in the best attribute field
        partitions = split_instances(instances, best_index)

        # Remove that attribute from the set of candidates for further splits
        remaining_candidate_attribute_indexes = [i for i in candidate_attribute_indexes if i != best_index]

        for attribute_value in partitions:
            if trace:
                print '{}Creating subtree for value {} ({}, {}, {}, {})'.format(
                    '> ' * trace,
                    attribute_value,
                    len(partitions[attribute_value]),
                    len(remaining_candidate_attribute_indexes),
                    0,
                    default_class)

            # Create a subtree for each value of the the best attribute
            subtree = create_decision_tree(
                partitions[attribute_value],
                remaining_candidate_attribute_indexes,
                default_class,
                (trace + 1 if trace else 0))

            # Add the new subtree to the empty dictionary object in the new tree/node we just created
            tree[best_index][attribute_value] = subtree

    return tree

def classify(tree, instance, default_class=0):
    '''Returns a classification label for instance, given a decision tree'''
    if not tree:
        return default_class
    if not isinstance(tree, dict):
        return tree
    attribute_values = tree.values()[0]
    attribute_index = tree.keys()[0]

    instance_attribute_value = instance[attribute_index-1]

    if instance_attribute_value not in attribute_values:
        return default_class
    return classify(attribute_values[instance_attribute_value], instance, default_class)

def main():
    data = pd.read_csv('training_equal_freq.csv')
    featurizer = CarFeaturizer()

    print "Transforming dataset into features..."
    X = featurizer.create_features(data, training=True)

    print "Training model..."
    print len(data.columns)

    model2 = train_model(X)

    submission_test = pd.read_csv('test_equal_freq.csv')

    get_refid = pd.read_csv('test.csv')
    A = featurizer.create_features(submission_test)

    print "Create predictions on submission set..."
    test_values = A.values.tolist()
    prediction = [0]*len(test_values)
    i = 0
    for row in test_values:
        prediction[i] = classify(model2, row)
        i += 1

    predictions = pd.Series(prediction)
    print predictions.value_counts()

    submission = pd.DataFrame({'RefId': get_refid.RefId, 'IsBadBuy': predictions})
    cols = submission.columns.tolist()
    cols = cols[-1:] + cols[:-1]

    submission = submission[cols]

    submission.to_csv('submission_final.csv', index=False)

if __name__ == '__main__':
    main()
