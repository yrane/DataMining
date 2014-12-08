from collections import defaultdict
import sys; import os
import random, math
from numpy import cumsum
from numpy.random import rand
import numpy as np

def ensemble_predict(X, k, all_attributes, all_class_prob, error_class):
    false_positive = 0
    false_negative = 0
    true_negative = 0
    true_positive = 0
    attr = []
    vals = []

    for x in X:
        y = x.split()

        row_values = [j for j in x.split(" ")]
        for val in range(1, len(row_values)):
            split_vals = row_values[val].split(":")
            attr.append(int(split_vals[0]))
            vals.append(int(split_vals[1]))
        length = len(vals)

        weights_class = [0.0] * 2

        for i in range(k):
            w = math.log((1 - error_class[i])/error_class[i])

            probability = predict(X,attr, vals, length, all_class_prob[i], all_attributes[i])

            if probability > 0.5:
                weights_class[0] += w
            else:
                weights_class[1] += w

        if weights_class[0] > weights_class[1]:
            probability = 1.0
        elif weights_class[0] < weights_class[1]:
            probability = 0.0
        else:
            probability = predict(X,attr, vals, length, all_class_prob[0], all_attributes[0])

        if probability > 0.5 and y[0] == '+1':
            true_positive += 1
        elif probability < 0.5 and y[0] == '-1':
            true_negative += 1
        elif probability > 0.5 and y[0] == '-1':
            false_positive += 1
        elif probability < 0.5 and y[0] == '+1':
            false_negative += 1

        del vals[:]
        del attr[:]

    print str(true_positive) + " " + str(false_negative) + " " + str(false_positive) + " " + str(true_negative)

def train_data(X,max_val, attr_size):
    positives = 0
    negatives = 0

    attributes = defaultdict(dict)

    for x in X:
        y = x.split()
        if y[0] == '+1':
            positives += 1
        else:
            negatives += 1

        row_values = [j for j in x.split(" ")]
        for val in range(1, len(row_values)):
            split_vals = row_values[val].split(":")

            if attributes[int(split_vals[0])].has_key(int(split_vals[1])) == False:
                if y[0] == '+1':
                    attributes[int(split_vals[0])][int(split_vals[1])] = (1,0)
                else:
                    attributes[int(split_vals[0])][int(split_vals[1])] = (0,1)
            else:
                temp_pos = attributes[int(split_vals[0])][int(split_vals[1])][0]
                temp_neg = attributes[int(split_vals[0])][int(split_vals[1])][1]
                if y[0] == '+1':
                    attributes[int(split_vals[0])][int(split_vals[1])] = (temp_pos + 1, temp_neg)
                else:
                    attributes[int(split_vals[0])][int(split_vals[1])] = (temp_pos, temp_neg + 1)

    class_probability = positives/float(len(X))

    # Calculate conditional probabilities
    for key, values in attributes.items():
        for value, count in values.items():
            total_attr_count = count[0] + count[1]
            prob_pos = count[0]/float(positives)
            prob_neg = count[1]/float(negatives)
            attributes[key][value] = (prob_pos, prob_neg)

    for i in range(1, attr_size+1):
        for value in range(1, int(max_val)+1):
            if attributes.has_key(int(i)):
                if attributes[int(i)].has_key(value):
                    continue
                else:
                    attributes[int(i)][value] = (0.0,0.0)
            else:
                attributes[int(i)][value] = (0.0,0.0)

    return attributes, class_probability

def predict(X,attributes, values, length, class_prob, p_conditionals):
  positive = class_prob

  for i in range(length):
    if p_conditionals.has_key(attributes[i]) == True and p_conditionals[attributes[i]].has_key(values[i]) == True:
        # if p_conditionals[attributes[i]][int(values[i])][0] == 0.0:
        #     p_conditionals[attributes[i]][int(values[i])] = (1.0/float(len(X)), p_conditionals[attributes[i]][int(values[i])][1])
        positive *= p_conditionals[attributes[i]][int(values[i])][0]

  negative = (1.0 - class_prob)

  for i in range(length):
    if p_conditionals.has_key(attributes[i]) == True and p_conditionals[attributes[i]].has_key(values[i]) == True:
        # if p_conditionals[attributes[i]][int(values[i])][1] == 0.0:
        #     p_conditionals[attributes[i]][int(values[i])] = (p_conditionals[attributes[i]][int(values[i])][0],1.0/float(len(X)))
        negative *= p_conditionals[attributes[i]][int(values[i])][1]

  if positive >= negative:
    return 1.0
  else:
    return 0.0

def call_predict(X, attributes, class_prob):
    # false_positive = 0
    # false_negative = 0
    # true_negative = 0
    # true_positive = 0
    attr = []
    vals = []

    error = [0.0]*len(X)
    row_count = 0
    for x in X:
        y = x.split()

        row_values = [j for j in x.split(" ")]
        for val in range(1, len(row_values)):
            split_vals = row_values[val].split(":")
            attr.append(int(split_vals[0]))
            vals.append(int(split_vals[1]))
        length = len(vals)

        probability = predict(X,attr, vals, length, class_prob, attributes)
        # print probability
        if probability > 0.5 and y[0] == '+1':
            # true_positive += 1
            error[row_count] = 0.0
        elif probability < 0.5 and y[0] == '-1':
            # true_negative += 1
            error[row_count] = 0.0
        elif probability > 0.5 and y[0] == '-1':
            # false_positive += 1
            error[row_count] = 1.0
        elif probability < 0.5 and y[0] == '+1':
            # false_negative += 1
            error[row_count] = 1.0

        row_count += 1
        del vals[:]
        del attr[:]

    return error

def recalculate_weights(error_row, weights, total_row):
    error_total = 0.0
    weights_redone = [0.0] * total_row
    sum_weights = 0.0
    sum_weights_redone = 0.0
    sum_again = 0.0

    for i in range(total_row):
        if error_row[i] == 1.0:
            error_total += weights[i]

        sum_weights += weights[i]
    # print M
    error_cl = error_total

    for i in range(total_row):
        if error_row[i] == 0.0:
            weights[i] *= (error_total/float((1-error_total)))
        sum_weights_redone += weights[i]

    # Normalizing the weights
    for i in range(total_row):
        weights[i] *=  (sum_weights/float(sum_weights_redone))
        sum_again += weights[i]

    return error_cl, weights

# script, train, test = sys.argv
assert len(sys.argv) == 3, "Enter 3 arguments. Program Name, Train File & Test File."
train = os.path.abspath(sys.argv[1])
test = os.path.abspath(sys.argv[2])

X = open(train).read()
A = open(test).read()
X = X.split("\n")
X = filter(None, X)

A = A.split("\n")
A = filter(None, A)

attr_train = defaultdict(dict)
attributes_test = defaultdict(dict)


for x in X:

    row_values = [j for j in x.split(" ")]
    for val in range(1, len(row_values)):
        split_vals = row_values[val].split(":")

        if attr_train[int(split_vals[0])].has_key(int(split_vals[1])) == False:

            attr_train[int(split_vals[0])][int(split_vals[1])] = 1
        else:
            attr_train[int(split_vals[0])][int(split_vals[1])] += 1

for a in A:

    row_values = [j for j in a.split(" ")]
    for val in range(1, len(row_values)):
        split_vals = row_values[val].split(":")

        if attributes_test[int(split_vals[0])].has_key(int(split_vals[1])) == False:

            attributes_test[int(split_vals[0])][int(split_vals[1])] = 1
        else:
            attributes_test[int(split_vals[0])][int(split_vals[1])] += 1

temp  = 0
temp2 = 0
for key, value in attr_train.items():
    if temp < max(value):
        temp = max(value)


for key, value in attributes_test.items():
    if temp2 < max(value):
        temp2 = max(value)

if temp2 > temp:
    max_val = temp2
else:
    max_val = temp

if max(attr_train) > max(attributes_test):
    attr_size = max(attr_train)
else:
    attr_size = max(attributes_test)


weights = [1/float(len(X))] * len(X)
error_class = [0.0] * 100
p_conditionals = [defaultdict(dict)] * 40
class_prob = [0.0] * 40
k = 0
while (True):
    if k == 5:
        break
    elif k == 0:
        p_conditionals[k],class_prob[k] = train_data(X, max_val, attr_size)
        error = call_predict(X, p_conditionals[k], class_prob[k])
        if sum(error) == 0.0:
            break
        error_class[k], weights = recalculate_weights(error, weights, len(X))
        k += 1
    else:

        indices = np.random.multinomial(len(X), weights, 1)

        index = 0
        count = 0
        for idx in indices[0]:
            for int_idx in range(idx):
                X[index] = X[count]
                index += 1
            count += 1

        pos = 0
        neg = 0
        for x in X:
            x_first = x.split()
            if x_first[0] == "+1":
                pos += 1
            else:
                neg += 1

        if pos == len(X) or neg == len(X):
            X = open(train).read()
            X = X.split("\n")
            X = filter(None, X)
            continue

        del error[:]
        p_conditionals[k],class_prob[k] = train_data(X,max_val,attr_size)
        X = open(train).read()
        X = X.split("\n")
        X = filter(None, X)

        error = call_predict(X, p_conditionals[k], class_prob[k])
        if sum(error) == 0.0:
            X = open(train).read()
            X = X.split("\n")
            X = filter(None, X)
            break
        # print k
        error_class[k], weights = recalculate_weights(error, weights, len(X))

        if error_class[k] > 0.5:
            # print error_class
            error_class[k] = 0.0
        else:
            k += 1

        X = open(train).read()
        X = X.split("\n")
        X = filter(None, X)


X = open(train).read()
X = X.split("\n")
X = filter(None, X)

# Predict using ensemble of classifiers
ensemble_predict(X, k, p_conditionals, class_prob, error_class)
ensemble_predict(A, k, p_conditionals, class_prob, error_class)
