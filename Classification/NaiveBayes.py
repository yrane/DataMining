from collections import defaultdict
import sys ; import os

def predict(X,attributes, values, length, class_prob, p_conditionals):
  positive = class_prob

  # print "In positive"
  for i in range(length):
    if p_conditionals.has_key(attributes[i]) == True and p_conditionals[attributes[i]].has_key(values[i]) == True:
        if p_conditionals[attributes[i]][int(values[i])][0] == 0.0:
          p_conditionals[attributes[i]][int(values[i])] = (1.0/float(len(X)), p_conditionals[attributes[i]][int(values[i])][1])
        positive *= p_conditionals[attributes[i]][int(values[i])][0]

  negative = (1.0 - class_prob)
  # print "in negative"
  for i in range(length):
    if p_conditionals.has_key(attributes[i]) == True and p_conditionals[attributes[i]].has_key(values[i]) == True:
        if p_conditionals[attributes[i]][int(values[i])][1] == 0.0:
          p_conditionals[attributes[i]][int(values[i])] = (p_conditionals[attributes[i]][int(values[i])][0],1.0/float(len(X)))
        negative *= p_conditionals[attributes[i]][int(values[i])][1]

  if positive >= negative:
    return 1.0
  else:
    return 0.0

def call_predict(X, attributes):
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

        probability = predict(X,attr, vals, length, class_prob, attributes)
        # print probability
        if probability > 0.5 and y[0] == '+1':
            true_positive += 1
        elif probability <= 0.5 and y[0] == '-1':
            true_negative += 1
        elif probability > 0.5 and y[0] == '-1':
            false_positive += 1
        elif probability <= 0.5 and y[0] == '+1':
            false_negative += 1

        del vals[:]
        del attr[:]

    print str(true_positive) + " " + str(false_negative) + " " + str(false_positive) + " " + str(true_negative)


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

positives = 0
negatives = 0
i = 0

total_rows = len(X)

attributes = defaultdict(dict)
attributes_test = defaultdict(dict)


for x in X:
    y = x.split()
    if y[0] == '+1':
        positives += 1
    else:
        negatives += 1
    i += 1

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

class_prob = positives/float(len(X))

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
for key, value in attributes.items():
    if temp < max(value):
        temp = max(value)

for key, value in attributes_test.items():
    if temp2 < max(value):
        temp2 = max(value)

if temp2 > temp:
    max_val = temp2
else:
    max_val = temp

if max(attributes) > max(attributes_test):
    attr_size = max(attributes)
else:
    attr_size = max(attributes_test)

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


call_predict(X, attributes)
call_predict(A, attributes)
