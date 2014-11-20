###############################################################################
# Overview:
# 1. Apriori Algorithm to Mine Frequent Patterns.
# 2. Min Support = 100 (Approx 1% of the total patterns given for all topics)
# 3. Derive Frequent Patterns and store in /patterns directory
# 4. Derive Closed Patterns and store in /closed directory
# 5. Derive Max Patterns and store in /max directory
# 6. Derive Ranks by Purity and store in /purity directory
# 7. Derive Coverage and store in /coverage directory
# 8. Derive Phraseness and store in /phraseness directory
#
# Author: Yogesh Rane
# Date: 10/23/2014
###############################################################################

import sys,os,math,operator

from itertools import chain, combinations
from collections import defaultdict

def GetItemsMinSupport(item_set, transaction_list, minSupport, freq_set):
        """calculates the support for items in the itemSet and returns a subset
       of the itemSet each of whose elements satisfies the minimum support"""
        new_itemset = set()
        local_set = defaultdict(int)

        for item in item_set:
                for transaction in transaction_list:
                        if item.issubset(transaction):
                                freq_set[item] += 1
                                local_set[item] += 1

        for item, count in local_set.items():
                if count >= minSupport:
                        new_itemset.add(item)

        return new_itemset


def joinSet(items, length):
  return set([i.union(j) for i in items for j in items if len(i.union(j)) == length])


def GetItemsTranList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = record.split()
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))              # Generate 1-itemSets
    return itemSet, transactionList


def runApriorialgo(data, minSupport):
    """run apriori algorithm"""
    item_set, transaction_list = GetItemsTranList(data)
    # This dictionary holds supports for all patters made
    freq_set = defaultdict(int)
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport
    large_set = dict()

    one_cset = GetItemsMinSupport(item_set,transaction_list,minSupport,freq_set)

    current_lset = one_cset

    i = 2
    while(current_lset != set([])):
        large_set[i-1] = current_lset
        current_lset = joinSet(current_lset, i)
        current_cset = GetItemsMinSupport(current_lset,transaction_list,minSupport,
                                                freq_set)
        current_lset = current_cset
        i = i + 1

    # Make a list of all the frequent patters found
    freqItems = []

    for key, value in large_set.items():
        freqItems.extend([(freq_set[item],tuple(item))
                           for item in value])

    # arrange them in descending order by support
    freqItems.sort(reverse=True)

    # Get numbers to words mapping from vocab file
    # numbers will be key
    words = {}
    with open("vocab.txt") as f:
        for line in f:
            (key, val) = line.split()
            words[int(key)] = val

    # Print all values, make files as per the topic file names
    file_to_make = file_name.split('-')
    file_new = "patterns/pattern-" + file_to_make[1]

    print "Frequent Patterns ==> " + file_new
    orig_stdout = sys.stdout
    f = file(file_new, 'w')
    sys.stdout = f

    patterns = {}
    for support, item in freqItems:
        print "%d " % support,
        nums = [int(n) for n in (' '.join(str(i) for i in item)).split()]
        n_tuple = frozenset(nums)
        patterns[n_tuple] = support
        for i in nums:
            print "%s" % words[i],
        print ""

    sys.stdout = orig_stdout
    f.close()

    return patterns, words

def calculate_closed_sets(patterns, words):
    """Calculate Closed Sets"""
    closed_set = {}
    closed = 1
    for pattern, support in patterns.items():
        for pattern2, support2 in patterns.items():
            if pattern != pattern2 and pattern.issubset(pattern2) and patterns[pattern] == patterns[pattern2]:
                closed = 0
                break

        if closed == 1:
            closed_set[pattern] = support
        elif closed == 0:
            closed = 1

    closed_sorted = sorted(closed_set, key=closed_set.get,reverse=True)

    file_to_make = file_name.split('-')
    file_new = "closed/closed-" + file_to_make[1]

    print "Closed Patterns ==> " + file_new
    orig_stdout = sys.stdout
    f = file(file_new, 'w')
    sys.stdout = f

    for pattern in closed_sorted:
        print "%d" % closed_set[pattern],
        nums = [int(n) for n in (' '.join(str(i) for i in pattern)).split()]
        for i in nums:
            print "%s" % words[i],
        print ""

    sys.stdout = orig_stdout
    f.close()

def calculate_max_sets(patterns, words):
    """Calculate Max Sets"""
    max_set = {}
    maxed = 1
    for pattern, support in patterns.items():
      for pattern2, support2 in patterns.items():
        if pattern != pattern2 and pattern.issubset(pattern2):
          # print "in max break"
          maxed = 0
          break
      if maxed == 1:
        max_set[pattern] = support
      elif maxed == 0:
        maxed = 1

    # Print Max Patterns and sort as per support values
    maxed_sorted = sorted(max_set, key=max_set.get,reverse=True)

    file_to_make = file_name.split('-')
    file_new = "max/max-" + file_to_make[1]

    print "Max Patterns ==> " + file_new
    print ""
    orig_stdout = sys.stdout
    f = file(file_new, 'w')
    sys.stdout = f

    for pattern in maxed_sorted:
        print "%d" % max_set[pattern],
        nums = [int(n) for n in (' '.join(str(i) for i in pattern)).split()]
        for i in nums:
            print "%s" % words[i],          # To derive words from numbers
        print ""

    sys.stdout = orig_stdout
    f.close()

def calculate_coverage(patterns, words):
    """Calculate Coverage of Patterns"""
    dt_val = {'0': 10047,
               '1': 9674,
               '2': 9959,
               '3': 10161,
               '4': 9845}

    coverage = {}
    file_to_make = file_name.split('-')
    file_new = "coverage/coverage-" + file_to_make[1]

    topic_number = file_to_make[1].split('.')
    topic_count = dt_val[topic_number[0]]
    for pattern, support in patterns.items():
      coverage[pattern] = float(support/float(topic_count))

    coverage_sorted = sorted(coverage, key=coverage.get,reverse=True)

    print "Coverage of Patterns ==> " + file_new

    orig_stdout = sys.stdout
    f = file(file_new, 'w')
    sys.stdout = f

    for pattern in coverage_sorted:
        print "%.4f" % coverage[pattern],
        nums = [int(n) for n in (' '.join(str(i) for i in pattern)).split()]
        for i in nums:
            print "%s" % words[i],          # To derive words from numbers
        print ""

    sys.stdout = orig_stdout
    f.close()

def calculate_phraseness(patterns, words):
    """Calculate Phraseness of Patterns"""
    dt_val = {'0': 10047,
               '1': 9674,
               '2': 9959,
               '3': 10161,
               '4': 9845}

    coverage_set = {}
    file_to_make = file_name.split('-')
    file_new = "phraseness/phraseness-" + file_to_make[1]

    print "Phraseness of Patterns ==> " + file_new
    print ""
    orig_stdout = sys.stdout
    f = file(file_new, 'w')
    sys.stdout = f

    phraseness = {}
    topic_number = file_to_make[1].split('.')
    topic_count = dt_val[topic_number[0]]
    for pattern, support in patterns.items():
        add_val = 0.0000
        for pattern2, support2 in patterns.items():
          if pattern2.issubset(pattern):
            if pattern == pattern2 or len(pattern2) > 1:
              continue
            add_val += float(math.log(support2/float(topic_count)))
        phraseness[pattern] = float(math.log(support/float(topic_count)) - add_val)

    # Print Max Patterns and sort as per support values

    phraseness_sorted = sorted(phraseness, key=phraseness.get,reverse=True)

    for pattern in phraseness_sorted:
        print "%.4f" % phraseness[pattern],
        nums = [int(n) for n in (' '.join(str(i) for i in pattern)).split()]
        for i in nums:
            print "%s" % words[i],          # To derive words from numbers
        print ""

    sys.stdout = orig_stdout
    f.close()


def calculate_purity(file_patterns):
  """Calculate Purity for all frequent patterns found in the files"""

# file_patterns is dictinoary of dictionary which we get in patterns
# key in inner dictionary are patterns and value in inner dictionary is Support
# Make another dictionary with key as patterns and value as Purity
  dt_vals = {'00': 10047, '01': 17326, '02': 17988, '03': 17999, '04': 17820,
             '10': 17326, '11': 9674, '12': 17446, '13': 17902, '14': 17486,
             '20': 17988, '21': 17446,'22': 9959, '23': 18077, '24': 17492,
             '30': 17999, '31': 17902,'32': 18077, '33': 10161, '34': 17912,
             '40': 17820, '41': 17486, '42': 17492, '43': 17912, '44': 9845}

  for topic, patterns in file_patterns.items():
    file_to_make = file_name.split('-')
    file_new = "purity/purity-" + str(topic) + ".txt"

    print "Purity File ==> " + file_new
    orig_stdout = sys.stdout
    f = file(file_new, 'w')
    sys.stdout = f
    purity = {}
    dt = str(topic) + str(topic)
    patterns_sorted = sorted(patterns.items(), key=operator.itemgetter(1),reverse=True)
    for pattern,val in patterns_sorted:
      prev_support = 0

      support = int(patterns[pattern])   #Gives the support

      for topic_other, patterns_other in file_patterns.items():
        value_flag = 0
        if topic == topic_other:
          continue
        dtt = str(topic) + str(topic_other)
        for pattern_other, support_other in patterns_other.items():
          if pattern_other == pattern:
            value_flag = 1
            f_support = (support+support_other)/float(dt_vals[dtt])
            if prev_support < f_support:
              prev_support = f_support
            break
        if value_flag == 0:
          f_support = (support)/float(dt_vals[dtt])
          if prev_support < f_support:
            prev_support = f_support

      pur_supp = {}   #So that I am able to sort later by purity and support
      if prev_support == 0:
        pur_val = float(math.log(support/float(dt_vals[dt])))
      else:
        pur_val = float(math.log(support/float(dt_vals[dt])) - math.log(float(prev_support)))

      new_measure = float((pur_val*support)/float(dt_vals[dt]))
      pur_supp[new_measure] = support
    #   pur_supp[pur_val] = support

      purity[pattern] = pur_supp #Dictionary inside dictionary
# In purity dictionary, I am storing another dictionary which has support as key
# purity as value. Pattern is key of the outer dictionary

    purity_sorted = sorted(purity, key=purity.get,reverse=True)

# Uncomment to print and sort by purity
    for pattern in purity_sorted:
      for purity_val,supp in purity[pattern].items():
        print "%.4f" % purity_val,
        break
      nums = [int(n) for n in (' '.join(str(i) for i in pattern)).split()]
      for i in nums:
        print "%s" % words[i],          # To derive words from numbers
      print ""

    sys.stdout = orig_stdout
    f.close()

def readfile(filename):
	return open(filename).read().split('\n')


# Make folder for Frequent Patterns if it does not exist
path = "./patterns"
if not os.path.exists(path):
  os.mkdir(path)

# Make folder for Closed Patterns if it does not exist
path = "./closed"
if not os.path.exists(path):
  os.mkdir(path)

# Make folder for Max Patterns if it does not exist
path = "./max"
if not os.path.exists(path):
  os.mkdir(path)

# Make folder for Max Patterns if it does not exist
path = "./purity"
if not os.path.exists(path):
  os.mkdir(path)

path = "./coverage"
if not os.path.exists(path):
  os.mkdir(path)

path = "./phraseness"
if not os.path.exists(path):
  os.mkdir(path)


file_patterns = {} #to store patterns generated by all topic files
patterns_found = {}

for i in range(0,5):
  file_name = "topic-" + str(i) + ".txt"
  terms = readfile(file_name)

  minSupport = 100
  patterns_found, words = runApriorialgo(terms, minSupport)

  calculate_closed_sets(patterns_found, words)
  calculate_max_sets(patterns_found, words)

  calculate_coverage(patterns_found, words)       #Bonus
  calculate_phraseness(patterns_found, words)     #Bonus

  file_patterns[i] = patterns_found

calculate_purity(file_patterns)
