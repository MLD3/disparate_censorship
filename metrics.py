from itertools import product 

import numpy as np

"""
    Adapted directly from https://github.com/CausalML/xauc/blob/master/xauc.py (Kallus et. al. 2019)
"""

'''
# from
#https://www.ibm.com/developerworks/community/blogs/jfp/entry/Fast_Computation_of_AUC_ROC_score?lang=en
# AUC-ROC = | {(i,j), i in pos, j in neg, p(i) > p(j)} | / (| pos | x | neg |)
# The equivalent version of this is, Pr [ LEFT > RIGHT ]
# now Y_true is group membership (of positive examples) , not positive level
'''
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)] #sort the predictions first
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n): # visit the examples in increasing order of predictions.
        y_i = y_true[i]
        nfalse += (1 - y_i) # negative (RIGHT) examples seen so far
        auc += y_i * nfalse # Each time we see a positive (LEFT) example we add the number of negative examples we've seen so far
    auc /= (nfalse * (n - nfalse))
    return auc

'''
cross_auc for the Ra0 > Rb1 error
function takes in scores for (a,0), (b,1)

Can also pass in R_a_1, R_b_1 for Pr[random group A positive > random group B positive] -> a different metric.
This treats an arbitrary group as a pseudo-negative class; here the goal is to see if one group is systematically
ranked higher than another. We *want* this to be ~50% (fair); values close to 1/0 = bias.
'''
def cross_auc(R_a_0, R_b_1):
    scores = np.concatenate([R_a_0, R_b_1])
    y_true = np.zeros(len(R_a_0)+len(R_b_1))
    y_true[0:len(R_a_0)] = 1 # Pr[ LEFT > RIGHT]; Y = 1 is the left (A0)
    return fast_auc(y_true, scores)

        
def cross_score(R_a_c, R_b_c):
    return np.fromiter(map(lambda x: x[1] - x[0], product(R_a_c, R_b_c)), float).mean()
    