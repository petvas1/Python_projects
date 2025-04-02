from itertools import (product,
                       permutations,
                       combinations,
                       combinations_with_replacement,
                       accumulate,
                       groupby)
import operator

# a = [1, 2]
# b = [3, 4]
# prod = product(a, b)
# print(list(prod))

# a = [1, 2, 3, 4]
# perm = permutations(a, 2)  # length = 2
# print(list(perm))
# comb = combinations(a, 2)
# print(list(comb))
# comb_wr = combinations_with_replacement(a, 2)
# print(list(comb_wr))

# acc = accumulate(a, func=operator.mul)
# print(list(acc))

# a = [1, 2, 5, 3, 4]
# acc = accumulate(a, func=max)
# print(list(acc))

def smaller_than_3(x):
    return x < 3


a = [1, 2, 3, 4]
group_obj = groupby(a, key=smaller_than_3)
for key, value in group_obj:
    print(key, list(value))

