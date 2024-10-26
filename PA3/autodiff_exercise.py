"""Pure Python Decision Tree Classifier and Regressor.

Simple binary decision tree classifier and regressor.
Splits for classification are based on Gini impurity. Splits for
regression are based on variance.

Author: Sayemum Hassan
Version: 1

"""

import scalarflow as sf

with sf.Graph() as g:
    a = sf.Constant(7.0)
    b = sf.Constant(3.0)
    sum = sf.Add(a, b)

    result = g.run(sum)
    print(result) # Prints 10.0

# a = sf.Constant(7.0)
# b = sf.Constant(3.0)
# sum = sf.Add(a, b)

# result = sf.get_current_graph.run(sum)
# print(result) # Prints 10.0