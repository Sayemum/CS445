"""Pure Python Decision Tree Classifier and Regressor.

Simple binary decision tree classifier and regressor.
Splits for classification are based on Gini impurity. Splits for
regression are based on variance.

Author: Sayemum Hassan
Version: 1

"""

import scalarflow as sf


def main():
    with sf.Graph() as g:
        x = sf.Constant(0, "x")
        y = sf.Constant(2, "y")
        
        l0 = sf.Pow(x, 2, "l0")
        l1 = sf.Pow(y, 2, "l1")
        
        l3 = sf.Add(l0, l1, "l3")
        
        l4 = sf.Pow(l3, 3, "l4")
        l5 = sf.Exp(x, "l5")
        
        l6 = sf.Multiply(l5, l4, "l6")
        
        g.gen_dot("scalarflow_autodiff.txt")


if __name__ == "__main__":
    main()
