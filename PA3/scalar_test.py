import scalarflow as sf
import math


def main():
    with sf.Graph() as g:
        # x = sf.Constant(2.0, "x")
        # y = sf.Placeholder("y")
        
        # sum1 = sf.Add(x, y)
        # sum2 = sf.Add(sum1, y)
        
        # print(x.derivative, y.derivative, sum1.derivative, sum2.derivative)
        # result = g.run(sum1, feed_dict={'y': 2.0} ,compute_derivatives=True)
        # print(x.derivative, y.derivative, sum1.derivative, sum2.derivative)
        # print(result)
        
        
        
        
        # AUTODIFF Exercises Sheet Example
        # x = sf.Constant(0.0, "x")
        # y = sf.Constant(2.0, "y")
        
        # x_sq = sf.Pow(x, 2, "l0")
        # y_sq = sf.Pow(y, 2, "l1")
        
        # add_x_y_sq = sf.Add(x_sq, y_sq, "l2")
        
        # cubed_sq = sf.Pow(add_x_y_sq, 3, "l3")
        
        # e = sf.Constant(math.e, "e")
        # e_x = sf.Pow(e, x.value, "l4")
        
        # mult = sf.Multiply(e_x, cubed_sq, "l5")
        # result = g.run(mult, compute_derivatives=True)
        # print(result)
        
        # nodes = [x, y, x_sq, y_sq, add_x_y_sq, cubed_sq, e, e_x, mult]
        # for node in nodes:
        #     print(f"({node.value} - {node.derivative})", end=" ")
        
        
        
        
        # ReLU Example
        x = sf.Variable(-3.0)
        relu_node = sf.ReLU(x)
        
        result = g.run(relu_node)
        print(result)  # Prints 0.0 because ReLU(-3) is 0

        x.assign(4.0)
        result = g.run(relu_node)
        print(result)  # Prints 4.0 because ReLU(4) is 4
        
        g.gen_dot("scalar_test.txt")


if __name__ == "__main__":
    main()
