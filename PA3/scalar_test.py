import scalarflow as sf


def main():
    with sf.Graph() as g:
        x = sf.Constant(2.0, "x")
        # print(x.value)
        # print(x.derivative)
        # y = sf.Constant(2.0, "y")
        
        # sum = sf.Divide(x, y)
        unary_op = sf.Abs(x)
        
        # result = g.run(sum)
        # print(result) # Prints 4.0
        # print()
        
        result = g.run(unary_op, compute_derivatives=True)
        print(result)
        
        g.gen_dot("scalar_test.txt")


if __name__ == "__main__":
    main()
