import scalarflow as sf


def main():
    with sf.Graph() as g:
        x = sf.Constant(2.0, "x")
        # print(x.value)
        # print(x.derivative)
        # y = sf.Constant(2.0, "y")
        
        # sum = sf.Add(x, y)
        
        # result = g.run(sum)
        # print(result) # Prints 4.0
        # print()
        
        result = g.run(x)
        print(result)
        
        g.gen_dot("scalar_test.txt")


if __name__ == "__main__":
    main()
