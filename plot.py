import ast
import matplotlib.pyplot as plt
import math


def main():
    # Read the Haskell output
    with open("output.txt", "r") as f:
        s = f.read()

    # Parse into Python data structure: [[(x, y), ...], ...]
    data = ast.literal_eval(s)
    data.pop(0)
    # Plot each list as a separate curve
    for i, curve in enumerate(data):
        xs = [pt[0] for pt in curve]
        ys = [pt[1] for pt in curve]
        plt.plot(xs, ys, label=f"iter {i}")
    if data:
        xs_ref = [pt[0] for pt in data[0]]
        ys_ref = [math.sin(x) for x in xs_ref]
        plt.plot(xs_ref, ys_ref, label="sin(x)", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y_hat")
    plt.title("Network outputs over training iterations")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
