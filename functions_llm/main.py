import math
import matplotlib.pyplot as plt
from geometric_functions import read_grid_size
from pattern_dispatcher import dispatch


def plot_pattern(points, grid_size, title="Generated Pattern"):
    """
    Plot (x, y, theta) points with headings as arrows.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    thetas = [p[2] for p in points]

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys, 'bo-', label='Path')
    plt.quiver(
        xs, ys,
        [math.cos(t) for t in thetas],
        [math.sin(t) for t in thetas],
        angles='xy', scale_units='xy', scale=5, color='r', label='Heading'
    )
    plt.title(title)
    plt.xlim(0, grid_size[0])
    plt.ylim(0, grid_size[1])
    plt.gca().set_aspect('equal', 'box')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    # Prompt user
    prompt = input("Enter your shape description:\n")

    # Generate points via dispatcher
    points = dispatch(prompt)
    if not points:
        print("No points generated.")
        return

    # Load grid size for plotting
    grid_size = read_grid_size()

    print(f"Generated {len(points)} points for prompt: '{prompt}'")

    # Plot
    plot_pattern(points, grid_size, title=prompt)


if __name__ == "__main__":
    main()
