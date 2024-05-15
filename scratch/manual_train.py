import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from arch import NN  # Assuming this is your neural network class
from matplotlib.widgets import Button, Slider

# Initialize the network and data as before
network = NN(layer_sizes=[2, 3, 2])
np.random.seed(0)
x_coords = np.random.uniform(-10, 10, 100)
y_coords = np.random.uniform(-10, 10, 100)
classes = np.where(x_coords**2 + y_coords**2 < 36, 0, 1)
data = [
    {
        "inputs": np.array([x, y]),
        "expected_outputs": np.array([c, 1 - c]),
    }
    for x, y, c in zip(x_coords, y_coords, classes)
]

# Function to classify input using the neural network
def classify_network(input_1, input_2, network):
    return network.forward([input_1, input_2])[0]

# Function to calculate total cost
def total_cost(data, network):
    cost = 0
    for item in data:
        prediction = network.forward(item["inputs"])
        cost += np.sum((prediction - item["expected_outputs"])**2)
    return cost / len(data)

# Setup figure and axis for plot and sliders
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.3, right=0.8)

# Initial plot setup
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.vectorize(lambda x, y: classify_network(x, y, network))(X, Y)
contour = plt.contourf(X, Y, Z, levels=[-1, 0.5, 2], cmap="bwr", alpha=0.5)
plt.scatter(x_coords, y_coords, c=["r" if cls == 1 else "b" for cls in classes])

# Create sliders for all network parameters
sliders = []
for layer_index, layer in enumerate(network.layers):
    for i in range(layer.weights.shape[0]):
        for j in range(layer.weights.shape[1]):
            ax_weight = plt.axes([0.82, 0.05 + 0.05 * len(sliders), 0.15, 0.02], facecolor='lightgoldenrodyellow')
            slider = Slider(ax_weight, f'W{layer_index}_{i}_{j}', -2.0, 2.0, valinit=layer.weights[i, j])
            sliders.append((slider, layer_index, i, j, 'weight'))

    for i in range(layer.biases.shape[0]):
        ax_bias = plt.axes([0.82, 0.05 + 0.05 * len(sliders), 0.15, 0.02], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_bias, f'B{layer_index}_{i}', -2.0, 2.0, valinit=layer.biases[i])
        sliders.append((slider, layer_index, i, None, 'bias'))

# Update function for sliders
def update(val, contour):
    for slider, layer_index, i, j, param_type in sliders:
        if param_type == 'weight':
            network.layers[layer_index].weights[i, j] = slider.val
        else:  # Bias
            network.layers[layer_index].biases[i] = slider.val

    Z = np.vectorize(lambda x, y: classify_network(x, y, network))(X, Y)
    for c in contour.collections:
        c.remove()  # Remove old contours
    contour = plt.contourf(X, Y, Z, levels=[-1, 0.5, 2], cmap="bwr", alpha=0.5)
    # Calculate and print total cost
    print(f"Total Cost: {total_cost(data, network):.4f}")
    fig.canvas.draw_idle()

for slider, _, _, _, _ in sliders:
    slider.on_changed(lambda val: update(val, contour))

plt.show()

