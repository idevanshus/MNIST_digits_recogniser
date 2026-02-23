# Project Architecture

NeuralDigit is built as a decoupled system where the heavy lifting (training) is done offline, and the inference is executed entirely on the client-side for low latency.

## 1. Neural Network Architecture

The model is a modern Convolutional Neural Network (CNN) designed for high accuracy and robustness.

### Layer Specifications
| Layer Type | Configuration | Activation |
| :--- | :--- | :--- |
| **Input** | 1x28x28 (Grayscale) | - |
| **Conv 1** | 3x3, 48 filters, Padding 1 | GELU + BatchNorm |
| **Conv 2** | 3x3, 96 filters, Padding 1 | GELU + BatchNorm |
| **Pooling 1** | 2x2 MaxPool (Stride 2) | Dropout (30%) |
| **Conv 3** | 3x3, 144 filters, Padding 1 | GELU + BatchNorm |
| **Conv 4** | 3x3, 192 filters, Padding 1 | GELU + BatchNorm |
| **Pooling 2** | 2x2 MaxPool (Stride 2) | Dropout (30%) |
| **Flatten** | 192 * 7 * 7 = 9408 | - |
| **FC 1** | 256 Nodes | GELU + BatchNorm |
| **FC 2 (Output)**| 10 Nodes | Softmax |

### Key Improvements
- **GELU Activations**: Used instead of ReLU for smoother gradients.
- **Batch Normalization**: Applied after every convolution and the first fully connected layer to stabilize training.
- **Dropout**: Strategic use of 30% dropout to prevent overfitting.
- **Label Smoothing**: Cross-entropy loss with 0.1 smoothing for better generalization.

## 2. Inference Pipeline

The bridge between PyTorch and the browser involves a multi-step conversion process:

1.  **PyTorch to ONNX**: The `export.py` script uses `torch.onnx.export` to serialize the model.
2.  **ONNX to TFJS**: We use `tensorflowjs_converter` to transform the ONNX graph into a sharded web-friendly format.
3.  **Loading in TFJS**: The web app uses `tf.loadGraphModel` to load the `model.json` from the `/public` directory.

## 3. Visualization Engine

The `NeuralNetworkVis.tsx` component implements a custom SVG-based visualization.

- **Neuron Grid**: Automatically calculates positions for layers.
- **Curved Synapses**: Uses Cubic Bezier curves to create an organic "web" feel.
- **Real-time Activations**:
    - Convolutional layer activations are simulated based on the max predicted digit to show "feature detection".
    - Output neurons show the actual probabilities from the Softmax layer.
    - Signal particles flow along active paths to visualize data movement.

## 4. UI/UX Design

- **Glassmorphism**: The interface uses semi-transparent cards with backdrop blurs.
- **Responsive Layout**: Adapts from mobile to large desktops using CSS Grid.
- **Smooth Animations**: Powered by Framer Motion for transitions and state changes.
