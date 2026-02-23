# Web Application Documentation

The NeuralDigit frontend is a high-performance, interactive Single Page Application (SPA) built with Next.js and TensorFlow.js.

## 1. Core Technologies

- **Framework**: Next.js 15 (App Router)
- **Styling**: Tailwind CSS
- **ML Runtime**: TensorFlow.js (WASM/WebGL backend)
- **Animations**: Framer Motion
- **Icons**: Lucide React

## 2. Key Components

### `DrawingCanvas.tsx`
Handles the signature-like drawing interface.
- Uses HTML5 Canvas API.
- Implements touch and mouse event listeners.
- Provides a `onDraw` callback that passes the canvas element for inference.

### `NeuralNetworkVis.tsx`
The visualization engine.
- Renders an SVG representation of the network.
- **Animation Sync**: Uses a particle system that triggers when user draws.
- **Responsive SVG**: Scales based on the container size while maintaining aspect ratio.

### `PredictionResults.tsx`
Displays the top prediction and probability bars for all digits (0-9).

## 3. Preprocessing Logic

Before the image from the canvas can be sent to the model, it must match the MNIST format:

1.  **Resizing**: The canvas image is resized to 28x28 pixels.
2.  **Grayscale**: Extracted as a single channel.
3.  **Color Inversion**: Ensure the input is white digit on black background (standard MNIST).
4.  **Normalization**: Pixel values are scaled and normalized using `(x - 0.1307) / 0.3081`.
5.  **Channel Ordering**: Transposed to NCHW format `[1, 1, 28, 28]` to match the PyTorch model expectations.

## 4. "Unknown Digit" Logic

To prevent the model from confidently guessing when the user draws nonsense, we implemented a threshold check:
- If the maximum probability from the Softmax output is **below 65%**, the application displays an "Unknown" state.

## 5. Development

To run locally:
```bash
npm install
npm run dev
```

The site will be available at `http://localhost:3000`.
