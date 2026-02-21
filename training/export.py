import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 144, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(144)
        self.conv4 = nn.Conv2d(144, 192, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(192)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(192 * 7 * 7, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.gelu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def export_to_tfjs(pytorch_model_path, output_dir):
    print("Loading model...")
    model = Net()
    model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
    model.eval()

    # Export to ONNX
    dummy_input = torch.randn(1, 1, 28, 28)
    onnx_path = "mnist.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      opset_version=12) # Use stable opset
    
    print(f"Model exported to {onnx_path}")
    print("To convert to TFJS, run:")
    print(f"tensorflowjs_converter --input_format=onnx --output_format=tfjs_layers_model {onnx_path} {output_dir}")

if __name__ == "__main__":
    if os.path.exists("mnist_cnn.pt"):
        export_to_tfjs("mnist_cnn.pt", "../web/public/model")
    else:
        print("mnist_cnn.pt not found. Train the model first.")
