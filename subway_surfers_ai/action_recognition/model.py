# subway_surfers_ai/action_recognition/model.py
import torch.nn as nn
import torch.nn.functional as F

class ActionRecognitionCNN(nn.Module):
    """
    A simple 2D CNN model for action recognition.
    It treats multiple grayscale frames as a multi-channel 2D image.
    """
    def __init__(self, num_frames=8, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(num_frames, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        # Assuming input image size is 64x64
        # After three convolutions with stride=2, the feature map size becomes 8x8
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1) # Flatten
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x