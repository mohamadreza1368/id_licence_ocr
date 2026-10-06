import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    Lightweight Depthwise Separable Convolution (MobileNet style)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x

class MobileOCRNet(nn.Module):
    """
    Ultra-lightweight Mobile OCR Architecture (~4MB):
    - MobileNet-style Depthwise Separable Convolutions
    - BiLSTM for Persian contextual sequence modeling
    - Linear projection + CTC Loss
    """
    def __init__(self, num_classes, in_channels=3, rnn_hidden=128):
        super().__init__()

        # Feature Extractor: Height downsampled 32 -> 1, Width downsampled 4x
        self.features = nn.Sequential(
            # Stage 1: (32, W) -> (16, W/2)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: (16, W/2) -> (8, W/4)
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3: (8, W/4) -> (4, W/4)
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Stage 4: (4, W/4) -> (2, W/4)
            DepthwiseSeparableConv(128, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Stage 5: (2, W/4) -> (1, W/4)
            DepthwiseSeparableConv(256, 256),
            nn.Conv2d(256, 256, kernel_size=(2, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.Hardswish()
        )

        # Sequence Modeling (BiLSTM)
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )

        # Final Classifier
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        """
        Input: [B, C, H=32, W]
        Returns:
            log_probs: [T, B, num_classes] for CTC Loss
        """
        features = self.features(x)         # [B, 256, 1, W_seq]
        features = features.squeeze(2)      # [B, 256, W_seq]
        features = features.permute(0, 2, 1) # [B, W_seq, 256]

        rnn_out, _ = self.rnn(features)     # [B, W_seq, 2*rnn_hidden]
        logits = self.fc(rnn_out)           # [B, W_seq, num_classes]

        # PyTorch CTCLoss expects [T, B, num_classes]
        log_probs = logits.permute(1, 0, 2).log_softmax(2)
        return log_probs

    def predict(self, x):
        """
        Inference forward pass:
        Input: [B, C, H=32, W]
        Returns: [B, T] argmax predictions
        """
        log_probs = self.forward(x)          # [T, B, num_classes]
        # permute back to [B, T, num_classes]
        preds = log_probs.permute(1, 0, 2).argmax(dim=2)
        return preds
