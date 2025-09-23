import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN(nn.Module):
    def __init__(self, num_classes=4):
        super(DANN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.label_predictor = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self.domain_discriminator = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 2),
        )
        self.grl = GradientReversalFunction.apply

    def forward(self, input_data, alpha=1.0):
        features = self.feature_extractor(input_data)
        features_flat = features.view(features.size(0), -1)
        reversed_features = self.grl(features_flat, alpha)
        domain_output = self.domain_discriminator(reversed_features)
        class_output = self.label_predictor(features_flat)
        return class_output, domain_output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备是: {device}")

    dummy_input = torch.randn(16, 3, 64, 64)

    model = DANN(num_classes=4)
    model.to(device)

    print(f"模型参数所在的设备: {next(model.parameters()).device}")

    dummy_input = dummy_input.to(device)
    print(f"输入数据所在的设备: {dummy_input.device}")

    model.eval()
    with torch.no_grad():
        class_pred, domain_pred = model(dummy_input, alpha=0.1)

    print("\n模型已创建并测试")
    print(f"输入形状: {dummy_input.shape}")
    print(f"标签预测输出形状: {class_pred.shape}")
    print(f"领域预测输出形状: {domain_pred.shape}")
