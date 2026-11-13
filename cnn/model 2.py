import torch
import torch.nn as nn

'''
Formula for output size of any conv/pool layer: out = floor( (in + 2·padding − kernel) / stride ) + 1
Formula for padding to keep the size the same (for odd kernels): padding = (kernel - 1) / 2

Terminology for convolutional layers:
- in_ch: number of input channels (e.g., 3 for RGB images)
- out_ch: number of output channels (number of filters)
- kernel_size: size of the convolutional kernel (e.g., 3 for a 3x3 kernel)
- stride: step size for moving the kernel across the input
- padding: number of pixels added to the input on each side

Terminology for neural network:
- dropout: a regularization technique where randomly selected neurons are ignored during training to prevent overfitting
'''


class ConvBlock(nn.Module):
    # in_ch is the number of input channels, and out_ch is the number of output channels. 
    # the channels correspond to the number of filters in the convolutional layer
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            # first convolution –> batch normalization –> ReLU activation

            # each of these returns a tensor, and the weights in the convolutional layer are learned during training.
            # we set kernel to 3 for now, but can change later, padding to 1 to keep output size same as input size, and bias to False since we are using batch normalization
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),

            # batch normalization normalizes the output of the conv2d layer, setting mean to 0 and variance to 1.
            # this is beneficial in training
            nn.BatchNorm2d(out_ch),

            # any negative values in the output of the batch normalization layer are set to 0, and positive values are kept as is.
            # this is what allows us to have a non-linear approximation
            nn.ReLU(inplace=True),


            # second convolution –> batch normalization –> ReLU activation
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            # max pooling layer with kernel size of 2 to halve size. we can also try with average pool to see how it works
            nn.MaxPool2d(kernel_size=2, stride = 2),
            # nn.AvgPool2d(kernel_size=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)
    
class CNN(nn.Module):
    # num classes refers to glyoma, meningioma, pituitary tumor, and no tumor
    def __init__(self, in_channels=1, num_classes=4, channels=(32, 64, 128, 256), dropout=0.5):
        super().__init__()

        # Here we build the ConvBlock classes. We automate with a loop incase we want to play around with number of channels
        blocks = []
        in_ch = in_channels
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            # we use adaptive in case input size is not fixed, and it will always return value. it returns the average of the input tensor across the spatial dimensions (height and width) and outputs a tensor of size (batch_size, channels, 1, 1)
            nn.AdaptiveAvgPool2d(1),
            # flatten to a 1D tensor so we can pass it to linear layer
            nn.Flatten(),
            # use dropout to prevent overfitting
            nn.Dropout(dropout),
            # linear layer to output the number of classes
            nn.Linear(channels[-1], num_classes),
)
    def forward(self, x):
        x = self.features(x)     
        x = self.classifier(x)   
        return x
    
if __name__ == "__main__":
    model = CNN()
    x = torch.randn(8, 1, 224, 224)  # fake batch: 8 grayscale 224x224 images

    # shape after each ConvBlock
    h = x
    for i, block in enumerate(model.features, start=1):
        h = block(h)
        print(f"after block {i}: {tuple(h.shape)}")

    # the whole model in one call
    out = model(x)
    print(f"output: {tuple(out.shape)}")

    # total number of learnable weights
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")