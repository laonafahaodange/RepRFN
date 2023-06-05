import torch


class CharbonnierLoss(torch.nn.Module):
    """L1 Charbonnierloss. from paper <<Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks>>"""

    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6  # 论文超参数

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)  # 论文里面没有取均值，这里应该是为了数值小
        return loss


class CustomLoss(torch.nn.Module):
    """pixle +fft L1 Charbonnierloss."""

    def __init__(self):
        super(CustomLoss, self).__init__()
        self.charbonnierloss = CharbonnierLoss()
        self.l1loss = torch.nn.L1Loss()

    def forward(self, X, Y):
        pixel_loss = self.charbonnierloss(X, Y)
        fft_loss = self.l1loss(torch.fft.fft2(X, dim=(-2, -1)),torch.fft.fft2(Y, dim=(-2, -1)))
        # fft_loss = self.charbonnierloss(torch.fft.fft2(X, dim=(-2, -1)),
        #                                 torch.fft.fft2(Y, dim=(-2, -1)))
        return 0.9 * pixel_loss + 0.1 * fft_loss


if __name__ == '__main__':
    input1 = torch.randn(2, 3, 256, 256)
    input2 = torch.randn(2, 3, 256, 256)
    custom_loss = CustomLoss()
    loss = custom_loss(input1, input2)
    print("loss:", loss)
