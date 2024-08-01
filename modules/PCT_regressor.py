

import torch
import torch.nn as nn
import torch.nn.functional as F


from .layers import FCBlock


class PCTRegressor(nn.Module):
    def __init__(self, ref_vertices: torch.Tensor, conv_regression=True):
        super(PCTRegressor, self).__init__()
        assert ref_vertices.shape == torch.Size([3, 1723])
        self.ref_vertices = ref_vertices  # (3, M)
        self.encoder1 = PCTEncoder(in_channels=3)

        if conv_regression:
            self.conv1 = nn.Conv1d(1024 + 3, 512, 1)
            self.conv2 = nn.Conv1d(512, 256, 1)
            self.conv3 = nn.Conv1d(256, 3, 1)
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
        else:
            self.encoder2 = PCTEncoder(in_channels=1024 + 3)
            self.fc1 = FCBlock(1024, 1024)
            self.fc2 = FCBlock(1024, 1024)
            self.fc3 = FCBlock(1024, 1723 * 3, activation=None)
        self.conv_regression = conv_regression

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        global_feat = self.encoder1(x)
        global_feat = global_feat.view(
            B, -1, 1).expand(-1, -1, self.ref_vertices.shape[-1])
        ref_vertices = self.ref_vertices.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat((ref_vertices, global_feat), dim=1)
        if self.conv_regression:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.conv3(x)
        else:
            global_feat = self.encoder2(x)
            x = self.fc3(self.fc2(self.fc1(global_feat)))
            x = x.reshape(B, 3, -1)
        return x + ref_vertices
        # return x


class PCTEncoder(nn.Module):
    def __init__(self, in_channels):
        super(PCTEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, 1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SALayer(128)
        self.sa2 = SALayer(128)
        self.sa3 = SALayer(128)
        self.sa4 = SALayer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        B, _, N = x.shape

        x = F.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(B, -1)
        return x


class SALayer(nn.Module):
    def __init__(self, channels):
        super(SALayer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = F.relu(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


if __name__ == '__main__':
    points = torch.randn((4, 3, 512)).cuda()
    ref_vertices = torch.randn((3, 1723)).cuda()
    model = PCTRegressor(ref_vertices).cuda()
    # import netron
    # import torch.onnx
    # onnx_path = "mymodel.onnx"
    # torch.onnx.export(model, points, onnx_path)
    # netron.start(onnx_path)
