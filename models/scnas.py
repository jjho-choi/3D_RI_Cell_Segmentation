import abc
import torch
import torch.nn as nn


class UpReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, momentum=0.5, affine=True):
        super(UpReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            ReluConvBn(C_in, C_out, kernel_size, stride, padding, momentum=momentum, affine=affine)
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, None, 2, 'trilinear', align_corners=True)
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, momentum=0.5, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv_1 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.InstanceNorm3d(C_out, momentum=momentum, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class ReluConvBn(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1, momentum=0.5, affine=False):
        super(ReluConvBn, self).__init__()
        self.op = nn.Sequential(
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.InstanceNorm3d(C_out, momentum=momentum, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class SEBlock(nn.Module):
    def __init__(self, in_ch, r, stride):
        super().__init__()
        self.pre_x = None
        if stride > 1:
            self.pre_x = FactorizedReduce(in_ch, in_ch, affine=False)
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_ch, in_ch // r),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // r, in_ch),
            nn.Sigmoid()
        )
        self.in_ch = in_ch

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if self.pre_x:
            x = self.pre_x(x)
        x = x.mul(se_weight)
        return x


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return x.view(*(x.shape[:2]), -1).mean(2)


class ReductionCell(nn.Module):
    def __init__(self, C):
        super(ReductionCell, self).__init__()

        self.edge_s0_0 = FactorizedReduce(C, C)
        self.edge_s1_0 = ReluConvBn(C, C, 3, 2, 2, 2)

        self.edge_s0_1 = FactorizedReduce(C, C)
        self.edge_0_1 = nn.AvgPool3d(3, 1, 1)

        self.edge_0_2 = Identity()
        self.edge_1_2 = ReluConvBn(C, C, 3, 1, 1)

        self.edge_0_3 = ReluConvBn(C, C, 3, 1, 1)
        self.edge_2_3 = nn.AvgPool3d(3, 1, 1)

    def forward(self, s0, s1):
        node_0 = self.edge_s0_0(s0) + self.edge_s1_0(s1)
        node_1 = self.edge_s0_1(s0) + self.edge_0_1(node_0)
        node_2 = self.edge_0_2(node_0) + self.edge_1_2(node_1)
        node_3 = self.edge_0_3(node_0) + self.edge_2_3(node_2)

        node = torch.cat([node_1, node_2, node_3], dim=1)

        return node


class EncoderNormalCell(nn.Module):
    def __init__(self, C):
        super(EncoderNormalCell, self).__init__()

        self.edge_s0_0 = ReluConvBn(C, C, 3, 1, 1)
        self.edge_s1_0 = ReluConvBn(C, C, 3, 1, 2, 2)

        self.edge_s0_1 = nn.AvgPool3d(3, 1, 1)
        self.edge_0_1 = Identity()

        self.edge_s0_2 = ReluConvBn(C, C, 3, 1, 1)
        self.edge_0_2 = Identity()

        self.edge_s0_3 = Identity()

    def forward(self, s0, s1):
        node_0 = self.edge_s0_0(s0) + self.edge_s1_0(s1)
        node_1 = self.edge_s0_1(s0) + self.edge_0_1(node_0)
        node_2 = self.edge_s0_2(s0) + self.edge_0_2(node_0)
        node_3 = self.edge_s0_3(s0)

        node = torch.cat([node_1, node_2, node_3], dim=1)

        return node


class ExpansionCell(nn.Module):
    def __init__(self, C):
        super(ExpansionCell, self).__init__()

        self.edge_s0_0 = ReluConvBn(C, C, 3, 1, 1)
        self.edge_s1_0 = ReluConvBn(C, C, 3, 1, 3, 3)

        self.edge_s0_1 = Identity()
        self.edge_0_1 = Identity()

        self.edge_s1_2 = ReluConvBn(C, C, 3, 1, 2, 2)
        self.edge_1_2 = Identity()

        self.edge_s0_3 = Identity()
        self.edge_s1_3 = nn.AvgPool3d(3, 1, 1)

    def forward(self, s0, s1):
        node_0 = self.edge_s0_0(s0) + self.edge_s1_0(s1)
        node_1 = self.edge_s0_1(s0) + self.edge_0_1(node_0)
        node_2 = self.edge_s1_2(s1) + self.edge_1_2(node_1)
        node_3 = self.edge_s0_3(s0) + self.edge_s1_3(s1)

        node = torch.cat([node_1, node_2, node_3], dim=1)

        return node


class DecoderNormalCell(nn.Module):
    def __init__(self, C):
        super(DecoderNormalCell, self).__init__()

        self.edge_s1_0 = ReluConvBn(C, C, 3, 1, 2, 2)

        self.edge_s1_1 = Identity()
        self.edge_0_1 = Identity()

        self.edge_s1_2 = ReluConvBn(C, C, 3, 1, 2, 2)
        self.edge_1_2 = Identity()

        self.edge_s0_3 = Identity()
        self.edge_0_3 = nn.AvgPool3d(3, 1, 1)

    def forward(self, s0, s1):
        node_0 = self.edge_s1_0(s1)
        node_1 = self.edge_s1_1(s1) + self.edge_0_1(node_0)
        node_2 = self.edge_s1_2(s1) + self.edge_1_2(node_1)
        node_3 = self.edge_s0_3(s0) + self.edge_0_3(node_0)

        node = torch.cat([node_1, node_2, node_3], dim=1)

        return node


class Cell(nn.Module):
    def __init__(self, multiplier, C_prev_prev, C_prev, C, prev_resized):
        super(Cell, self).__init__()
        self.multiplier = multiplier
        self.C = C
        self.preprocess0, self.preprocess1 = self.preprocess(C_prev_prev, C_prev, self.C, prev_resized)
        self._ops = self.ops()
        self._seblock = SEBlock(C * self.multiplier, 6, stride=1)

    @abc.abstractmethod
    def ops(self):
        pass

    @abc.abstractmethod
    def preprocess(self, C_prev_prev, C_prev, C, prev_resized):
        pass

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        s = self._ops(s0, s1)
        o = self._seblock(s)

        return o


class CellEnc(Cell):
    def __init__(self, multiplier, C_prev_prev, C_prev, C, reduction, prev_resized):
        self.reduction = reduction
        super(CellEnc, self).__init__(multiplier, C_prev_prev, C_prev, C, prev_resized)

    def ops(self):
        if self.reduction:
            _ops = ReductionCell(self.C)
        else:
            _ops = EncoderNormalCell(self.C)
        return _ops

    def preprocess(self, C_prev_prev, C_prev, C, prev_resized):
        if prev_resized:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReluConvBn(C_prev_prev, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReluConvBn(C_prev, C, 1, 1, 0, affine=False)

        return self.preprocess0, self.preprocess1


class CellDec(Cell):
    def __init__(self, multiplier, C_prev_prev, C_prev, C, increase, prev_resized):
        self.increase = increase
        super(CellDec, self).__init__(multiplier, C_prev_prev, C_prev, C, prev_resized)

    def ops(self):
        if self.increase:
            return ExpansionCell(self.C)
        else:
            return DecoderNormalCell(self.C)

    def preprocess(self, C_prev_prev, C_prev, C, prev_resized):
        if prev_resized:
            self.preprocess0 = UpReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
            self.preprocess1 = ReluConvBn(C_prev, C, 1, 1, 0, affine=False)
        elif self.increase:
            self.preprocess0 = UpReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
            self.preprocess1 = UpReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        else:
            self.preprocess0 = ReluConvBn(C_prev_prev, C, 1, 1, 0, affine=False)
            self.preprocess1 = ReluConvBn(C_prev, C, 1, 1, 0, affine=False)
        return self.preprocess0, self.preprocess1


class ScNas(nn.Module):
    def __init__(self, num_feature, num_layers, num_multiplier, num_class):
        super(ScNas, self).__init__()
        self.C = num_feature
        self._layers = num_layers
        self._multiplier = num_multiplier
        self._num_classes = num_class

        C_curr = self._multiplier * self.C
        self.stem = nn.Sequential(
            nn.Conv3d(1, C_curr, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=False),
            SEBlock(C_curr, 6, stride=1),
            nn.InstanceNorm3d(C_curr, momentum=0.5, affine=False),
        )

        resize = [1, 0] * self._layers

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        self.enc_cells = nn.ModuleList()
        reduction_prev = False
        for i in range(self._layers):
            if resize[i] == 1:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = CellEnc(self._multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)

            reduction_prev = reduction
            self.enc_cells.append(cell)
            C_prev_prev, C_prev = C_prev, self._multiplier * C_curr

        self.dec_cells = nn.ModuleList()
        increase_prev = False
        for i in range(self._layers):
            if resize[len(resize) - 1 - i] == 1:
                C_curr //= 2
                increase = True
            else:
                increase = False

            cell = CellDec(self._multiplier, C_prev_prev, C_prev, C_curr, increase, increase_prev)
            C_prev_prev, C_prev = C_prev, self._multiplier * C_curr

            increase_prev = increase
            self.dec_cells.append(cell)

        self.out = nn.Conv3d(C_prev, self._num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        enc_feats = []
        s0 = s1 = self.stem(x)
        enc_feats.append(s1)

        for i, cell in enumerate(self.enc_cells):
            # print('enc{}'.format(i))
            s0, s1 = s1, cell(s0, s1)
            enc_feats.append(s1)
        for enc_feat in enc_feats:
            print(enc_feat.size())
        enc_feats.pop()

        for i, cell in enumerate(self.dec_cells):
            # print('dec{}'.format(i))
            s0, s1 = s1, cell(s0, s1)

            low = enc_feats.pop()
            s1 += low

        out = self.out(s1)

        return out

