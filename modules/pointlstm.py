import torch
import torch.nn as nn
import numpy as np


class PointLSTMCell(nn.Module):
    def __init__(self, pts_num, in_channels, hidden_dim, offset_dim, bias):
        super(PointLSTMCell, self).__init__()
        self.bias = bias
        self.pts_num = pts_num
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.offset_dim = offset_dim
        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d((None, 1)))
        self.conv = nn.Conv2d(in_channels=self.in_channels + self.offset_dim +
                              self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=(1, 1),
                              bias=self.bias)

    def forward(self, input_tensor, hidden_state, cell_state):
        hidden_state[:, :4] -= input_tensor[:, :4]
        combined = torch.cat([input_tensor, hidden_state], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv,
                                             self.hidden_dim,
                                             dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * cell_state + i * g
        h_next = o * torch.tanh(c_next)
        return self.pool(h_next), self.pool(c_next)

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.pts_num,
                            1).cuda(),
                torch.zeros(batch_size, self.hidden_dim, self.pts_num,
                            1).cuda())


class PointLSTM(nn.Module):
    def __init__(self,
                 pts_num,
                 in_channels,
                 hidden_dim,
                 offset_dim,
                 num_layers,
                 topk=16,
                 offsets=False,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False):
        super(PointLSTM, self).__init__()
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.bias = bias
        self.topk = topk
        self.offsets = offsets
        self.pts_num = pts_num
        self.in_channels = in_channels
        self.offset_dim = offset_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_in_channels = self.in_channels if i == 0 else self.hidden_dim[
                i - 1] + 4
            cell_list.append(
                PointLSTMCell(pts_num=self.pts_num,
                              in_channels=cur_in_channels,
                              hidden_dim=self.hidden_dim[i],
                              offset_dim=self.offset_dim,
                              bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        # batch, timestep, c, n (N points, M neighbor)
        if not self.batch_first:
            # (t, b, c, n) -> (b, t, c, n)
            input_tensor = input_tensor.permute(1, 0, 2, 3)
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []
        position = input_tensor[:, :, :4]  # B, T, 4, N
        if self.offsets:
            centroids = torch.mean(position[:, :, :3], dim=3)
            group_offsets = (centroids[:, :-1] - centroids[:, 1:])[:, :, :,
                                                                   None]
            group_ind = torch.cat((
                self.group_points(position[:, 0, :3],
                                  position[:, 0, :3],
                                  dim=2,
                                  topk=self.topk).unsqueeze(1),
                self.group_points(position[:, 1:, :3] + group_offsets,
                                  position[:, :-1, :3],
                                  dim=3,
                                  topk=self.topk),
            ),
                dim=1)
        else:
            group_ind = torch.cat((
                self.group_points(position[:, 0, :3],
                                  position[:, 0, :3],
                                  dim=2,
                                  topk=self.topk).unsqueeze(1),
                self.group_points(position[:, 1:, :3],
                                  position[:, :-1, :3],
                                  dim=3,
                                  topk=self.topk),
            ),
                dim=1)
        seq_len = input_tensor.shape[1]

        cur_layer_input = input_tensor.unsqueeze(-1)
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                past = 0 if t == 0 else t - 1
                center_pts = cur_layer_input[:,
                                             t].expand(-1, -1, -1, self.topk)
                h_with_pos = torch.cat((position[:, past].unsqueeze(-1), h),
                                       dim=1)
                h_grouped = h_with_pos.squeeze(-1).unsqueeze(1).expand(-1, self.pts_num, -1, -1). \
                    gather(3, group_ind[:, t].unsqueeze(2).
                           expand(-1, -1, self.hidden_dim[layer_idx] + self.offset_dim, -1)) \
                    .permute(0, 2, 1, 3)
                c_grouped = c.squeeze(-1).unsqueeze(1).expand(-1, self.pts_num, -1, -1). \
                    gather(3, group_ind[:, t].unsqueeze(2).expand(-1, -1, self.hidden_dim[layer_idx], -1)) \
                    .permute(0, 2, 1, 3)
                h, c = self.cell_list[layer_idx](
                    input_tensor=center_pts.clone(),
                    hidden_state=h_grouped.clone(),
                    cell_state=c_grouped.clone())
                output_inner.append(h)
            layer_output = torch.cat(
                (position.unsqueeze(-1), torch.stack(output_inner, dim=1)),
                dim=2)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list, group_ind

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    def group_points(self, array1, array2, dim, topk):
        dist, _, _ = self.array_distance(array1, array2, dim)
        dists, idx = torch.topk(dist, topk, -1, largest=False, sorted=False)
        return idx

    @staticmethod
    def array_distance(array1, array2, dim):
        # return array1.shape[-1] * array2.shape[-1] matrix
        distance_mat = array1.unsqueeze(dim + 1) - array2.unsqueeze(dim)
        mat_shape = distance_mat.shape
        mat_shape = mat_shape[:1] + (array1.shape[1], ) + mat_shape[2:]
        array1 = array1.unsqueeze(dim + 1).expand(mat_shape)
        array2 = array2.unsqueeze(dim).expand(mat_shape)
        distance_mat = torch.sqrt((distance_mat**2).sum(dim - 1))
        return distance_mat, array1, array2

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    @staticmethod
    def tensor2numpy(tensor, name="test"):
        np.save(name, tensor.cpu().detach().numpy())


class MLPBlock(nn.Module):
    def __init__(self, out_channel, dimension, with_bn=True):
        super(MLPBlock, self).__init__()
        self.layer_list = []
        if dimension == 1:
            for idx, channels in enumerate(out_channel[:-1]):
                if with_bn:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv1d(channels,
                                      out_channel[idx + 1],
                                      kernel_size=1),
                            nn.BatchNorm1d(out_channel[idx]),
                            nn.ReLU(inplace=True),
                        ))
                else:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv1d(channels,
                                      out_channel[idx + 1],
                                      kernel_size=1), ))
        elif dimension == 2:
            for idx, channels in enumerate(out_channel[:-1]):
                if with_bn:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv2d(channels,
                                      out_channel[idx + 1],
                                      kernel_size=(1, 1)),
                            nn.BatchNorm2d(out_channel[idx + 1]),
                            nn.ReLU(inplace=True),
                        ))
                else:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv2d(channels,
                                      out_channel[idx + 1],
                                      kernel_size=(1, 1)), ))
        self.layer_list = nn.ModuleList(self.layer_list)

    def forward(self, output):
        for layer in self.layer_list:
            output = layer(output)
        return output


class MotionBlock(nn.Module):
    def __init__(self, out_channel, dimension, embedding_dim):
        super(MotionBlock, self).__init__()
        self.layer_list = []
        self.embedding_dim = embedding_dim
        if dimension == 1:
            self.layer_list.append(
                nn.Sequential(
                    nn.Conv1d(embedding_dim, out_channel[-1], kernel_size=1),
                    nn.BatchNorm1d(out_channel[-1]),
                    nn.ReLU(inplace=True),
                ))
            for idx, channels in enumerate(out_channel[:-1]):
                self.layer_list.append(
                    nn.Sequential(
                        nn.Conv1d(channels,
                                  out_channel[idx + 1],
                                  kernel_size=1),
                        nn.BatchNorm1d(out_channel[idx + 1]),
                        nn.ReLU(inplace=True),
                    ))
        elif dimension == 2:
            self.layer_list.append(
                nn.Sequential(
                    nn.Conv2d(embedding_dim,
                              out_channel[-1],
                              kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channel[-1]),
                    nn.ReLU(inplace=True),
                ))
            for idx, channels in enumerate(out_channel[:-1]):
                self.layer_list.append(
                    nn.Sequential(
                        nn.Conv2d(channels,
                                  out_channel[idx + 1],
                                  kernel_size=(1, 1)),
                        nn.BatchNorm2d(out_channel[idx + 1]),
                        nn.ReLU(inplace=True),
                    ))
        self.layer_list = nn.ModuleList(self.layer_list)

    def forward(self, output):
        position_embedding = self.layer_list[0](output[:, :self.embedding_dim])
        feature_embedding = output[:, self.embedding_dim:]
        for layer in self.layer_list[1:]:
            feature_embedding = layer(feature_embedding)
        return position_embedding * feature_embedding


class GroupOperation(object):
    def __init__(self):
        pass

    def group_points(self, distance_dim, array1, array2, knn, dim):
        matrix, a1, a2 = self.array_distance(array1, array2, distance_dim, dim)
        dists, inputs_idx = torch.topk(matrix,
                                       knn,
                                       -1,
                                       largest=False,
                                       sorted=True)
        neighbor = a2.gather(
            -1,
            inputs_idx.unsqueeze(1).expand(dists.shape[:1] + (a2.shape[1], ) +
                                           dists.shape[1:]))
        offsets = array1.unsqueeze(dim + 1) - neighbor
        offsets[:, :3] /= torch.sum(offsets[:, :3]**2,
                                    dim=1).unsqueeze(1)**0.5 + 1e-8
        return offsets

    def st_group_points(self, array, interval, distance_dim, knn, dim):
        # array B, 4 + D, T, N
        batchsize, channels, timestep, num_pts = array.shape
        if interval // 2 > 0:
            array_padded = torch.cat(
                (array[:, :, 0].unsqueeze(2).expand(-1, -1, interval // 2, -1),
                 array, array[:, :, -1].unsqueeze(2).expand(
                     -1, -1, interval // 2, -1)),
                dim=2)
        else:
            array_padded = array
        # array B, 4 + D, T, N
        # array_padded B, 4 + D, T + (interval // 2) * 2, N
        neighbor_points = torch.zeros(
            batchsize, channels, timestep,
            num_pts * interval).to(array.device)  # B, 4 + D, T, interval * N
        for i in range(timestep):
            neighbor_points[:, :, i] = array_padded[:, :, i:i + interval].view(
                batchsize, channels, -1)
        matrix, a1, a2 = self.array_distance(array, neighbor_points,
                                             distance_dim, dim)
        # matrix: B, T, N, interval * N
        # a1: B, 4 + D, T, N, interval * N
        # a2: B, 4 + D, T, N, interval * N, a2 is the copies of neighbor_points
        dists, inputs_idx = torch.topk(
            matrix, knn, -1, largest=False,
            sorted=True)  # dists: B, T, N, K; inputs_idx: B, T, N, K
        neighbor = a2.gather(-1,
                             inputs_idx.unsqueeze(1).expand(
                                 dists.shape[:1] + (a2.shape[1], ) +
                                 dists.shape[1:]))  # B, 4 + D, T, N, K
        array = array.unsqueeze(-1).expand_as(neighbor)  # B, 4 + D, T, N, K
        ret_features = torch.cat(
            (array[:, :4] - neighbor[:, :4], array[:, 4:], neighbor[:, 4:]),
            dim=1)  # B, 4 + D + D, T, N, K
        return ret_features

    def array_distance(self, array1, array2, dist, dim):
        # return array1.shape[-1] * array2.shape[-1] matrix
        # array1: B, D, T, N1
        # array2: B, D, T, N2
        distance_mat = array1.unsqueeze(dim + 1)[:, dist] - array2.unsqueeze(
            dim)[:, dist]
        # B, 3, T, N1, 1 - B, 3, T, 1, N2 = B, 3, T, N1, N2
        mat_shape = distance_mat.shape  # B, 3, T, N1, N2
        mat_shape = mat_shape[:1] + (
            array1.shape[1], ) + mat_shape[2:]  # B, D, T, N1, N2
        array1 = array1.unsqueeze(dim + 1).expand(mat_shape)
        array2 = array2.unsqueeze(dim).expand(mat_shape)
        distance_mat = torch.sqrt((distance_mat**2).sum(1))
        return distance_mat, array1, array2


class PointLSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = MLPBlock([4, 16, 32], 2)
        self.pool1 = nn.AdaptiveMaxPool2d((None, 1))
        self.stage2 = MotionBlock([
            64,
            64,
        ], 2, 4)
        self.pool2 = nn.AdaptiveMaxPool2d((None, 1))
        self.stage4 = MotionBlock([
            256,
            256,
        ], 2, 4)
        self.pool4 = nn.AdaptiveMaxPool2d((None, 1))
        self.stage5 = MLPBlock([256, 1024], 2)

        self.group = GroupOperation()
        self.downsample = (2, 2, 2)
        self.knn = (16, 24, 48, 12)
        pts_size = 512
        self.lstm = PointLSTM(pts_num=pts_size // self.downsample[0],
                              in_channels=68,
                              hidden_dim=128,
                              offset_dim=4,
                              num_layers=1,
                              topk=5)

    def forward(self, data):  # (B, T, N, 4) 最后附带上了时间维，时间维也会做标准化
        inputs = data['human_points']
        B, T, N, _ = inputs.shape
        timestamp = torch.arange(T).reshape(1, T, 1, 1).repeat(B, 1, N, 1)
        timestamp = (timestamp - (T - 1) / 2) / (T - 1) * 2
        inputs = torch.cat((inputs, timestamp.to(inputs.device)), dim=-1)
        inputs = inputs.permute(0, 3, 1, 2)

        # B * 4 * T * N
        batchsize, in_dims, timestep, pts_num = inputs.shape

        # stage 1: intra-frame
        ret_array1 = self.group.group_points(distance_dim=[0, 1, 2],
                                             array1=inputs,
                                             array2=inputs,
                                             knn=self.knn[0],
                                             dim=3)  # B, 4, T, N, K
        ret_array1 = ret_array1.contiguous().view(batchsize, in_dims,
                                                  timestep * pts_num,
                                                  -1)  # B, 4, T * N, K
        fea1 = self.pool1(self.stage1(ret_array1)).view(
            batchsize, -1, timestep, pts_num)  # B, D1, T, N
        fea1 = torch.cat((inputs, fea1), dim=1)  # B, 4 + D1, T, N

        # stage 2: inter-frame, early
        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        ret_group_array2 = self.group.st_group_points(
            fea1, 3, [0, 1, 2], self.knn[1], 3)  # B, 4 + D1 + D1, T, N, K2
        ret_array2, inputs, _ = self.select_ind(ret_group_array2, inputs,
                                                batchsize, in_dims, timestep,
                                                pts_num)
        # ret_array2: B, 4 + D1 + D1, T * (N / 2), K2
        # inputs: B, 4, T, N / 2
        fea2 = self.pool2(self.stage2(ret_array2)).view(
            batchsize, -1, timestep, pts_num)  #
        # B, D2, T * (N / 2), K2 -> B, D2, T * (N / 2), 1 -> B, D2, T, N / 2
        fea2 = torch.cat((inputs, fea2), dim=1)  # B, 4 + D_2, T, N / 2

        # stage 3: inter-frame, middle, applying lstm in this stage
        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]

        output = self.lstm(fea2.permute(0, 2, 1, 3))
        fea3 = output[0][0].squeeze(-1).permute(0, 2, 1, 3)
        ret_group_array3 = self.group.st_group_points(fea2, 3, [0, 1, 2],
                                                      self.knn[2], 3)
        ret_array3, inputs, ind = self.select_ind(ret_group_array3, inputs,
                                                  batchsize, in_dims, timestep,
                                                  pts_num)
        fea3 = fea3.gather(-1,
                           ind.unsqueeze(1).expand(-1, fea3.shape[1], -1, -1))

        # stage 4: inter-frame, late
        in_dims = fea3.shape[1] * 2 - 4
        pts_num //= self.downsample[2]
        ret_group_array4 = self.group.st_group_points(fea3, 3, [0, 1, 2],
                                                      self.knn[3], 3)
        ret_array4, inputs, _ = self.select_ind(ret_group_array4, inputs,
                                                batchsize, in_dims, timestep,
                                                pts_num)
        fea4 = self.pool4(self.stage4(ret_array4)).view(
            batchsize, -1, timestep, pts_num)
        output = self.stage5(fea4)
        output = output.max(dim=-1)[0]
        output = output.transpose(1, 2)
        return output
        # return torch.cat((inputs, fea4), dim=1)

    def select_ind(self, group_array, inputs, batchsize, in_dim, timestep,
                   pts_num):
        ind = self.weight_select(group_array, pts_num)
        ret_group_array = group_array.gather(
            -2,
            ind.unsqueeze(1).unsqueeze(-1).expand(-1, group_array.shape[1], -1,
                                                  -1, group_array.shape[-1]))
        ret_group_array = ret_group_array.view(batchsize, in_dim,
                                               timestep * pts_num, -1)
        inputs = inputs.gather(
            -1,
            ind.unsqueeze(1).expand(-1, inputs.shape[1], -1, -1))
        return ret_group_array, inputs, ind

    @staticmethod
    def weight_select(position, topk):
        # select points with larger ranges
        weights = torch.max(torch.sum(position[:, :3]**2, dim=1), dim=-1)[0]
        dists, idx = torch.topk(weights, topk, -1, largest=True, sorted=False)
        return idx


if __name__ == '__main__':
    sampler = PointLSTMEncoder().cuda()
    x = torch.randn((2, 16, 512, 4)).cuda()
    print(sampler(x).shape)
