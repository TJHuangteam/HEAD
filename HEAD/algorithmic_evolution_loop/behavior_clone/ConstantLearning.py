import torch
import torch.nn as nn
import torch.nn.functional as F

class MoPELinearBlock(nn.Module):
    def __init__(self, col, depth, n_in, n_out):
        super(MoPELinearBlock, self).__init__()
        self.w = nn.Linear(n_in, n_out)
        self.u = nn.ModuleList([nn.Linear(n_in, n_out) for _ in range(col)]) if depth > 0 else None

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        cur_column_out = self.w(inputs[-1])
        if self.u:
            prev_columns_out = [mod(x) for mod, x in zip(self.u, inputs)]
            return F.relu(cur_column_out + sum(prev_columns_out))
        return F.relu(cur_column_out)

class MoPELastBlock(nn.Module):
    def __init__(self, col, depth, n_in, n_out):
        super(MoPELastBlock, self).__init__()
        self.w = nn.Linear(n_in, n_out)
        self.u = nn.ModuleList([nn.Linear(n_in, n_out) for _ in range(col)]) if depth > 0 else None

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        cur_column_out = self.w(inputs[-1])
        if self.u:
            prev_columns_out = [mod(x) for mod, x in zip(self.u, inputs)]
            return cur_column_out + sum(prev_columns_out)
        return cur_column_out

class GatingNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

class MoPE(nn.Module):
    def __init__(self, n_layers):
        super(MoPE, self).__init__()
        self.n_layers = n_layers
        self.columns = nn.ModuleList([])
        self.gating_layer = None
        self.use_cuda = torch.cuda.is_available()

    def _initialize_gating_layer(self, input_size, num_experts):
        self.gating_layer = GatingNetwork(input_size, num_experts)
        if self.use_cuda:
            self.gating_layer.cuda()

    def _reset_gating_layer(self):
        if self.gating_layer is not None:
            for layer in self.gating_layer.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def _combine_expert_outputs(self, inputs):
        combined_input = torch.cat(inputs, dim=1)
        gating_weights = self.gating_layer(combined_input)
        combined_output = sum(gating_weights[:, i].unsqueeze(1) * inputs[i] for i in range(len(inputs)))
        return combined_output, gating_weights

    def forward(self, x, task_id=-1):
        assert self.columns, 'PNN 至少需要包含一列 (缺少对 new_task() 的调用)'
        inputs = [c[0](x) for c in self.columns]
        for l in range(1, self.n_layers):
            inputs = [column[l](inputs[:i + 1]) for i, column in enumerate(self.columns)]


        if self.gating_layer is None or len(self.columns) != self.gating_layer.fc.out_features:
            input_size = sum([inputs[i].size(1) for i in range(len(inputs))])
            self._initialize_gating_layer(input_size, len(self.columns))
        return self._combine_expert_outputs(inputs)

    def new_task(self, sizes):
        assert len(sizes) == self.n_layers + 1, "层数和尺寸不匹配。"
        task_id = len(self.columns)
        modules = [
            MoPELinearBlock(task_id, i, sizes[i], sizes[i + 1]) if i != self.n_layers - 1
            else MoPELastBlock(task_id, i, sizes[i], sizes[i + 1])
            for i in range(self.n_layers)
        ]
        self.columns.append(nn.ModuleList(modules))

        input_size = sizes[-1] * len(self.columns)
        self._initialize_gating_layer(input_size, len(self.columns))
        self._reset_gating_layer()

        if self.use_cuda:
            self.cuda()

    def freeze_columns(self, skip=None):
        skip = skip or []
        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False

    def parameters(self, col=None):
        if col is None:
            return super(MoPE, self).parameters()
        return self.columns[col].parameters()

    def cuda(self, *args, **kwargs):
        super(MoPE, self).cuda(*args, **kwargs)
        if self.gating_layer:
            self.gating_layer.cuda()

    def cpu(self):
        super(MoPE, self).cpu()
        if self.gating_layer:
            self.gating_layer.cpu()















