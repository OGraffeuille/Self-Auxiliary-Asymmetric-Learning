import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
import numpy as np

from copy import deepcopy
from itertools import combinations

#from utils_visualisation import plot_arch


# --------------------------------------------------------------------------------
# Define DeepLab Modules
# --------------------------------------------------------------------------------
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )]

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # take pre-defined ResNet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu

        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# --------------------------------------------------------------------------------
# DeepLabv3 models (for NYUv2 and cityscapes experiments)
# --------------------------------------------------------------------------------

class MTLDeepLabv3(nn.Module):
    def __init__(self, tasks, aux_transfer_function_nlayers=None):
        super(MTLDeepLabv3, self).__init__()
        backbone = ResnetDilated(resnet.resnet50())
        ch = [256, 512, 1024, 2048]

        self.tasks = tasks

        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)
        self.shared_layer1 = backbone.layer1
        self.shared_layer2 = backbone.layer2
        self.shared_layer3 = backbone.layer3
        self.shared_layer4 = backbone.layer4

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([DeepLabHead(ch[-1], self.tasks[t]) for t in self.tasks])

        # Add transfer function similar to fine-tune (used in certain setups, e.g. --decoder_only_aux)
        if aux_transfer_function_nlayers is not None:
            self.aux_transfer_functions = nn.ModuleDict({})
            for i, task in enumerate(self.tasks):
                if ".aux" in task:
                    self.transfer_function = nn.Sequential()
                    for j in range(aux_transfer_function_nlayers):
                        self.transfer_function.append(nn.Conv2d(ch[-1], ch[-1], kernel_size=3, stride=1, padding=1, bias=False))
                        self.transfer_function.append(nn.BatchNorm2d(ch[-1]))
                        self.transfer_function.append(nn.ReLU())
                    self.aux_transfer_functions[str(i)] = self.transfer_function


    def forward(self, x, verbose=False):
        _, _, im_h, im_w = x.shape

        # Shared convolution
        x = self.shared_conv(x)
        x = self.shared_layer1(x)
        x = self.shared_layer2(x)
        x = self.shared_layer3(x)
        x = self.shared_layer4(x)

        # Task specific decoders
        out = [x for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            if hasattr(self, "aux_transfer_functions"):
                if ".aux" in t:
                    out[i] = self.aux_transfer_functions[str(i)](out[i])
            out[i] = F.interpolate(self.decoders[i](out[i]), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if 'normal' in t:
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out

    def shared_modules(self):
        return [self.shared_conv,
                self.shared_layer1,
                self.shared_layer2,
                self.shared_layer3,
                self.shared_layer4]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

class MTANDeepLabv3(nn.Module):
    def __init__(self, tasks):
        super(MTANDeepLabv3, self).__init__()
        backbone = ResnetDilated(resnet.resnet50())
        ch = [256, 512, 1024, 2048]

        self.tasks = tasks
        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)

        # We will apply the attention over the last bottleneck layer in the ResNet.
        self.shared_layer1_b = backbone.layer1[:-1]
        self.shared_layer1_t = backbone.layer1[-1]

        self.shared_layer2_b = backbone.layer2[:-1]
        self.shared_layer2_t = backbone.layer2[-1]

        self.shared_layer3_b = backbone.layer3[:-1]
        self.shared_layer3_t = backbone.layer3[-1]

        self.shared_layer4_b = backbone.layer4[:-1]
        self.shared_layer4_t = backbone.layer4[-1]

        # Define task specific attention modules using a similar bottleneck design in residual block
        self.encoder_att_1 = nn.ModuleList([self.att_layer(ch[0], ch[0] // 4, ch[0]) for _ in self.tasks])
        self.encoder_att_2 = nn.ModuleList([self.att_layer(2 * ch[1], ch[1] // 4, ch[1]) for _ in self.tasks])
        self.encoder_att_3 = nn.ModuleList([self.att_layer(2 * ch[2], ch[2] // 4, ch[2]) for _ in self.tasks])
        self.encoder_att_4 = nn.ModuleList([self.att_layer(2 * ch[3], ch[3] // 4, ch[3]) for _ in self.tasks])

        # Define task shared attention encoders using residual bottleneck layers
        self.encoder_block_att_1 = self.conv_layer(ch[0], ch[1] // 4)
        self.encoder_block_att_2 = self.conv_layer(ch[1], ch[2] // 4)
        self.encoder_block_att_3 = self.conv_layer(ch[2], ch[3] // 4)

        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([DeepLabHead(ch[-1], self.tasks[t]) for t in self.tasks])

    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid()
        )

    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(nn.Conv2d(in_channel, 4 * out_channel, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(4 * out_channel))
        return resnet.Bottleneck(in_channel, out_channel, downsample=downsample)

    def forward(self, x, verbose=False):
        _, _, im_h, im_w = x.shape

        # Shared convolution
        x = self.shared_conv(x)

        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)

        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Attention block 1 -> Apply attention over last residual block
        a_1_mask = [att_i(u_1_b) for att_i in self.encoder_att_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]

        # Attention block 2 -> Apply attention over last residual block
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]

        # Attention block 3 -> Apply attention over last residual block
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]

        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]

        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](a_4[i]), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if 'normal' in t:
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out

    def shared_modules(self):
        return [self.shared_conv,
                self.shared_layer1_b,
                self.shared_layer1_t,
                self.shared_layer2_b,
                self.shared_layer2_t,
                self.shared_layer3_b,
                self.shared_layer3_t,
                self.shared_layer4_b,
                self.shared_layer4_t,
                self.encoder_block_att_1,
                self.encoder_block_att_2,
                self.encoder_block_att_3]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

class BranchDeepLabv3(nn.Module):
    def __init__(self, train_tasks, arch):
        super(BranchDeepLabv3, self).__init__()

        # Arch is a 2D tensor of shape (num_tasks, network_depth)
        # Each row corresponds to a task and contains the path through the network encoder
        # the element [i,j] corresponds to the module selected by task i at layer j

        self.train_tasks = train_tasks
        self.num_tasks = len(train_tasks)
        self.num_layers = 5

        self.arch_dict = arch
        self.arch_arr = np.array([self.arch_dict[task] for task in self.train_tasks])
        self.arch_path_strs = {task: ''.join(map(str, self.arch_arr[i])) for i, task in enumerate(self.train_tasks)}

        assert self.arch_arr.shape[
                   0] == self.num_tasks, "ERROR: Number of tasks and rows in arch must be the same, instead of {} and {}".format(
            self.arch_arr.shape[0], self.num_tasks)
        assert self.arch_arr.shape[
                   1] == self.num_layers, "ERROR: Number of layers in the Resnet-50 network must be {} instead of {}".format(
            self.arch_arr.shape[1], self.num_layers)

        self.num_modules = np.max(self.arch_arr, axis=0).astype(int) + 1

        # Print the path that will be taken by each task
        for i, task in enumerate(self.train_tasks):
            print("Task:", task, "Path:", self.arch_path_strs[task])

        # Generate multiple resnets to copy layers from
        backbone = ResnetDilated(resnet.resnet50())
        backbones = [deepcopy(backbone) for _ in range(max(self.num_modules))]
        ch = [256, 512, 1024, 2048]

        self.shared_conv = nn.ModuleList([
            nn.Sequential(backbones[i].conv1, backbones[i].bn1, backbones[i].relu1, backbones[i].maxpool)
            for i in range(self.num_modules[0])])
        self.shared_layer1 = nn.ModuleList([backbones[i].layer1 for i in range(self.num_modules[1])])
        self.shared_layer2 = nn.ModuleList([backbones[i].layer2 for i in range(self.num_modules[2])])
        self.shared_layer3 = nn.ModuleList([backbones[i].layer3 for i in range(self.num_modules[3])])
        self.shared_layer4 = nn.ModuleList([backbones[i].layer4 for i in range(self.num_modules[4])])

        self.layer_list = [self.shared_conv, self.shared_layer1, self.shared_layer2, self.shared_layer3,
                           self.shared_layer4]

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([DeepLabHead(ch[-1], self.train_tasks[t]) for t in self.train_tasks])

    def forward(self, x, verbose=False):

        _, _, im_h, im_w = x.shape

        ### BRANCHING ENCODER ###

        # Created a list of dictionaries to store the outputs of each layer
        # The dictionaries are indexed by the path through the network
        xs = [{} for i in range(self.num_layers + 1)]
        xs[0][""] = x
        for layer_ind, layer in enumerate(self.layer_list):

            # For each path through the network so far
            for x_path in xs[layer_ind]:

                # Determine the next branches to make, removing duplicates
                next_modules = set(
                    [int(self.arch_path_strs[task][layer_ind]) for task in self.train_tasks if self.arch_path_strs[task].startswith(x_path)])
                for module in next_modules:
                    branch_path = x_path + str(module)
                    xs[layer_ind + 1][branch_path] = layer[self.arch_arr[module, layer_ind]](xs[layer_ind][x_path])

        # Format the final branch outputs according to the original task ordering
        x_out = [xs[-1][self.arch_path_strs[task]] for task in self.train_tasks]

        ### TASK-SPECIFIC DECODERS ###

        for t, t_name in enumerate(self.train_tasks):
            x_out[t] = F.interpolate(self.decoders[t](x_out[t]), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if 'normal' in t_name:
                x_out[t] = x_out[t] / torch.norm(x_out[t], p=2, dim=1, keepdim=True)

        return x_out

    def shared_modules(self):
        shared_modules = []
        for i in range(self.num_layers):
            if self.num_modules[i] == 1:
                shared_modules.append(self.layer_list[i])
        return shared_modules

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

class FinetuneDeepLabv3(nn.Module):
    def __init__(self, task, load_filename, transfer_function_nlayers):
        super(FinetuneDeepLabv3, self).__init__()
        self.task = task
        self.task_name = list(task.keys())[0]
        self.transfer_function_nlayers = transfer_function_nlayers

        backbone = ResnetDilated(resnet.resnet50())
        ch = [256, 512, 1024, 2048]

        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)
        self.shared_layer1 = backbone.layer1
        self.shared_layer2 = backbone.layer2
        self.shared_layer3 = backbone.layer3
        self.shared_layer4 = backbone.layer4

        # Load existing encoder weights
        checkpoint = torch.load(f"logging/{load_filename}")
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        for mm in self.shared_modules():
            for param in mm.parameters():
                param.requires_grad = False

        # Add transfer function
        self.transfer_function_layers = nn.ModuleList()
        for i in range(transfer_function_nlayers):
            self.transfer_function_layers.append(nn.Conv2d(ch[-1], ch[-1], kernel_size=3, stride=1, padding=1, bias=False))
            self.transfer_function_layers.append(nn.BatchNorm2d(ch[-1]))
            self.transfer_function_layers.append(nn.ReLU())
        self.transfer_function = nn.Sequential(*self.transfer_function_layers)

        # Define task-specific decoders using ASPP modules
        self.decoder = DeepLabHead(ch[-1], self.task[self.task_name])


    def forward(self, x, verbose=False):
        _, _, im_h, im_w = x.shape

        # Shared convolution
        x = self.shared_conv(x)
        x = self.shared_layer1(x)
        x = self.shared_layer2(x)
        x = self.shared_layer3(x)
        x = self.shared_layer4(x)

        x = self.transfer_function(x)

        out = F.interpolate(self.decoder(x), size=[im_h, im_w], mode='bilinear', align_corners=True)
        if 'normal' in self.task_name:
            out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return [out]

    def shared_modules(self):
        return [self.shared_conv,
                self.shared_layer1,
                self.shared_layer2,
                self.shared_layer3,
                self.shared_layer4]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

class AsymSearch(nn.Module):
    def __init__(self, base_model, n_modules_per_layer, device, constrain_aux=False):
        super(AsymSearch, self).__init__()

        self.tasks = base_model.tasks
        self.n_tasks = len(self.tasks)
        self.n_layers = len(base_model.shared_modules())
        self.n_modules_per_layer = n_modules_per_layer
        self.tau = None
        self.constrain_aux = constrain_aux

        # Copy backbone shared modules such that all backbones have identical initialisation
        shared_modules = base_model.shared_modules()
        self.encoders = nn.ModuleList(
            [nn.ModuleList([deepcopy(shared_modules[i]) for _ in range(self.n_modules_per_layer)]) for i in
             range(self.n_layers)])

        # Copy the decoders for each task such that auxiliary tasks have identical initialisation
        self.task_inds = {t: i for i, t in enumerate(self.tasks)}
        self.decoders = nn.ModuleList(
            [deepcopy(base_model.decoders[self.task_inds[t.split(".aux")[0]]]) for t in self.tasks])

        const_v = 1e-3  # I suspect this might as well be 0

        # Define the search space for the architecture
        # self.alphas[layer] = a [task,i_input,i_output] dimensional tensor that defines connections between modules
        self.alphas = []  # nn.ParameterList()
        alpha = const_v * torch.randn(self.n_tasks, 1, self.n_modules_per_layer)
        self.alphas.append(nn.Parameter(alpha.to(device)))  # First layer
        for _ in range(1, self.n_layers):
            alpha = const_v * torch.randn(self.n_tasks, self.n_modules_per_layer,
                                          self.n_modules_per_layer)  # Intermediate layers
            self.alphas.append(nn.Parameter(alpha.to(device)))

        self.generate_gumbels()

        # Possibly add? Decoder layer

        # Define the loss weights for the tasks
        # self.lambdas = nn.Parameter(torch.ones(self.n_tasks))#, self.n_auxiliaries_per_task + 1))

    def forward(self, x, reuse_gumbels=False, verbose=False):

        _, _, im_h, im_w = x.shape

        # Gumbel distribution: equivalent to -ln(-ln(uniform(0,1))
        if not reuse_gumbels:
            self.generate_gumbels()

        # Get a sampled path through the network for a task
        # A path is a list of length n_layers of straightthrough estimator 1hot vectors which indicate which module to use at each layer
        def get_sample_path(t, restricted_path=None):
            sample_path = []
            input_ID = 0  # Index of cell used in previous layer; we sample sequentially based on the module used on the previous layer
            for layer in range(self.n_layers):
                # Based on GDAS implementation: https://github.com/D-X-Y/AutoDL-Projects/blob/main/exps/NAS-Bench-201-algos/GDAS.py
                while True:
                    alphas = self.alphas[layer][t][input_ID]
                    g = self.gumbels[layer][t][input_ID]
                    logits = (alphas.log_softmax(dim=0) + g) / self.tau
                    if restricted_path is not None:
                        logits[restricted_path[layer] == 1] = -float('inf')
                    probs = nn.functional.softmax(logits, dim=0)
                    index = probs.max(-1, keepdim=True)[1]
                    one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                    hardwts = one_h - probs.detach() + probs
                    if torch.isinf(probs).any() or torch.isnan(probs).any():  # torch.isinf(g).any() or
                        print("Repeat sampling process...")
                        continue
                    else:
                        input_ID = index.item()
                        sample_path.append(hardwts)
                        break
            return sample_path

        sample_arch = []

        # Generate paths normally (independently)
        if not self.constrain_aux:
            for t in range(self.n_tasks):
                sample_arch.append(get_sample_path(t))

        # Generate architecture for primary tasks first such that auxiliary tasks can't re-use the same modules
        else:
            sample_arch = [[] for _ in range(self.n_tasks)]
            for t, task_name in enumerate(self.tasks):
                if not ".aux" in task_name:
                    sample_arch[t] = get_sample_path(t)
            for t, task_name in enumerate(self.tasks):
                if ".aux" in task_name:
                    pri_task = self.task_inds[task_name.split(".aux")[0]]
                    sample_arch[t] = get_sample_path(t, restricted_path=sample_arch[pri_task])

        if verbose:
            print("Sampled architecture: ")
            for t in range(self.n_tasks):
                print("Task: ", t, "Arch: ", [ind.max(-1, keepdim=True)[1].item() for ind in sample_arch[t]])

        ### ENCODER PATHS ###

        # Compute sample arch for each task
        xs = [x.clone() for _ in range(self.n_tasks)]
        for t, t_name in enumerate(self.tasks):

            # Compute sample arch for each task
            for layer in range(self.n_layers):
                next_module_straight_through, next_module_index = sample_arch[t][layer].max(-1, keepdim=True)
                xs[t] = self.encoders[layer][next_module_index](xs[t] * next_module_straight_through)

            # Compute the final decoder layer
            xs[t] = self.decoders[t](xs[t])

            # Compute the final output for each task
            xs[t] = F.interpolate(xs[t], size=[im_h, im_w], mode='bilinear', align_corners=True)
            if 'normal' in t_name:
                xs[t] = xs[t] / torch.norm(xs[t], p=2, dim=1, keepdim=True)

        return xs

    # Gumbels are the random variable in the concrete distribution that enable differentiable sampling
    def generate_gumbels(self):
        self.gumbels = [-torch.empty_like(a).exponential_().log() for a in self.alphas]

    def set_gumbels(self, gumbels):
        self.gumbels = gumbels

    def get_gumbels(self):
        return self.gumbels

    def set_tau(self, epoch, epoch_max, tau_start, tau_end, exp=False):
        if exp:
            t_s = np.log(tau_start)
            t_e = np.log(tau_end)
            self.tau = np.exp(t_s - (t_s - t_e) * epoch / (epoch_max - 1))
        else:
            self.tau = tau_start - (tau_start - tau_end) * epoch / (epoch_max - 1)
        print("Setting tau to: {:.3f}.", self.tau)

    def get_alpha(self):
        return [a.detach().cpu().numpy() for a in self.alphas]

    def plot_arch(self, filename, epoch, lambdas=None):
        plot_arch(self.alphas, lambdas, self.tasks, self.tau, filename, epoch)

    def shared_modules(self):
        raise NotImplementedError

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

class DeepLabv3AsymTransferv1(nn.Module):
    def __init__(self, train_tasks, arch, asym_v1_no_alphas_test=False, device='cuda'):
        super(DeepLabv3AsymTransferv1, self).__init__()

        # Arch is a 2D tensor of shape (num_tasks, network_depth)
        # Each row corresponds to a task and contains the path through the network encoder
        # the element [i,j] corresponds to the module selected by task i at layer j

        self.train_tasks = train_tasks
        self.task_names = list(train_tasks.keys())
        self.num_tasks = len(train_tasks)
        self.num_layers = 5
        self.tau = None

        self.asym_v1_no_alphas_test = asym_v1_no_alphas_test

        self.arch_dict = arch
        self.arch_arr = np.array([self.arch_dict[task] for task in self.train_tasks])
        self.arch_path_strs = {task: ''.join(map(str, self.arch_arr[i])) for i, task in enumerate(self.train_tasks)}

        assert self.arch_arr.shape[0] == self.num_tasks, "ERROR: Number of tasks and rows in arch must be the same, instead of {} and {}".format(
            self.arch_arr.shape[0], self.num_tasks)
        assert self.arch_arr.shape[1] == self.num_layers, "ERROR: Number of layers in the Resnet-50 network must be {} instead of {}".format(
            self.arch_arr.shape[1], self.num_layers)
        assert not any([".aux" in t for t in self.train_tasks]), "ERROR: This architecture cannot have pre-defined auxiliary tasks"

        self.num_modules = np.max(self.arch_arr, axis=0).astype(int) + 1

        # Add auxiliary tasks
        n_aux = {t: 0 for t in self.task_names}
        unique_paths = list(set(self.arch_path_strs.values()))
        arch_base_path_strs = self.arch_path_strs.copy()
        self.aux_task_list = []
        for unique_path in unique_paths:
            for task, path in arch_base_path_strs.items():
                if path != unique_path:
                    n_aux[task] += 1
                    task_aux_name = f"{task}.aux{n_aux[task]:d}"
                    self.train_tasks[task_aux_name] = self.train_tasks[task]
                    self.arch_path_strs[task_aux_name] = unique_path
                    self.aux_task_list.append(task_aux_name)

        if not self.asym_v1_no_alphas_test:
            self.alphas = nn.Parameter(torch.zeros((len(self.aux_task_list),2)).to(device))
            self.generate_gumbels()

        # Print the path that will be taken by each task
        for i, task in enumerate(self.train_tasks):
            print("Task: ", task, "Path: ", self.arch_path_strs[task])

        # Generate multiple resnets to copy layers from
        backbone = ResnetDilated(resnet.resnet50())
        backbones = [deepcopy(backbone) for _ in range(max(self.num_modules))]
        ch = [256, 512, 1024, 2048]

        self.shared_conv = nn.ModuleList([
            nn.Sequential(backbones[i].conv1, backbones[i].bn1, backbones[i].relu1, backbones[i].maxpool)
            for i in range(self.num_modules[0])])
        self.shared_layer1 = nn.ModuleList([backbones[i].layer1 for i in range(self.num_modules[1])])
        self.shared_layer2 = nn.ModuleList([backbones[i].layer2 for i in range(self.num_modules[2])])
        self.shared_layer3 = nn.ModuleList([backbones[i].layer3 for i in range(self.num_modules[3])])
        self.shared_layer4 = nn.ModuleList([backbones[i].layer4 for i in range(self.num_modules[4])])

        self.layer_list = [self.shared_conv, self.shared_layer1, self.shared_layer2, self.shared_layer3,
                           self.shared_layer4]

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([DeepLabHead(ch[-1], self.train_tasks[t]) for t in self.train_tasks])


    def forward(self, x, reuse_gumbels=False, verbose=False):

        _, _, im_h, im_w = x.shape

        # Gumbel distribution: equivalent to -ln(-ln(uniform(0,1))
        if not reuse_gumbels and not self.asym_v1_no_alphas_test:
            self.generate_gumbels()

        ### BRANCHING ENCODER ###

        # Created a list of dictionaries to store the outputs of each layer
        # The dictionaries are indexed by the path through the network
        xs = [{} for i in range(self.num_layers + 1)]
        xs[0][""] = x
        for layer_ind, layer in enumerate(self.layer_list):

            # For each path through the network so far
            for x_path in xs[layer_ind]:

                # Determine the next branches to make, removing duplicates
                next_modules = set(
                    [int(self.arch_path_strs[task][layer_ind]) for task in self.train_tasks if
                     self.arch_path_strs[task].startswith(x_path)])
                for module in next_modules:
                    branch_path = x_path + str(module)
                    xs[layer_ind + 1][branch_path] = layer[self.arch_arr[module, layer_ind]](xs[layer_ind][x_path])

        if not self.asym_v1_no_alphas_test:
            logits = (self.alphas.log_softmax(dim=1) + self.gumbels) / self.tau
            # probs = nn.functional.softmax(logits, dim=0)
            # probs = torch.sigmoid(logits)
            probs = nn.functional.softmax(logits, dim=1) # fix the rest... but basically have sample between 2 params for each aux task, whether to include or not (like LTB)
            # index = probs.max(-1, keepdim=True)[1]
            # one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            one_h = torch.round(probs).int()
            hardwts = one_h - probs.detach() + probs
            if torch.isinf(self.gumbels).any() or torch.isinf(probs).any() or torch.isnan(probs).any():
                print("WARNING: Broken gumbels: ", (self.gumbels))

            #print("Alphas", self.alphas)
            #print("Gumbels: ", self.gumbels)
            #print("Logits: ", logits)
            #print("Probabilities: ", probs)
            #print("Hardwts: ", hardwts)
            #print()

        ### TASK-SPECIFIC DECODERS ###

        x_out = []
        for i, task in enumerate(self.train_tasks):
            x = xs[-1][self.arch_path_strs[task]]

            x = F.interpolate(self.decoders[i](x), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if 'normal' in task:
                x = x / torch.norm(x, p=2, dim=1, keepdim=True)

            if not self.asym_v1_no_alphas_test:
                if ".aux" in task:
                    x = x * hardwts[self.aux_task_list.index(task)]
            x_out.append(x)

        return x_out


    # Sample architecture parameters to determine which task decoders to use
    # Based on GDAS implementation: https://github.com/D-X-Y/AutoDL-Projects/blob/main/exps/NAS-Bench-201-algos/GDAS.py
    def generate_gumbels(self):
        self.gumbels = -torch.empty_like(self.alphas).exponential_().log()  # Gumbel distribution: equivalent to -ln(-ln(uniform(0,1))

    def set_gumbels(self, gumbels):
        self.gumbels = gumbels

    def get_gumbels(self):
        return self.gumbels

    def set_tau(self, epoch, epoch_max, tau_start, tau_end):
        self.tau = tau_start - (tau_start - tau_end) * epoch / (epoch_max - 1)
        print("Setting tau to: ", self.tau)

    def get_alpha(self):
        return self.alphas.detach().cpu().numpy()

    def shared_modules(self):
        print("TODO: Implement shared_modules for GeneralDeepLabv3")
        return [self.shared_conv,
                self.shared_layer1,
                self.shared_layer2,
                self.shared_layer3,
                self.shared_layer4]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

class AsymSearchDeepLabv3(nn.Module):
    def __init__(self, tasks, n_modules_per_layer, device='cuda'):
        super(AsymSearchDeepLabv3, self).__init__()

        # To test: init with randn or ones?

        self.tasks = tasks
        self.n_tasks = len(tasks)
        self.n_layers = 5
        self.n_modules_per_layer = n_modules_per_layer
        #self.n_auxiliaries_per_task = n_auxiliaries_per_task
        self.tau = None

        # Define the search space for the architecture
        # self.alphas[task][layer] = parameters that define connections between modules
        self.alphas = [[] for _ in range(self.n_tasks)]
        for t in range(self.n_tasks):
            # FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3FIX1e-3
            alpha = 1e3 * torch.randn(1, self.n_modules_per_layer)  # First layer
            self.alphas[t].append(nn.Parameter(alpha.to(device)))  
            for _ in range(1, self.n_layers): 
                alpha = 1e3 * torch.randn(self.n_modules_per_layer, self.n_modules_per_layer)  # Intermediate layers
                self.alphas[t].append(nn.Parameter(alpha.to(device)))                         
            #self.alphas[t].append(nn.Parameter(1e-3 * torch.randn(self.n_modules_per_layer, self.n_auxiliaries_per_task + 1)).to(device)) # Decoder layer

        # Define the loss weights for the tasks
        #self.lambdas = nn.Parameter(torch.ones(self.n_tasks))#, self.n_auxiliaries_per_task + 1))
        # Do we need these? or just use autolambda implementation uh?                

        # Generate multiple resnets to copy layers from
        backbones = [ResnetDilated(resnet.resnet50()) for _ in range(self.n_modules_per_layer)]
        ch = [256, 512, 1024, 2048]
        
        self.shared_conv = nn.ModuleList([nn.Sequential(backbones[i].conv1, backbones[i].bn1, backbones[i].relu1, backbones[i].maxpool) for i in range(self.n_modules_per_layer)])
        self.shared_layer1 = nn.ModuleList([backbones[i].layer1 for i in range(self.n_modules_per_layer)])
        self.shared_layer2 = nn.ModuleList([backbones[i].layer2 for i in range(self.n_modules_per_layer)])
        self.shared_layer3 = nn.ModuleList([backbones[i].layer3 for i in range(self.n_modules_per_layer)])
        self.shared_layer4 = nn.ModuleList([backbones[i].layer4 for i in range(self.n_modules_per_layer)])

        self.layer_list = [self.shared_conv, self.shared_layer1, self.shared_layer2, self.shared_layer3, self.shared_layer4]

        # Define task-specific decoders using ASPP modules (n_auxiliaries_per_task + 1 decoders per task)
        self.decoders = nn.ModuleList([DeepLabHead(ch[-1], self.tasks[t]) for t in self.tasks])


    def forward(self, x, verbose=False):

        _, _, im_h, im_w = x.shape

        # Sample a pathway through the network for each task
        sample_arch = []
        for t in range(self.n_tasks):
            sample_arch.append([])
            input_ID = 0 # Index of cell used in previous layer; we sample sequentially based on prev input?
            for layer in range(self.n_layers):
                sample_arch[t].append([])
                # Based on GDAS implementation: https://github.com/D-X-Y/AutoDL-Projects/blob/main/exps/NAS-Bench-201-algos/GDAS.py
                while True:
                    alphas = self.alphas[t][layer][input_ID]
                    gumbels = -torch.empty_like(alphas).exponential_().log()  # Gumbel distribution: equivalent to -ln(-ln(uniform(0,1))
                    logits = (alphas.log_softmax(dim=0) + gumbels) / self.tau
                    probs = nn.functional.softmax(logits, dim=0)
                    index = probs.max(-1, keepdim=True)[1]
                    one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                    hardwts = one_h - probs.detach() + probs
                    if torch.isinf(gumbels).any() or torch.isinf(probs).any() or torch.isnan(probs).any():
                        print("Repeat sampling process...")
                        continue
                    else:
                        input_ID = index.item()
                        sample_arch[t][layer] = hardwts
                        break
        
        #print("Sampled architecture: ")
        if torch.rand(1) < 0.01:
            for t in range(self.n_tasks):
                #for t_ID in range(self.n_auxiliaries_per_task + 1):
                print("Task: ", t, "Arch: ", [ind.max(-1, keepdim=True)[1].item() for ind in sample_arch[t]])
        
        ### ENCODER PATHS ###
                        
        # Compute sample arch for each task
        xs = [x.clone() for _ in range(self.n_tasks)]
        for t, t_name in enumerate(self.tasks):

            # Compute sample arch for each task
            for layer in range(self.n_layers):
                next_module_straight_through, next_module_index = sample_arch[t][layer].max(-1, keepdim=True)
                #print("Layer: ", layer, " next_module_straight_through", next_module_straight_through, " next_module_index ", next_module_index)
                xs[t] = self.layer_list[layer][next_module_index](xs[t] * next_module_straight_through)

            # Compute the final decoder layer
            xs[t] = self.decoders[t](xs[t])

            # Compute the final output for each task
            xs[t] = F.interpolate(xs[t], size=[im_h, im_w], mode='bilinear', align_corners=True)
            if 'normal' in t_name:
                xs[t] = xs[t] / torch.norm(xs[t], p=2, dim=1, keepdim=True)
            
        return xs

    def set_tau(self, epoch, epoch_max, tau_start, tau_end):
        self.tau = tau_start - (tau_start - tau_end) * epoch / (epoch_max - 1)
        print("Setting tau to: ", self.tau)

    def print_arch(self, filename="arch.png"):
        pass
        #plot_arch(self.alphas, self.tasks, filename)

    def shared_modules(self):
        print("TODO: Implement shared_modules for GeneralDeepLabv3")
        return [self.shared_conv,
                self.shared_layer1,
                self.shared_layer2,
                self.shared_layer3,
                self.shared_layer4]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()


# --------------------------------------------------------------------------------
# Define ResNet (for CelebA experiments)
# --------------------------------------------------------------------------------
class MTLResNet(nn.Module):
    def __init__(self, tasks):
        super(MTLResNet, self).__init__()
        self.tasks = tasks
        backbone = resnet.resnet18()
        ch = [64, 128, 256, 512]

        # take pre-defined ResNet, except AvgPool and FC
        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch[-1], ch[-1]),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Linear(ch[-1], self.tasks[t]),
            ) for t in self.tasks
        ])

    def forward(self, x, verbose=False):
        x = self.shared_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze()
        out = [self.fcs[i](x) for i, _ in enumerate(self.tasks)]
        return out

    def shared_modules(self):
        return [self.shared_conv,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4]

class FinetuneResNet(nn.Module):
    def __init__(self, task, load_filename="", transfer_function_nlayers=2):
        super(FinetuneResNet, self).__init__()
        self.task = task
        self.task_name = list(task.keys())[0]
        backbone = resnet.resnet18()
        ch = [64, 128, 256, 512]

        # take pre-defined ResNet, except AvgPool and FC
        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Load existing encoder weights
        checkpoint = torch.load(f"logging/{load_filename}")
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Freeze shared (loaded) weights
        for mm in self.shared_modules():
            for param in mm.parameters():
                param.requires_grad = False

        # Add transfer function
        self.transfer_function_layers = nn.ModuleList()
        for i in range(transfer_function_nlayers):
            self.transfer_function_layers.append(nn.Conv2d(ch[-1], ch[-1], kernel_size=3, stride=1, padding=1, bias=False))
            self.transfer_function_layers.append(nn.BatchNorm2d(ch[-1]))
            self.transfer_function_layers.append(nn.ReLU())
        self.transfer_function = nn.Sequential(*self.transfer_function_layers)

        self.fc = nn.Sequential(
                nn.Linear(ch[-1], ch[-1]),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Linear(ch[-1], self.task[self.task_name]),
            )

    def forward(self, x, verbose=False):
        x = self.shared_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.transfer_function(x)
        x = self.avgpool(x).squeeze()
        out = self.fc(x)
        return [out]

    def shared_modules(self):
        return [self.shared_conv,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4]

class BranchResNet(nn.Module):
    def __init__(self, train_tasks, arch):
        super(BranchResNet, self).__init__()

        # Arch is a 2D tensor of shape (num_tasks, network_depth)
        # Each row corresponds to a task and contains the path through the network encoder
        # the element [i,j] corresponds to the module selected by task i at layer j

        self.train_tasks = train_tasks
        self.num_tasks = len(train_tasks)
        self.num_layers = 5

        self.arch_dict = arch
        self.arch_arr = np.array([self.arch_dict[task] for task in self.train_tasks])
        self.arch_path_strs = {task: ''.join(map(str, self.arch_arr[i])) for i, task in enumerate(self.train_tasks)}

        assert self.arch_arr.shape[
                   0] == self.num_tasks, "ERROR: Number of tasks and rows in arch must be the same, instead of {} and {}".format(
            self.arch_arr.shape[0], self.num_tasks)
        assert self.arch_arr.shape[
                   1] == self.num_layers, "ERROR: Number of layers in the Resnet-50 network must be {} instead of {}".format(
            self.arch_arr.shape[1], self.num_layers)

        self.num_modules = np.max(self.arch_arr, axis=0).astype(int) + 1

        # Print the path that will be taken by each task
        for i, task in enumerate(self.train_tasks):
            print("Task:", task, "Path:", self.arch_path_strs[task])

        # Generate multiple resnets to copy layers from
        backbone = resnet.resnet18()
        backbones = [deepcopy(backbone) for _ in range(max(self.num_modules))]
        ch = [64, 128, 256, 512]

        self.shared_conv = nn.ModuleList([
            nn.Sequential(backbones[i].conv1, backbones[i].bn1, backbones[i].relu, backbones[i].maxpool)
            for i in range(self.num_modules[0])])
        self.shared_layer1 = nn.ModuleList([backbones[i].layer1 for i in range(self.num_modules[1])])
        self.shared_layer2 = nn.ModuleList([backbones[i].layer2 for i in range(self.num_modules[2])])
        self.shared_layer3 = nn.ModuleList([backbones[i].layer3 for i in range(self.num_modules[3])])
        self.shared_layer4 = nn.ModuleList([backbones[i].layer4 for i in range(self.num_modules[4])])
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.layer_list = [self.shared_conv, self.shared_layer1, self.shared_layer2, self.shared_layer3,
                           self.shared_layer4]

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch[-1], ch[-1]),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Linear(ch[-1], self.train_tasks[t]),
            ) for t in self.train_tasks
        ])


    def forward(self, x, verbose=False):

        _, _, im_h, im_w = x.shape

        ### BRANCHING ENCODER ###

        # Created a list of dictionaries to store the outputs of each layer
        # The dictionaries are indexed by the path through the network
        xs = [{} for i in range(self.num_layers + 1)]
        xs[0][""] = x
        for layer_ind, layer in enumerate(self.layer_list):

            # For each path through the network so far
            for x_path in xs[layer_ind]:

                # Determine the next branches to make, removing duplicates
                next_modules = set(
                    [int(self.arch_path_strs[task][layer_ind]) for task in self.train_tasks if self.arch_path_strs[task].startswith(x_path)])
                for module in next_modules:
                    branch_path = x_path + str(module)
                    xs[layer_ind + 1][branch_path] = layer[self.arch_arr[module, layer_ind]](xs[layer_ind][x_path])

        # Format the final branch outputs according to the original task ordering
        x_out = [xs[-1][self.arch_path_strs[task]] for task in self.train_tasks]
        x_out = [self.avgpool(x).squeeze() for x in x_out]


        ### TASK-SPECIFIC DECODERS ###

        for t, t_name in enumerate(self.train_tasks):
            x_out[t] = self.decoders[t](x_out[t])

        return x_out

    def shared_modules(self):
        shared_modules = []
        for i in range(self.num_layers):
            if self.num_modules[i] == 1:
                shared_modules.append(self.layer_list[i])
        return shared_modules

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

# --------------------------------------------------------------------------------
# Define VGG-16 (for CIFAR-100 experiments)
# --------------------------------------------------------------------------------
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn_list = nn.ModuleList()

        for i in range(num_classes):
            self.bn_list.append(nn.BatchNorm2d(num_features))

    def forward(self, x, y):
        out = self.bn_list[y](x)
        return out


class MTLVGG16(nn.Module):
    def __init__(self, num_tasks):
        super(MTLVGG16, self).__init__()
        filter = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.num_tasks = num_tasks

        # define VGG-16 block
        network_layers = []
        channel_in = 3
        for ch in filter:
            if ch == 'M':
                network_layers += [nn.MaxPool2d(2, 2)]
            else:
                network_layers += [nn.Conv2d(channel_in, ch, kernel_size=3, padding=1),
                                   ConditionalBatchNorm2d(ch, num_tasks),
                                   nn.ReLU(inplace=True)]
                channel_in = ch

        self.network_block = nn.Sequential(*network_layers)

        # define classifiers here
        self.classifier = nn.ModuleList()
        for i in range(num_tasks):
            self.classifier.append(nn.Sequential(nn.Linear(filter[-1], 5)))

    def forward(self, x, task_id):
        for layer in self.network_block:
            if isinstance(layer, ConditionalBatchNorm2d):
                x = layer(x, task_id)
            else:
                x = layer(x)

        x = F.adaptive_avg_pool2d(x, 1)
        pred = self.classifier[task_id](x.view(x.shape[0], -1))
        return pred

# --------------------------------------------------------------------------------
# Define Linear networks (for tabular datasets)
# --------------------------------------------------------------------------------
class MTLMLP(nn.Module):
    def __init__(self, tasks, d_features, d_hidden=32, n_shared_layers=3, n_task_layers=1):
        super(MTLMLP, self).__init__()
        self.tasks = tasks

        assert n_shared_layers > 0, "ERROR: Must have at least one shared layer."
        assert n_task_layers > 0, "ERROR: Must have at least one task-specific layer."

        shared_layers = nn.ModuleList()
        for i in range(n_shared_layers):
            d_in = d_features if i==0 else d_hidden
            d_out = d_hidden
            shared_layers.append(nn.Linear(d_in, d_out))
            shared_layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*shared_layers)

        self.decoders = nn.ModuleList()
        for i, task in enumerate(self.tasks):
            task_decoder = nn.ModuleList()
            for j in range(n_task_layers - 1):
                task_decoder.append(nn.Linear(d_hidden, d_hidden))
                task_decoder.append(nn.ReLU(inplace=True))
            task_decoder.append(nn.Linear(d_hidden, self.tasks[task]))
            self.decoders.append(nn.Sequential(*task_decoder))

    def forward(self, x, verbose=False):

        x = self.encoder(x)

        out = []
        for i, task in enumerate(self.tasks):
            out.append(self.decoders[i](x))

        return out

class FinetuneMLP(nn.Module):
    def __init__(self, task, d_features, d_hidden=32, n_shared_layers=3, n_task_layers=1, load_filename="", transfer_function_nlayers=0):
        super(FinetuneMLP, self).__init__()
        self.task = task
        self.task_name = list(task.keys())[0]
        self.transfer_function_nlayers = transfer_function_nlayers

        assert n_shared_layers > 0, "ERROR: Must have at least one shared layer."
        assert n_task_layers > 0, "ERROR: Must have at least one task-specific layer."

        self.shared_layers = nn.ModuleList()
        for i in range(n_shared_layers):
            d_in = d_features if i==0 else d_hidden
            d_out = d_hidden
            self.shared_layers.append(nn.Linear(d_in, d_out))
            self.shared_layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*self.shared_layers)

        # Load existing encoder weights
        checkpoint = torch.load(f"weights/{load_filename}")
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Freeze shared (loaded) weights
        for mm in self.shared_modules():
            for param in mm.parameters():
                param.requires_grad = False

        # Add transfer function
        self.transfer_function_layers = nn.ModuleList()
        for i in range(transfer_function_nlayers):
            self.transfer_function_layers.append(nn.Linear(d_hidden, d_hidden))
            self.transfer_function_layers.append(nn.ReLU())
        self.transfer_function = nn.Sequential(*self.transfer_function_layers)

        self.decoder_layers = nn.ModuleList()
        for j in range(n_task_layers - 1):
            self.decoder_layers.append(nn.Linear(d_hidden, d_hidden))
            self.decoder_layers.append(nn.ReLU(inplace=True))
        self.decoder_layers.append(nn.Linear(d_hidden, self.task[self.task_name]))
        self.decoder = nn.Sequential(*self.decoder_layers)

    def forward(self, x, verbose=False):

        x = self.encoder(x)

        if self.transfer_function_nlayers > 0:
            x = self.transfer_function(x)

        out = self.decoder(x)
        return [out]

    def shared_modules(self):
        return self.shared_layers

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

class BranchMLP(nn.Module):
    def __init__(self, train_tasks, arch, d_features, d_hidden=32):
        super(BranchMLP, self).__init__()

        self.train_tasks = train_tasks
        self.num_tasks = len(train_tasks)

        self.arch_dict = arch
        self.arch_arr = np.array([self.arch_dict[task] for task in self.train_tasks])
        self.arch_path_strs = {task: ''.join(map(str, self.arch_arr[i])) for i, task in enumerate(self.train_tasks)}
        self.num_layers = self.arch_arr.shape[1]

        assert self.arch_arr.shape[0] == self.num_tasks, "ERROR: Number of tasks and rows in arch must be the same, instead of {} and {}".format(
            self.arch_arr.shape[0], self.num_tasks)

        self.num_modules = np.max(self.arch_arr, axis=0).astype(int) + 1

        # Print the path that will be taken by each task
        for i, task in enumerate(self.train_tasks):
            print("Task: ", task, "Path: ", self.arch_path_strs[task])

        # Define base encoder layers
        self.layer_list = nn.ModuleList()
        for i in range(self.num_layers):
            d_in = d_features if i == 0 else d_hidden
            d_out = d_hidden
            layer_list_i = nn.ModuleList()
            for j in range(self.num_modules[i]):
                layer_list_i.append(
                    nn.Sequential(
                        nn.Linear(d_in, d_out),
                        nn.ReLU(inplace=True)
                    ))
            self.layer_list.append(layer_list_i)

        # Define task-specific decoders
        self.decoders = nn.ModuleList()
        for i, task in enumerate(self.train_tasks):
            self.decoders.append(nn.Linear(d_hidden, self.train_tasks[task]))

    def forward(self, x, verbose=False):

        xs = [{} for i in range(self.num_layers + 1)]
        xs[0][""] = x
        for layer_ind, layer in enumerate(self.layer_list):
                
            for x_path in xs[layer_ind]:

                next_modules = set(
                    [int(self.arch_path_strs[task][layer_ind]) for task in self.train_tasks if
                    self.arch_path_strs[task].startswith(x_path)])
                for module in next_modules:
                    branch_path = x_path + str(module)
                    xs[layer_ind + 1][branch_path] = layer[self.arch_arr[module, layer_ind]](xs[layer_ind][x_path])

        # Format the final branch outputs according to the original task ordering
        x_out = [xs[-1][self.arch_path_strs[task]] for task in self.train_tasks]

        ### TASK-SPECIFIC DECODERS ###

        for t, t_name in enumerate(self.train_tasks):
            x_out[t] = self.decoders[t](x_out[t])

        return x_out


    def shared_modules(self):
        shared_modules = []
        for i in range(self.num_layers):
            if self.num_modules[i] == 1:
                shared_modules.append(self.layer_list[i])
        return shared_modules

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()