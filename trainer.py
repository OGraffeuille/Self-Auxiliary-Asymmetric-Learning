import argparse
import datetime, time
import os.path
import pickle, csv
import re

import torch
import torch.optim as optim
import torch.utils.data.sampler as sampler

from auto_lambda import AutoLambda
from create_network import *
from create_dataset import *
from utils import *
#from utils_arch_from_alphas import *

parser = argparse.ArgumentParser(description='Multi-task/Auxiliary Learning: Dense Prediction Tasks')
parser.add_argument('--mode', default='none', type=str)
parser.add_argument('--port', default='none', type=str)

parser.add_argument('--name', default='exp', type=str, help='any experiment name')
parser.add_argument('--network', default='shared', type=str, help='shared, mtan, branch, transfer')
parser.add_argument('--backbone', default=None, type=str, help='deeplabv3, resnet, vgg, mlp')
parser.add_argument('--optim', default='sgd', type=str, help="name of optimisation algorithm: sgd, adam")
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--weight', default='equal', type=str, help='weighting methods: equal, dwa, uncert, autol')
parser.add_argument('--grad_method', default='none', type=str, help='graddrop, pcgrad, cagrad')
parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
parser.add_argument('--autol_init', default=0.1, type=float, help='initialisation for auto-lambda')
parser.add_argument('--autol_lr', default=1e-4, type=float, help='learning rate for auto-lambda')
parser.add_argument('--tasks', default='all', nargs="+", type=str, help='training tasks')
parser.add_argument('--pri_tasks', default=None, nargs="+", type=str, help='primary tasks, use all for MTL setting')
parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2, cityscapes')
parser.add_argument('--epoch', default=None, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=None, type=int, help='batch size')
parser.add_argument('--lr', default=None, type=float, help='learning rate')
parser.add_argument('--lr_scheduler', default='cosine', type=str, help='cosine, exponential')
parser.add_argument('--seed', default=0, type=int, help='random seed ID')
parser.add_argument('--freeze_bb', action='store_true', help='freeze backbone layers')
parser.add_argument('--load_weights', default=None, type=str, help='load pre-trained weights')
parser.add_argument('--arch_file', default=None, type=str, help='use defined branching architecture')
parser.add_argument('--lookahead_loss_test', action='store_true', help='perform lookahead loss test')
parser.add_argument('--normalise_omega', default=True, help='normlises omegas after each epoch to balance source task effects.')

opt = parser.parse_args()

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

# create logging folder to store training weights and losses
if not os.path.exists('logging'):
    os.makedirs('logging')

def save_model(save_path):
    torch.save({'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'sched_state_dict': scheduler.state_dict()}, 
                save_path)
    print(f"Model weights saved to {save_path}")

def load_model(load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if not opt.freeze_bb:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr

# Does a virtual training step based on the loss of the given task, then computes the loss of the other tasks
def lookahead_loss_test(model, tasks, train_data, train_target, val_data, val_target, test_data, test_target, test_metric, lr):

    is_train_mode = model.training
    current_params = deepcopy(model.state_dict())

    # Record lookahead losses for each task at each step [task_trained_on x task_being_evaluated x [train, val, test metrics]]
    lookahead_losses = np.full((len(tasks) + 1, len(tasks), 7), np.nan)

    # Record angles between tasks [task1 x task2 x (dot/angle)]
    task_affinities = np.full((len(tasks), len(tasks), 2), np.nan)

    # Get current task performances for comparison
    model.train()
    train_pred = model(train_data)
    train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(tasks)]
    lookahead_losses[0, :, 0] = [l.item() for l in train_loss]

    with torch.no_grad():
        model.eval()
        val_pred = model(val_data)
        val_loss = [compute_loss(val_pred[i], val_target[task_id], task_id) for i, task_id in enumerate(tasks)]
        lookahead_losses[0, :, 1] = [l.item() for l in val_loss]

        test_pred = model(test_data)
        test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(tasks)]
        test_metric.update_metric(test_pred, test_target, test_loss)
        test_metric.compute_metric()
        for task_eval_index, task_eval in enumerate(tasks):
            metric_index = 2  # Start after train/vali records
            for metric in test_metric.metric[task_eval]:
                if metric not in ["loss", "mean"]:
                    lookahead_losses[0, task_eval_index, metric_index] = test_metric.metric[task_eval][metric][0]
                    metric_index += 1
        test_metric.reset(reset_metrics=True)

    # Compute gradients for the network parameters given each task
    task_grads = []
    for t, task_id in enumerate(tasks):
        task_grads.append(torch.autograd.grad(train_loss[t], model.parameters(), retain_graph=True, allow_unused=True))

    # Calculate angle between gradients
    for task1_index, task1 in enumerate(tasks):
        for task2_index, task2 in enumerate(tasks):
            if task1_index > task2_index:
                grads_in_common = [i for i, p in enumerate(model.parameters()) if (task_grads[task1_index][i] is not None and task_grads[task2_index][i] is not None)]
                if len(grads_in_common) == 0:
                    print(f"WARNING: No grads in common between {task1} and {task2}.")
                else:
                    grads_task1 = torch.concat([task_grads[task1_index][g].flatten() for g in grads_in_common])
                    grads_task2 = torch.concat([task_grads[task2_index][g].flatten() for g in grads_in_common])
                    dot = torch.dot(grads_task1, grads_task2).item()
                    angle = torch.acos(dot / torch.norm(grads_task1) / torch.norm(grads_task2)).item()
                    task_affinities[task1_index, task2_index, 0] = dot
                    task_affinities[task1_index, task2_index, 1] = angle

    # Apply virtual step and compute loss for each task
    for task_index, task in enumerate(tasks):

        for param, grad in zip(model.parameters(), task_grads[task_index]):
            if grad is not None:
                param.data -= grad * lr

        # Training batch
        model.train()
        with torch.no_grad():
            train_pred = model(train_data)
            train_loss = [compute_loss(train_pred[i], train_target[task], task) for i, task in enumerate(tasks)]
            lookahead_losses[task_index + 1, :, 0] = [l.item() for l in train_loss]

            # Validation batch
            model.eval()
            val_pred = model(val_data)
            val_loss = [compute_loss(val_pred[i], val_target[task], task) for i, task in enumerate(tasks)]
            lookahead_losses[task_index + 1, :, 1] = [l.item() for l in val_loss]

            # Test set
            test_pred = model(test_data)
            test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(tasks)]
            test_metric.update_metric(test_pred, test_target, test_loss)
            test_metric.compute_metric()
            for task_eval_index, task_eval in enumerate(tasks):
                metric_index = 2 # Start after train/vali records
                for metric in test_metric.metric[task_eval]:
                    if metric not in ["loss", "mean"]:
                        lookahead_losses[task_index + 1, task_eval_index, metric_index] = test_metric.metric[task_eval][metric][0]
                        metric_index += 1
            test_metric.reset(reset_metrics=True)

        # Load original model state
        model.load_state_dict(current_params)
        
    if is_train_mode:
        model.train()

    return lookahead_losses, task_affinities



# define model, optimiser and scheduler
device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
train_tasks = create_task_flags(opt.tasks, opt.dataset)
has_aux_tasks = any([".aux" in t for t in train_tasks.keys()])
if has_aux_tasks:
    assert opt.network == "branch", "ERROR: Self-auxiliary tasks are only implemented with branch architecture."

if opt.pri_tasks is None:
    opt.pri_tasks = [t for t in train_tasks if ".aux" not in t]

pri_tasks = create_task_flags(opt.pri_tasks, opt.dataset)

train_tasks_str = ''.join(task.title() + ' + ' for task in train_tasks.keys())[:-3]
pri_tasks_str = ''.join(task.title() + ' + ' for task in pri_tasks.keys())[:-3]

print('Dataset: {} | Training Task: {} | Primary Task: {} in Multi-task / Auxiliary Learning Mode with {}'
      .format(opt.dataset.title(), train_tasks_str, pri_tasks_str, opt.network.upper()))
print('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {}'
      .format(opt.weight.title(), opt.grad_method.upper()))

if opt.backbone is None:
    if opt.dataset in ['nyuv2', 'cityscapes']: opt.backbone = 'deeplabv3'
    if opt.dataset in ['celeba']: opt.backbone = 'resnet'
    if opt.dataset in ['robotarm', 'global_chl']:
        opt.backbone = 'mlp'
        if opt.dataset == 'robotarm': d_features = 2
        if opt.dataset == 'global_chl': d_features = 16

if opt.network == 'shared':
    if opt.backbone == 'deeplabv3':
        model = MTLDeepLabv3(train_tasks).to(device)
    elif opt.backbone == 'resnet':
        model = MTLResNet(train_tasks).to(device)
    elif opt.backbone == 'mlp':
        model = MTLMLP(train_tasks, d_features=d_features).to(device)
    else:
        assert ValueError, "ERROR: {} backbone not implemented.".format(opt.backbone)

elif opt.network == 'finetune':
    assert len(train_tasks.keys()) == 1, "ERROR: Fine tune model can only have one target task."
    assert opt.finetune_load_filename is not None, "ERROR: Define load filename to fine tune network."
    print("WARNING: Using finetune layers.")
    load_filepath = os.path.join(opt.dataset, opt.finetune_load_filename)
    if opt.backbone == 'deeplabv3':
        model = FinetuneDeepLabv3(train_tasks, load_filepath, 2).to(device)
    elif opt.backbone == 'resnet':
        model = FinetuneResNet(train_tasks, load_filepath, 2).to(device)
    elif opt.backbone == 'mlp':
        model = FinetuneMLP(train_tasks, d_features=d_features, load_filename=load_filepath, transfer_function_nlayers=1).to(device)
    else:
        assert ValueError, "ERROR: {} backbone not implemented.".format(opt.backbone)

elif opt.network == 'mtan':
    model = MTANDeepLabv3(train_tasks).to(device)

elif opt.network == 'branch':
    if opt.arch_file is None:
        opt.arch_file = "arch_halfshared_{}{}".format(
            "regression" if opt.dataset in ["global_chl", "robotarm"] else opt.dataset,
            "_aux" if has_aux_tasks else ""
        )

    #arch = np.genfromtxt(f"load/{opt.arch_file}.csv", delimiter=',', dtype=int)
    arch = {}
    with open(os.path.join("load", f"{opt.arch_file}.csv"), newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            arch[row[0]] = [int(x) for x in row[1:]]
    if opt.backbone == 'deeplabv3':
        model = BranchDeepLabv3(train_tasks, arch).to(device)
    elif opt.backbone == 'resnet':
        model = BranchResNet(train_tasks, arch).to(device)
    elif opt.backbone == 'mlp':
        model = BranchMLP(train_tasks, arch, d_features).to(device)
    else:
        assert ValueError, "ERROR: {} backbone not implemented.".format(opt.backbone)

    unique_path_indices = {} # Dict of path: task_ids
    for i, task in enumerate(train_tasks):
        if model.arch_path_strs[task] in unique_path_indices:
            unique_path_indices[model.arch_path_strs[task]].append(i)
        else:
            unique_path_indices[model.arch_path_strs[task]] = [i]
    pri_task_auxs = {} # Dict of pri_task_id: aux_task_ids
    for i, pri_task in enumerate(pri_tasks):
        pri_task_auxs[pri_task] = [task for task in train_tasks if (task != pri_task) and (model.arch_path_strs[task] == model.arch_path_strs[pri_task])]

else:
    raise ValueError("ERROR: Invalid network type: {}".format(opt.network))

# Print model parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {n_params/1e6:.3f}m,")

# define dataset
if opt.dataset == 'nyuv2':
    dataset_path = 'dataset/nyuv2'
    train_set = NYUv2(root=dataset_path, partition="train", augmentation=True)
    vali_set = NYUv2(root=dataset_path, partition="vali")
    test_set = NYUv2(root=dataset_path, partition="test")
    batch_size = opt.batch_size if opt.batch_size is not None else 4
    opt.epoch = opt.epoch if opt.epoch is not None else 200
    opt.lr = opt.lr if opt.lr is not None else 0.1

elif opt.dataset == 'cityscapes':
    dataset_path = 'dataset/cityscapes'
    train_set = CityScapes(root=dataset_path, partition="train", augmentation=True)
    vali_set = CityScapes(root=dataset_path, partition="vali")
    test_set = CityScapes(root=dataset_path, partition="test")
    batch_size = opt.batch_size if opt.batch_size is not None else 4
    opt.epoch = opt.epoch if opt.epoch is not None else 200
    opt.lr = opt.lr if opt.lr is not None else 0.1

elif opt.dataset == 'celeba':
    dataset_path = 'dataset/celeba'
    train_set = CelebA(root=dataset_path, split="train", data_frac=0.1)
    vali_set = CelebA(root=dataset_path, split="vali", data_frac=0.1)
    test_set = CelebA(root=dataset_path, split="test", data_frac=0.1)
    batch_size = opt.batch_size if opt.batch_size is not None else 512
    opt.epoch = opt.epoch if opt.epoch is not None else 25#20
    opt.lr = opt.lr if opt.lr is not None else 0.1#001

elif opt.dataset == 'robotarm':
    dataset_path = 'dataset/robotarm'
    train_set = RobotArm(root=dataset_path, tasks=train_tasks, n_train_per_task=100, split="train")
    vali_set = RobotArm(root=dataset_path, tasks=train_tasks, n_train_per_task=100, split="vali")
    test_set = RobotArm(root=dataset_path, tasks=train_tasks, n_train_per_task=100, split="test")
    batch_size = opt.batch_size if opt.batch_size is not None else 50
    opt.epoch = opt.epoch if opt.epoch is not None else 1000
    opt.lr = opt.lr if opt.lr is not None else 0.001

elif 'global' in opt.dataset:
    dataset_path = 'dataset/global'
    label = opt.dataset.split('_')[-1]
    assert label in ['chl', 'tss', 'cdom'], f"ERROR: Unknown label in dataset {opt.dataset}."
    train_set = Gloria(root=dataset_path, tasks=train_tasks, split="train", label=label)
    vali_set = Gloria(root=dataset_path, tasks=train_tasks, split="vali", label=label)
    test_set = Gloria(root=dataset_path, tasks=train_tasks, split="test", label=label)
    batch_size = opt.batch_size if opt.batch_size is not None else 10
    opt.epoch = opt.epoch if opt.epoch is not None else 1000
    opt.lr = opt.lr if opt.lr is not None else 0.001

# Create mapping from existing tasks to the base task
def get_base_task(task):
    return task.split(".aux")[0]

# choose task weighting here
if opt.weight == 'uncert':
    logsigma = torch.tensor([-0.7] * len(train_tasks), requires_grad=True, device=device)
    params = list(model.parameters()) + [logsigma]
    logsigma_ls = np.zeros([opt.epoch, len(train_tasks)], dtype=np.float32)

if opt.weight in ['dwa', 'equal']:
    T = 2.0  # temperature used in dwa
    lambda_weight = np.ones([opt.epoch, len(train_tasks)])
    params = model.parameters()

if opt.weight == 'autol':
    params = model.parameters()
    autol = AutoLambda(model, device, train_tasks, pri_tasks, opt.autol_init)
    meta_weight_ls = np.zeros([opt.epoch, len(train_tasks)], dtype=np.float32)
    meta_optimizer = optim.Adam([autol.lambdas], lr=opt.autol_lr)


# load pre-trained weights and training scheme
if opt.load_weights is not None:
    load_model(f"logging/{opt.load_weights}.pth")

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    # Maintain order in tabular datasets for equal spread of tasks
    shuffle=False if opt.dataset in ['robotarm', 'global_chl'] else True,
    #num_workers=4
)

# Validation set loader that repeats forever - used for optimisation of hyperparameters in autol-lambda
if opt.weight == 'autol' or opt.lookahead_loss_test:
    vali_train_loader = torch.utils.data.DataLoader(
        dataset=vali_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        #num_workers=4
    )

vali_loader = torch.utils.data.DataLoader(
    dataset=vali_set,
    batch_size=batch_size,
    shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False
)
train_batch = len(train_loader)
vali_batch = len(vali_loader)
test_batch = len(test_loader)

if opt.optim == "sgd":
    optimizer = optim.SGD(params, lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)
elif opt.optim == "adam":
    optimizer = optim.Adam(params, lr=opt.lr, weight_decay=opt.weight_decay)#, momentum=0.9)
else:
    raise ValueError("ERROR: Invalid optimizer type: {}".format(opt.optim))

if opt.lr_scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epoch)
elif opt.lr_scheduler == 'exponential':
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
elif opt.lr_scheduler == 'none':
    scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1)


# apply gradient methods
if opt.grad_method != 'none':
    rng = np.random.default_rng()
    grad_dims = []
    for mm in model.shared_modules():
        for param in mm.parameters():
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), len(train_tasks)).to(device)

if opt.lookahead_loss_test:
    lookahead_loss_log = []
    task_affinities_log = []
    if opt.dataset == 'nyuv2':
        test_set_lookahead = NYUv2(root=dataset_path, partition="test", noise=opt.with_noise)
    elif opt.dataset == 'cityscapes':
        test_set_lookahead = CityScapes(root=dataset_path, partition="test", noise=opt.with_noise)
    elif opt.dataset == 'celeba':
        test_set_lookahead = CelebA(root=dataset_path, split="test", data_frac=0.1)  # , img_size=32)
    elif opt.dataset == 'robotarm':
        test_set_lookahead = RobotArm(root=dataset_path, tasks=train_tasks, n_train_per_task=100, split="test")
    elif opt.dataset == 'global_chl':
        test_set_lookahead = Gloria(root=dataset_path, tasks=train_tasks, split="test", label=label)
    test_loader_lookahead = torch.utils.data.DataLoader(
        dataset=test_set_lookahead,
        batch_size=batch_size,
        shuffle=True
    )
    test_metric_lookahead = TaskMetric(train_tasks, train_tasks, batch_size, opt.epoch, opt.dataset, include_mtl=True)

if any([".aux." in task for task in pri_tasks]):
    print("WARNING: Auxiliary task in primary tasks? {:}".format(pri_tasks))

# Train and evaluate multi-task network
t0 = datetime.datetime.now()
tic = time.time()
best_epoch = -1
train_metric = TaskMetric(train_tasks, train_tasks, batch_size, opt.epoch, opt.dataset, include_mtl=True)
vali_metric = TaskMetric(train_tasks, train_tasks, batch_size, opt.epoch, opt.dataset, include_mtl=True)
for index in range(opt.epoch):

    print('Epoch: {}/{}. Time elapsed: {:.1f} seconds'.format(index, opt.epoch, datetime.datetime.now().timestamp() - t0.timestamp()))
    for param_group in optimizer.param_groups:
        print(f"Effective LR = {param_group['lr']}")

    # apply Dynamic Weight Average
    if opt.weight == 'dwa':
        if index == 0 or index == 1:
            lambda_weight[index, :] = 1.0
        else:
            w = []
            for i, t in enumerate(train_tasks):
                w += [train_metric.metric[t]["loss"][index - 1] / train_metric.metric[t]["loss"][index - 2]]
                
            w = torch.softmax(torch.tensor(w) / T, dim=0)
            lambda_weight[index] = len(train_tasks) * w.numpy()

    # iteration for all batches
    model.train()
    train_dataset = iter(train_loader)
    if opt.weight == 'autol' or opt.lookahead_loss_test:
        vali_train_dataset = iter(vali_train_loader)
    if opt.lookahead_loss_test:
        test_dataset_lookahead = iter(test_loader_lookahead)

    for k in range(train_batch):

        if k % 50 == 0:
            print('Batch... {}/{}'.format(k, train_batch))
            if opt.weight == "autol":
                print({list(train_tasks.keys())[i]: autol.lambdas[i].item() for i in range(len(train_tasks))})

        try:
            train_data, train_target = train_dataset.next()
        except StopIteration:  # Reset train_loader if it reaches the end (due to olaux)
            train_dataset = iter(train_loader)
            train_data, train_target = train_dataset.next()
        train_data = train_data.to(device)

        train_target = {task_id: train_target[get_base_task(task_id)].to(device) for task_id in train_tasks.keys()}

        # get next validation batch if required
        if opt.weight == 'autol' or opt.lookahead_loss_test:
            try:
                vali_train_data, vali_train_target = vali_train_dataset.next()
            except StopIteration: # Reset vali_train_loader_iter if it reaches the end, since |validation set| < |training set|
                vali_train_dataset = iter(vali_train_loader)
                vali_train_data, vali_train_target = vali_train_dataset.next()
            vali_train_data = vali_train_data.to(device)
            vali_train_target = {task_id: vali_train_target[get_base_task(task_id)].to(device) for task_id in train_tasks.keys()}

        optimizer.zero_grad()

        if opt.lookahead_loss_test and k % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            if opt.dataset == "cityscapes": # Spagetti: this doesn't work with cityscapes for some reason so we just use vali twice
                print("WARNING: Spagetti, currently we are not computing test metrics in lookahead.")
                lookahead_loss, task_affinities = lookahead_loss_test(model, train_tasks, train_data, train_target,
                                                                      vali_train_data, vali_train_target,
                                                                      vali_train_data, vali_train_target,
                                                                      test_metric_lookahead, current_lr)
            else:
                try:
                    test_data, test_target = test_dataset_lookahead.next()
                except Exception as e:
                    test_dataset_lookahead = iter(test_loader_lookahead)
                    test_data, test_target = test_dataset_lookahead.next()
                test_data = test_data.to(device) #.unsqueeze(0)
                test_target = {task_id: test_target[get_base_task(task_id)].to(device) for task_id in train_tasks.keys()} #.unsqueeze(0)
                lookahead_loss, task_affinities = lookahead_loss_test(model, train_tasks, train_data, train_target,
                                                                  vali_train_data, vali_train_target, test_data,
                                                                  test_target, test_metric_lookahead, current_lr)
            lookahead_loss_log.append(lookahead_loss)
            task_affinities_log.append(task_affinities)

        # update meta-weights with Auto-Lambda
        if opt.weight == 'autol':
            meta_optimizer.zero_grad()
            autol.unrolled_backward(train_data, train_target, vali_train_data, vali_train_target, scheduler.get_last_lr()[0], optimizer)
            meta_optimizer.step()

        # update multi-task network parameters with task weights
        train_pred = model(train_data, verbose=(k == 0))
        train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

        train_loss_tmp = [0] * len(train_tasks)

        if opt.weight in ['equal', 'dwa']:
            weights = lambda_weight[index]
            if opt.normalise_omega and has_aux_tasks:
                w = np.empty_like(weights)
                for key in unique_path_indices:
                    inds = unique_path_indices[key]
                    w[inds] = np.mean(weights[inds]) / len(inds)
                weights = w
            train_loss_tmp = [w * train_loss[i] for i, w in enumerate(weights)]

        elif opt.weight == 'uncert':
            train_loss_tmp = [1 / (2 * torch.exp(w)) * train_loss[i] + w / 2 for i, w in enumerate(logsigma)]

        elif opt.weight == 'autol':
            weights = autol.lambdas
            if opt.normalise_omega and has_aux_tasks:
                w = torch.empty_like(weights)
                for key in unique_path_indices:
                    inds = unique_path_indices[key]
                    w[inds] = torch.mean(weights[inds]) / len(inds)
                weights = w
            train_loss_tmp = [w * train_loss[i] for i, w in enumerate(weights)]

        loss = sum(train_loss_tmp)

        if opt.grad_method == 'none':
            loss.backward()

        # gradient-based methods applied here:
        elif opt.grad_method == "graddrop":
            for i in range(len(train_tasks)):
                if not train_loss_tmp[i].grad_fn is None:
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(model, grads, grad_dims, i)
                    model.zero_grad_shared_modules()
            g = graddrop(grads)
            overwrite_grad(model, g, grad_dims, len(train_tasks))

        elif opt.grad_method == "pcgrad":
            for i in range(len(train_tasks)):
                if not train_loss_tmp[i].grad_fn is None:
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(model, grads, grad_dims, i)
                    model.zero_grad_shared_modules()
            g = pcgrad(grads, rng, len(train_tasks))
            overwrite_grad(model, g, grad_dims, len(train_tasks))

        elif opt.grad_method == "cagrad":
            for i in range(len(train_tasks)):
                if not train_loss_tmp[i].grad_fn is None:
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(model, grads, grad_dims, i)
                    model.zero_grad_shared_modules()
            g = cagrad(grads, len(train_tasks), 0.4, rescale=1)
            overwrite_grad(model, g, grad_dims, len(train_tasks))

        optimizer.step()

        train_metric.update_metric(train_pred, train_target, train_loss)
        
    train_str = train_metric.compute_metric()
    train_metric.reset()

    # evaluating vali data
    model.eval()
    with torch.no_grad():
        vali_dataset = iter(vali_loader)
        for k in range(vali_batch):
            vali_data, vali_target = vali_dataset.next()
            vali_data = vali_data.to(device)
            vali_target = {task_id: vali_target[get_base_task(task_id)].to(device) for task_id in train_tasks.keys()}

            vali_pred = model(vali_data)
            vali_loss = [compute_loss(vali_pred[i], vali_target[task_id], task_id) for i, task_id in enumerate(train_tasks) if vali_pred[i] is not None]
            
            vali_metric.update_metric(vali_pred, vali_target, vali_loss)
            
    vali_str = vali_metric.compute_metric()
    vali_metric.reset()

    scheduler.step()

    print('Epoch {:04d}\nTrain:{}vali:{}'.format(index, train_str, vali_str))

    if opt.weight == 'autol':
        meta_weight_ls[index] = autol.lambdas.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'vali_loss': vali_metric.metric,
                'weight': meta_weight_ls}

        print(get_weight_str(meta_weight_ls[index], train_tasks))

    if opt.weight in ['dwa', 'equal']:
        dict = {'train_loss': train_metric.metric, 'vali_loss': vali_metric.metric,
                'weight': lambda_weight}

        print(get_weight_str(lambda_weight[index], train_tasks))

    if opt.weight == 'uncert':
        logsigma_ls[index] = logsigma.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'vali_loss': vali_metric.metric,
                'weight': logsigma_ls}

        print(get_weight_str(1 / (2 * np.exp(logsigma_ls[index])), train_tasks))

    dict['train_tasks'] = train_tasks
    dict['pri_tasks'] = pri_tasks

    if opt.network == 'branch':
        dict['pri_task_auxs'] = pri_task_auxs

    if opt.lookahead_loss_test:
        dict['lookahead_loss'] = np.stack(lookahead_loss_log)
        dict['task_affinity'] = np.stack(task_affinities_log)

    # Filename for saving results and model
    filepath_results = os.path.join('logging', opt.dataset)
    filepath_weights = os.path.join('weights', opt.dataset)
    for fp in ["weights", "logging", filepath_results, filepath_weights]:
        if not os.path.exists(fp):
            os.mkdir(fp)
    fn_save = opt.name
    for action in parser._actions:
        arg_value = getattr(opt, action.dest, None)
        str_arg_value = re.sub(r'[\'\\/*?:"<>|]', "", str(arg_value))
        if arg_value != action.default and action.dest not in ["help", "name", "seed"] and len(str_arg_value) < 30:
            fn_save += f",{action.dest}={str_arg_value}"
    fn_save += ",{}_{}".format(opt.seed, t0.strftime('%Y%m%d_%H%M%S'))
    fn_save_results = os.path.join(filepath_results, fn_save + '.pkl')
    fn_save_weights = os.path.join(filepath_weights, fn_save + '.pth')

    # Determine if this is the optimal performance so far
    print("Best performance: {:.4f}".format(vali_metric.get_best_performance("all")))
    print("Current performance: {:.4f}".format(vali_metric.get_current_performance("all")))
    if vali_metric.get_best_performance("all") == vali_metric.get_current_performance("all"):
        save_model(fn_save_weights)
        best_epoch = index

    dict['best_epoch'] = best_epoch

# Test performance with optimal weights

load_model(fn_save_weights)
model.eval()
test_metric = TaskMetric(train_tasks, train_tasks, batch_size, opt.epoch, opt.dataset, include_mtl=True)
with torch.no_grad():
    test_dataset = iter(test_loader)
    for k in range(test_batch):
        test_data, test_target = test_dataset.next()
        test_data = test_data.to(device)#.unsqueeze(0)
        test_target = {task_id: test_target[get_base_task(task_id)].to(device) for task_id in train_tasks.keys()}#.unsqueeze(0)
        test_pred = model(test_data)
        test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
        test_metric.update_metric(test_pred, test_target, test_loss)
test_str = test_metric.compute_metric()
print('Test: {}'.format(test_str))
dict['test_loss'] = test_metric.metric
with open(fn_save_results, 'wb') as f:
    pickle.dump(dict, f)
