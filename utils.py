from copy import deepcopy
from scipy.optimize import minimize

import torch
import torch.nn.functional as F
import numpy as np


"""
Define task metrics, loss functions and model trainer here.
"""


class ConfMatrix(object):
    """
    For mIoU and other pixel-level classification tasks.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def reset(self):
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).item()


def create_task_flags(tasks, dataset):
    """
    Record task and its prediction dimension.
    Noise prediction is only applied in auxiliary learning.
    """
    if dataset == "nyuv2":
        task_dim_list = {'seg': 13, 'depth': 1, 'normal': 3}
    elif dataset == "cityscapes":
        task_dim_list = {'seg': 19, 'part_seg': 10, 'disp': 1}
    elif dataset == "celeba":
        task_dim_list = {f'class_{i:d}': 2 for i in range(9)}
    elif dataset == "robotarm":
        task_dim_list = {f'regression_{i:d}': 3 for i in range(10)}
    elif dataset == "global_chl":
        task_dim_list = {f'regression_{i:d}': 1 for i in range(10)}
    else:
        assert ValueError, "ERROR: Unknown dataset {}.".format(dataset)

    if tasks == 'all_aux_positive' or any(['all_aux_positive' in t for t in tasks]): # Use tasks which were found to provide positive results empirically; used for SAAL-e and SAAL-ew
        positive_tasks = {
            "nyuv2": ["seg", "depth", "normal.aux1", "normal.aux2"],
            "cityscapes": ["seg", "part_seg", "disp", "part_seg.aux1", "part_seg.aux2", "seg.aux2"],
            "celeba": ["class_0", "class_1", "class_2", "class_3", "class_4", "class_5", "class_6", "class_7", "class_8", "class_8.aux2", "class_8.aux4", "class_8.aux6", "class_8.aux7", "class_7.aux2", "class_7.aux3", "class_7.aux4", "class_7.aux5", "class_7.aux7", "class_6.aux4", "class_6.aux5", "class_5.aux5", "class_5.aux6", "class_5.aux8", "class_4.aux2", "class_4.aux5", "class_4.aux6", "class_3.aux2", "class_3.aux3", "class_3.aux5", "class_3.aux6", "class_3.aux7", "class_3.aux8", "class_2.aux2", "class_2.aux6", "class_2.aux7", "class_2.aux8", "class_1.aux3", "class_1.aux4", "class_1.aux5", "class_1.aux7", "class_0.aux1", "class_0.aux2", "class_0.aux3", "class_0.aux4", "class_0.aux5"],
            "robotarm": ["regression_0", "regression_0.aux1", "regression_0.aux3", "regression_0.aux5", "regression_0.aux6", "regression_0.aux8", "regression_1", "regression_1.aux2", "regression_1.aux4", "regression_2", "regression_2.aux1", "regression_2.aux3", "regression_2.aux4", "regression_2.aux9", "regression_3", "regression_3.aux6", "regression_3.aux8", "regression_4", "regression_4.aux1", "regression_4.aux4", "regression_4.aux5", "regression_4.aux7", "regression_4.aux9", "regression_5", "regression_5.aux6", "regression_5.aux8", "regression_6", "regression_6.aux2", "regression_6.aux3", "regression_6.aux5", "regression_6.aux7", "regression_6.aux9", "regression_7", "regression_7.aux4", "regression_7.aux6", "regression_7.aux8", "regression_7.aux9", "regression_8", "regression_8.aux3", "regression_8.aux5", "regression_8.aux7", "regression_9", "regression_9.aux2", "regression_9.aux6"],
            "global_chl": ["regression_0", "regression_0.aux1", "regression_0.aux8", "regression_0.aux9", "regression_1", "regression_1.aux1", "regression_1.aux4", "regression_1.aux5", "regression_1.aux6", "regression_1.aux7", "regression_1.aux8", "regression_2", "regression_2.aux3", "regression_2.aux5", "regression_2.aux6", "regression_2.aux7", "regression_2.aux9", "regression_3", "regression_3.aux1", "regression_3.aux2", "regression_3.aux4", "regression_3.aux5", "regression_3.aux6", "regression_3.aux8", "regression_4", "regression_4.aux1", "regression_4.aux2", "regression_4.aux3", "regression_4.aux4", "regression_4.aux5", "regression_4.aux9", "regression_5", "regression_5.aux1", "regression_5.aux2", "regression_5.aux3", "regression_5.aux4", "regression_5.aux6", "regression_5.aux8", "regression_5.aux9", "regression_6", "regression_6.aux1", "regression_6.aux2", "regression_6.aux3", "regression_6.aux5", "regression_6.aux6", "regression_6.aux7", "regression_6.aux8", "regression_6.aux9", "regression_7", "regression_7.aux1", "regression_7.aux2", "regression_7.aux3", "regression_7.aux5", "regression_7.aux6", "regression_7.aux8", "regression_8", "regression_8.aux1", "regression_8.aux3", "regression_8.aux6", "regression_8.aux7", "regression_8.aux8", "regression_8.aux9", "regression_9", "regression_9.aux1", "regression_9.aux3", "regression_9.aux4", "regression_9.aux6", "regression_9.aux8", "regression_9.aux9"]
        }
        tasks = positive_tasks[dataset]        
    elif tasks == 'all_aux' or any(['all_aux' in t for t in tasks]):
        ts = list(task_dim_list.keys())
        tasks = [t + a for t in ts for a in [""] + [f".aux{i}" for i in range(1, len(ts))]]
    elif tasks == 'all' or any(['all' in t for t in tasks]):
        tasks = list(task_dim_list.keys())
    if type(tasks) == str:
        tasks = [tasks]

    task_flags = {}
    for task in tasks:
        task_base = task.split(".aux")[0]
        task_flags[task] = task_dim_list[task_base]

    return task_flags


def get_weight_str(weight, tasks):
    """
    Record task weighting.
    """
    weight_str = 'Task Weighting | '
    for i, task_id in enumerate(tasks):
        weight_str += '{} {:.04f} '.format(task_id.title(), weight[i])
    return weight_str


def get_weight_str_ranked(weight, tasks, rank_num):
    """
    Record top-k ranked task weighting.
    """
    rank_idx = np.argsort(weight)

    if type(tasks) == dict:
        tasks = list(tasks.keys())

    top_str = 'Top {}: '.format(rank_num)
    bot_str = 'Bottom {}: '.format(rank_num)
    for i in range(rank_num):
        top_str += '{} {:.02f} '.format(tasks[rank_idx[-i-1]].title(), weight[rank_idx[-i-1]])
        bot_str += '{} {:.02f} '.format(tasks[rank_idx[i]].title(), weight[rank_idx[i]])

    return 'Task Weighting | {}| {}'.format(top_str, bot_str)


def compute_loss(pred, gt, task_id):
    """
    Compute task-specific loss.
    """

    if pred is None:
        return torch.tensor(0.)

    # Remove elements with invalid labels (e.g. when data doesn't have a label for each task)
    nan_gt = torch.isnan(gt)
    while nan_gt.dim() > 1:
        nan_gt = nan_gt.all(dim=-1)
    gt = gt[~nan_gt]
    pred = pred[~nan_gt]

    task_type = task_id.split(".aux")[0]
    if task_type in ['seg', 'part_seg'] or 'class' in task_id:
        # Cross Entropy Loss with Ignored Index (values are -1)
        loss = F.cross_entropy(pred, gt, ignore_index=-1)

    # Note: Some other papers use dot product for normal loss
    elif task_type in ['normal', 'depth', 'disp', 'noise']:
        # L1 Loss with Ignored Region (values are 0 or -1)
        invalid_idx = -1 if task_id == 'disp' else 0
        valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
        loss = torch.sum(F.l1_loss(pred, gt, reduction='none').masked_select(valid_mask)) \
                / torch.nonzero(valid_mask, as_tuple=False).size(0)

    elif "regression" in task_type:

        if pred.shape[0] == 0:
            return torch.tensor(0.)
        loss = F.mse_loss(pred, gt)

    else:
        raise ValueError("ERROR: Unknown task type {}.".format(task_id))
    
    return loss


class TaskMetric:
    def __init__(self, train_tasks, pri_tasks, batch_size, epochs, dataset, include_mtl=False):
        self.train_tasks = train_tasks
        self.pri_tasks = pri_tasks
        self.batch_size = batch_size
        self.dataset = dataset
        self.include_mtl = include_mtl
        self.metric = {}#key: {} for key in train_tasks.keys()} #np.zeros([epochs, 2]) for key in train_tasks.keys()}  # record loss & task-specific metric
        self.data_counter = {task_id: 0 for task_id in self.train_tasks}
        self.epoch_counter = 0
        self.conf_mtx = {}

        task_metrics = {
            'seg':      ['mIoU', 'acc'], 
            'part_seg': ['mIoU', 'acc'],
            'depth':    ['abs_err', 'rel_err'],
            'disp':     ['abs_err'],#, 'rel_err'],
            'normal':   ['mean_err', 'median_err', 'within_1125', 'within_225', 'within_30'],
        }
        task_metrics.update({f'class_{i:d}': ['acc'] for i in range(40)})
        task_metrics.update({f'regression_{i:d}': ['rmse'] for i in range(100)})

        for task in self.train_tasks:
            task_type = task.split(".aux")[0]
            if task_type in ['seg', 'part_seg']:
                self.conf_mtx[task] = ConfMatrix(self.train_tasks[task])

            metrics_to_include = ['loss'] + task_metrics[task_type] + (['mean'] if include_mtl else [])
            self.metric[task] = {key: np.zeros((epochs,)) for key in metrics_to_include}

        if include_mtl:  # include multi-task performance (relative averaged task improvement)
            self.metric['all'] = {"mean": np.zeros((epochs,))}

    def reset(self, reset_metrics=False):
        """
        Reset data counter and confusion matrices.
        """

        # Should only be used in analysis e.g. lookahead loss tests - resets metrics entirely
        if not reset_metrics:
            self.epoch_counter += 1
        else:
            for task in self.metric:
                for metric in self.metric[task]:
                    self.metric[task][metric] = np.zeros_like(self.metric[task][metric])
        for task_id in self.data_counter:
            self.data_counter[task_id] = 0
        if len(self.conf_mtx) > 0:
            for i in self.conf_mtx:
                self.conf_mtx[i].reset()



    # Gradually computes average losses and metrics over batches to avoid computing in one
    def update_metric(self, task_pred, task_gt, task_loss):
        """
        Update batch-wise metric for each task.
            :param task_pred: [TASK_PRED1, TASK_PRED2, ...]
            :param task_gt: {'TASK_ID1': TASK_GT1, 'TASK_ID2': TASK_GT2, ...}
            :param task_loss: [TASK_LOSS1, TASK_LOSS2, ...]
        """

        #curr_bs = task_pred[0].shape[0]
        #r = self.data_counter / (self.data_counter + curr_bs / self.batch_size)
        e = self.epoch_counter

        with (torch.no_grad()):
            for loss, pred, (task_id, gt) in zip(task_loss, task_pred, task_gt.items()):

                pred = pred.detach()

                task_type = task_id.split(".aux")[0]

                # Remove elements with invalid labels (e.g. when data doesn't have a label for each task)
                nan_gt = torch.isnan(gt)
                while nan_gt.dim() > 1:
                    nan_gt = nan_gt.all(dim=-1)
                gt = gt[~nan_gt]
                pred = pred[~nan_gt]

                curr_bs = pred.shape[0]
                if curr_bs == 0:
                    continue

                r = self.data_counter[task_id] / (self.data_counter[task_id] + curr_bs)
                self.data_counter[task_id] += curr_bs
                
                # All tasks have a loss function to compute
                self.metric[task_id]['loss'][e] = r * self.metric[task_id]['loss'][e] + (1 - r) * loss.item()

                # Compute task-specific metrics
                if task_type in ['seg', 'part_seg']:
                    acc = torch.mean((pred.argmax(1) == gt).float()).item()
                    self.metric[task_id]["acc"][e] = r * self.metric[task_id]["acc"][e] + (1 - r) * acc 
                    # Update confusion matrix (mIoU will be computed directly in the Confusion Matrix)
                    self.conf_mtx[task_id].update(pred.argmax(1).flatten(), gt.flatten())

                if 'class' in task_type:
                    pred_label = pred.data.max(1)[1]
                    acc = pred_label.eq(gt).sum().item() / pred_label.shape[0]
                    self.metric[task_id]['acc'][e] = r * self.metric[task_id]['acc'][e] + (1 - r) * acc

                if 'regression' in task_type:
                    rmse = torch.sqrt(torch.mean(torch.pow(pred - gt, 2)))
                    self.metric[task_id]['rmse'][e] = r * self.metric[task_id]['rmse'][e] + (1 - r) * rmse

                if task_type in ['depth', 'disp', 'noise']:

                    invalid_idx = -1 if task_id == 'disp' else 0
                    valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
                    abs_err = torch.abs(pred - gt)
                    rel_err = torch.abs(pred - gt) / (gt + 1e-6)
                    abs_err = torch.mean(abs_err.masked_select(valid_mask)).item()
                    rel_err = torch.mean(rel_err.masked_select(valid_mask)).item()
                    self.metric[task_id]['abs_err'][e] = r * self.metric[task_id]['abs_err'][e] + (1 - r) * abs_err
                    if task_type != "disp":
                        self.metric[task_id]['rel_err'][e] = r * self.metric[task_id]['rel_err'][e] + (1 - r) * rel_err

                if task_type in ['normal']:
                    valid_mask = (torch.sum(gt, dim=1) != 0).to(pred.device)
                    degree_error = torch.acos(torch.clamp(torch.sum(pred * gt, dim=1).masked_select(valid_mask), -1, 1))
                    err = torch.rad2deg(degree_error)
                    self.metric[task_id]['mean_err'][e]    = r * self.metric[task_id]['mean_err'][e]    + (1 - r) * torch.mean(err).item() / 100
                    self.metric[task_id]['median_err'][e]  = r * self.metric[task_id]['median_err'][e]  + (1 - r) * torch.median(err).item() / 100
                    self.metric[task_id]['within_1125'][e] = r * self.metric[task_id]['within_1125'][e] + (1 - r) * torch.mean((err < 11.25).float()).item() 
                    self.metric[task_id]['within_225'][e]  = r * self.metric[task_id]['within_225'][e]  + (1 - r) * torch.mean((err < 22.5).float()).item() 
                    self.metric[task_id]['within_30'][e]   = r * self.metric[task_id]['within_30'][e]   + (1 - r) * torch.mean((err < 30).float()).item()

    # Needs to be called after update_metric because it computes mIoU from the confusion matrix
    def compute_metric(self, only_pri=False):
        metric_str = ''
        e = self.epoch_counter
        tasks = self.pri_tasks if only_pri else self.train_tasks  # only print primary tasks performance in evaluation

        # Compute metric for image segmentation metrics (mIoU) from confusion matrix
        for task_id in tasks:
            task_type = task_id.split(".aux")[0]
            if task_type in ['seg', 'part_seg']:
                self.metric[task_id]['mIoU'][e] = self.conf_mtx[task_id].get_metrics()
            metric_str += " |"+  task_id  + ''.join([' {} {:.4f}'.format(m, self.metric[task_id][m][e]) for m in self.metric[task_id]])

        if self.include_mtl:
                        # Pre-computed single task learning performance using trainer_dense_single.py
            if self.dataset == 'nyuv2':
                stl = {'seg': {'acc': 0.578818, 'mIoU': 0.392830}, 
                       'depth': {'abs_err': 0.653261, 'rel_err': 0.275089}, 
                       'normal': {'mean_err': 0.223930, 'median_err': 0.153261, 'within_1125': 0.408927, 'within_225': 0.659008, 'within_30': 0.747722}} 
            elif self.dataset == 'cityscapes':
                stl = {'seg': {'acc': 0.8433, 'mIoU': 0.5423},
                       'part_seg': {'acc': 0.9768, 'mIoU': 0.5292},
                       #'depth': {'abs_err': 0., 'rel_err': 0.},
                       'disp': {'abs_err': 0.6969}}#, 'rel_err': 0.}}
            elif self.dataset == 'cifar100':
                stl = {'class_0': {'acc': 0.6865}, 'class_1': {'acc': 0.8100}, 'class_2': {'acc': 0.8234}, 'class_3': {'acc': 0.8371}, 'class_4': {'acc': 0.8910},
                       'class_5': {'acc': 0.8872}, 'class_6': {'acc': 0.8475}, 'class_7': {'acc': 0.8588}, 'class_8': {'acc': 0.8707}, 'class_9': {'acc': 0.9015},
                       'class_10': {'acc': 0.8976}, 'class_11': {'acc': 0.8488}, 'class_12': {'acc': 0.9033}, 'class_13': {'acc': 0.8441}, 'class_14': {'acc': 0.5537},
                       'class_15': {'acc': 0.7584}, 'class_16': {'acc': 0.7279}, 'class_17': {'acc': 0.7537}, 'class_18': {'acc': 0.9148}, 'class_19': {'acc': 0.9469}}
            elif self.dataset == 'celeba':
                accs = [0.9296, 0.7972, 0.8154, 0.8337, 0.9794, 0.9500, 0.6962, 0.8233, 0.8843, 0.9534,
                        0.9533, 0.8767, 0.9120, 0.9496, 0.9582, 0.9840, 0.9648, 0.9745, 0.9034, 0.8610,
                        0.9655, 0.9205, 0.9610, 0.8540, 0.9493, 0.7421, 0.9666, 0.7466, 0.9232, 0.9433,
                        0.9713, 0.9145, 0.8007, 0.7991, 0.8463, 0.9820, 0.9306, 0.8625, 0.9406, 0.8666] # Actually MTL (fullsize, 0.0001 lr) performance
                stl = {f'class_{i:d}': {'acc': accs[i]} for i in range(9)}
            elif self.dataset == 'robotarm':
                rmses = [0.6565, 0.6997, 0.6991, 0.7396, 0.6705, 0.8119, 0.6753, 0.6654, 0.7327, 0.6673]
                stl = {f'regression_{i:d}': {'rmse': rmses[i]} for i in range(10)}
            elif self.dataset == 'global_chl':
                rmses = [0.492, 1.5333, 0.3873, 0.1528, 0.7766, 0.1473, 0.4462, 0.6553, 0.1631, 0.6054]
                stl = {f'regression_{i:d}': {'rmse': rmses[i]} for i in range(10)}
            else:
                raise ValueError("ERROR: Base STL performances for dataset {} unknown.".format(self.dataset))

            # Compute improvement across tasks and metrics. Delta performance is described in Recon paper, but seems standard
            delta_tasks = []
            for task_id in self.train_tasks:
                task_type = task_id.split(".aux")[0]

                delta_task = []
                for m in self.metric[task_id]:
                    if m in ['mIoU', 'acc', 'within_1125', 'within_225', 'within_30']:
                        delta_task.append((self.metric[task_id][m][e] - stl[task_type][m]) / stl[task_type][m])
                    elif m in ['abs_err', 'rel_err', 'mean_err', 'median_err', 'rmse']:
                        delta_task.append(-(self.metric[task_id][m][e] - stl[task_type][m]) / stl[task_type][m])
                    else:
                        assert m in ['loss', 'mean'], "ERROR: Unknown metric type {}.".format(m)
                delta_task = sum(delta_task) / len(delta_task)
                self.metric[task_id]['mean'][e] = delta_task

                delta_tasks.append(delta_task)

            self.metric['all']['mean'][e] = sum(delta_tasks) / len(delta_tasks)
            metric_str += ' all {:.4f}'.format(self.metric['all']['mean'][e])
            
        return metric_str

    def get_best_performance(self, task):
        e = self.epoch_counter
        if self.include_mtl:
            return max(self.metric[task]['mean'][:e])
        else:
            if task in ['class']:
                return max(self.metric[task]['acc'][:e])
            elif task in ['seg', 'part_seg']: 
                return max(self.metric[task]['mIoU'][:e])
            elif task in ['depth', 'disp']: 
                return min(self.metric[task]['abs_err'][:e])
            elif task in ['normal']:
                return min(self.metric[task]['mean_err'][:e])
            elif task in ['normal_', 'normal__']:
                return min(self.metric[task]['mean_err'][:e])
            else:
                assert False, "ERROR: Unknown task type {}.".format(task)

    def get_current_performance(self, task):
        e = self.epoch_counter
        if self.include_mtl:
            return self.metric[task]['mean'][e-1]
        else:
            if task in ['class']:
                return self.metric[task]['acc'][e-1]
            elif task in ['seg', 'part_seg']: 
                return self.metric[task]['mIoU'][e-1]
            elif task in ['depth', 'disp']: 
                return self.metric[task]['abs_err'][e-1]
            elif task in ['normal']:
                return self.metric[task]['mean_err'][e-1]
            elif task in ['normal_', 'normal__']:
                return self.metric[task]['mean_err'][e-1]
            else:
                assert False, "ERROR: Unknown task type {}.".format(task)


"""
Define Gradient-based frameworks here. 
Based on https://github.com/Cranial-XIX/CAGrad/blob/main/cityscapes/utils.py
"""


def graddrop(grads):
    P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1) + 1e-8))
    U = torch.rand_like(grads[:, 0])
    M = P.gt(U).view(-1, 1) * grads.gt(0) + P.lt(U).view(-1, 1) * grads.lt(0)
    g = (grads * M.float()).mean(1)
    return g


def pcgrad(grads, rng, num_tasks):
    grad_vec = grads.t()

    shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
    for i in range(num_tasks):
        task_indices = np.arange(num_tasks)
        task_indices[i] = task_indices[-1]
        shuffled_task_indices[i] = task_indices[:-1]
        rng.shuffle(shuffled_task_indices[i])
    shuffled_task_indices = shuffled_task_indices.T

    normalized_grad_vec = grad_vec / (grad_vec.norm(dim=1, keepdim=True) + 1e-8)  # num_tasks x dim
    modified_grad_vec = deepcopy(grad_vec)
    for task_indices in shuffled_task_indices:
        normalized_shuffled_grad = normalized_grad_vec[task_indices]  # num_tasks x dim
        dot = (modified_grad_vec * normalized_shuffled_grad).sum(dim=1, keepdim=True)   # num_tasks x dim
        modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
    g = modified_grad_vec.mean(dim=0)
    return g


def cagrad(grads, num_tasks, alpha=0.5, rescale=1):
    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(num_tasks) / num_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (x.reshape(1, num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + c * np.sqrt(
            x.reshape(1, num_tasks).dot(A).dot(x.reshape(num_tasks, 1)) + 1e-8)).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha ** 2)
    else:
        return g / (1 + alpha)

def grad2vec(m, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    for mm in m.shared_modules():
        for p in mm.parameters():
            grad = p.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

def overwrite_grad(m, newgrad, grad_dims, num_tasks):
    newgrad = newgrad * num_tasks  # to match the sum loss
    cnt = 0
    for mm in m.shared_modules():
        for param in mm.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1
