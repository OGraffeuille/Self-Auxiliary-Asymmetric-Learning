import copy
from utils import *


################################################################################################################
# This is a modified implementation of the Auto-lambda algorithm, with the following differences
#   - compatible with self-auxiliary tasks:
#   - added normalisation function to normalise the influence of primary tasks
################################################################################################################


class AutoLambda:
    def __init__(self, model, device, train_tasks, pri_tasks, weight_init=1):

        self.model = model
        self.model_ = copy.deepcopy(model)
        self.lambdas = torch.tensor([weight_init] * len(train_tasks), requires_grad=True, device=device)
        self.train_tasks = train_tasks
        self.pri_tasks = pri_tasks

    def virtual_step(self, train_x, train_y, alpha, model_optim):
        """
        Compute unrolled network theta' (virtual step)
        """

        # forward & compute loss
        if type(train_x) == list:  # multi-domain setting [many-to-many]
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:  # single-domain setting [one-to-many]
            train_pred = self.model(train_x)

        train_loss = self.model_fit(train_pred, train_y)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.lambdas)])

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)

        # do virtual step (update gradient): theta' = theta - alpha * sum_i lambda_i * L_i(f_theta(x_i), y_i)
        with torch.no_grad():
            for weight, weight_, grad in zip(self.model.parameters(), self.model_.parameters(), gradients):
                if grad is not None:  # Grad=None if the weight is not used in the computation graph
                    if 'momentum' in model_optim.param_groups[0].keys():  # used in SGD with momentum
                        m = model_optim.state[weight].get('momentum_buffer', 0.) * model_optim.param_groups[0]['momentum']
                    else:
                        m = 0
                    weight_.copy_(weight - alpha * (m + grad + model_optim.param_groups[0]['weight_decay'] * weight))

    def unrolled_backward(self, train_x, train_y, val_x, val_y, alpha, model_optim):
        """
        Compute un-rolled loss and backward its gradients
        """

        # do virtual step (calc theta`)
        self.virtual_step(train_x, train_y, alpha, model_optim)

        # define weighting for primary tasks (with binary weights)
        pri_weights = []
        for t in self.train_tasks:
            if t in self.pri_tasks:
                pri_weights += [1.0]
            else:
                pri_weights += [0.0]

        # compute validation data loss on primary tasks
        if type(val_x) == list:
            val_pred = [self.model_(x, t) for t, x in enumerate(val_x)]
        else:
            val_pred = self.model_(val_x)
        val_loss = self.model_fit(val_pred, val_y)
        loss = sum([w * val_loss[i] for i, w in enumerate(pri_weights)])

        # compute hessian via finite difference approximation
        model_weights_ = tuple(self.model_.parameters())
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)
        hessian = self.compute_hessian(d_model, train_x, train_y)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip([self.lambdas], hessian):
                mw.grad = - alpha * h


    def compute_hessian(self, d_model, train_x, train_y):
        norm = torch.cat([w.view(-1) for w in d_model if w is not None]).norm()
        eps = 0.01 / norm

        # \theta+ = \theta + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                if d is not None:
                    p += eps * d

        if type(train_x) == list:
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:
            train_pred = self.model(train_x)
        train_loss = self.model_fit(train_pred, train_y)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.lambdas)])
        d_weight_p = torch.autograd.grad(loss, self.lambdas)

        # \theta- = \theta - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                if d is not None:
                    p -= 2 * eps * d

        if type(train_x) == list:
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:
            train_pred = self.model(train_x)
        train_loss = self.model_fit(train_pred, train_y)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.lambdas)])
        d_weight_n = torch.autograd.grad(loss, self.lambdas)

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                if d is not None:
                    p += eps * d    

        hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian

    def model_fit(self, pred, targets):
        """
        define task specific losses
        """
        loss = [compute_loss(pred[i], targets[task_id], task_id) for i, task_id in enumerate(self.train_tasks)]
        return loss

    def normalise_lambdas(self, inds_to_normalise):
        normalise_factors = torch.empty_like(self.lambdas)
        for key in inds_to_normalise:
            inds = inds_to_normalise[key]
            normalise_factors[inds] = 1 / torch.mean(self.lambdas[inds]).detach()
        self.lambdas.data = self.lambdas.data * normalise_factors


