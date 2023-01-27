import os
import time
import torch
import numpy as np
from termcolor import cprint
from NLR_torch_iekf import TORCHIEKF
from NLR_utils import prepare_data, generate_normalize_u_p

#####
# set_normalize_u required for this version!
#####

max_loss = 2e1
max_grad_norm = 1e0
min_lr = 1e-5
criterion = torch.nn.MSELoss(reduction="sum")
lr_initprocesscov_net = 1e-4
weight_decay_initprocesscov_net = 0e-8
lr_mesnet = {'cov_net': 1e-4,
             'cov_lin': 1e-4,
             }
weight_decay_mesnet = {'cov_net': 1e-8,
                       'cov_lin': 1e-8,
                       }


def compute_delta_p(p):
    # sample at 1 Hz
    p = p[::100]
    delta_p = p[1:] - p[:-1]
    return delta_p


def train_filter(args):
    print("data loading")
    # prepare data
    t, pose0, p_gt, v0, u, i_t = prepare_data(for_torch=True)
    print("data loaded")
    # prepare iekf filter
    iekf = prepare_filter(args, u)
    print("iekf loaded")
    # get ground truth relative translation
    delta_p_gt = compute_delta_p(p_gt)
    print("delta_p_gt calculated")

    save_iekf(args, iekf)
    optimizer = set_optimizer(iekf)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loop(epoch, iekf, optimizer, t, pose0, v0, u, i_t, delta_p_gt)
        save_iekf(args, iekf)
        print("Amount of time spent for 1 epoch: {}s\n".format(int(time.time() - start_time)))
        start_time = time.time()


def prepare_filter(args, u):
    iekf = TORCHIEKF()

    # set dataset parameter
    iekf.filter_parameters = args.parameter_class()
    iekf.set_param_attr()
    if type(iekf.g).__module__ == np.__name__:
        iekf.g = torch.from_numpy(iekf.g).double()

    # load model
    if args.continue_training:
        iekf.load(args)
    iekf.train()
    # init u_loc and u_std
    generate_normalize_u_p(args, u)
    iekf.set_normalize_u(args)
    return iekf


def train_loop(epoch, iekf, optimizer, t, pose0, v0, u, i_t, delta_p_gt):
    loss_train = 0
    optimizer.zero_grad()

    loss = batch_step(iekf, delta_p_gt, t, pose0, v0, u, i_t)

    if loss == -1 or torch.isnan(loss):
        cprint("loss is invalid", 'yellow')
    # elif loss > max_loss:
    #     cprint("loss is too high {:.5f}".format(loss), 'yellow')
    else:
        loss_train += loss
        cprint("loss: {:.5f}".format(loss))

    if loss_train == 0:
        return
    loss_train.backward()
    # loss_train.cuda().backward()
    g_norm = torch.nn.utils.clip_grad_norm_(iekf.parameters(), max_grad_norm)
    if np.isnan(g_norm) or g_norm > 3 * max_grad_norm:
        cprint("gradient norm: {:.5f}".format(g_norm), 'yellow')
        optimizer.zero_grad()
    else:
        optimizer.step()
        optimizer.zero_grad()
        cprint("gradient norm: {:.5f}".format(g_norm))
    print('Train Epoch: {:2d} \tLoss: {:.5f}'.format(epoch, loss_train))
    return loss_train


def save_iekf(args, iekf):
    file_name = os.path.join(args.path_temp, "iekfnets.p")
    torch.save(iekf.state_dict(), file_name)
    print("The IEKF nets are saved in the file " + file_name)


def batch_step(iekf, delta_p_gt, t, pose0, v0, u, i_t):
    iekf.set_Q()
    measurements_covs = iekf.forward_nets(u)
    Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(t, u, measurements_covs, v0, t.shape[0], pose0)
    delta_p = compute_delta_p(p[i_t])
    distance = delta_p_gt.norm(dim=1).unsqueeze(-1)
    delta_p = delta_p.double() / distance.double()
    delta_p_gt = delta_p_gt.double() / distance.double()
    if delta_p is None:
        return -1
    loss = criterion(delta_p, delta_p_gt)
    return loss


def set_optimizer(iekf):
    param_list = [{'params': iekf.initprocesscov_net.parameters(),
                   'lr': lr_initprocesscov_net,
                   'weight_decay': weight_decay_initprocesscov_net}]
    for key, value in lr_mesnet.items():
        param_list.append({'params': getattr(iekf.mes_net, key).parameters(),
                           'lr': value,
                           'weight_decay': weight_decay_mesnet[key]
                           })
    optimizer = torch.optim.Adam(param_list)
    return optimizer


## TODO: add noise to u
def prepare_data_filter(dataset, dataset_name, Ns, iekf, seq_dim):
    # get data with trainable instant
    t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)
    t = t[Ns[0]: Ns[1]]
    ang_gt = ang_gt[Ns[0]: Ns[1]]
    p_gt = p_gt[Ns[0]: Ns[1]] - p_gt[Ns[0]]
    v_gt = v_gt[Ns[0]: Ns[1]]
    u = u[Ns[0]: Ns[1]]

    # subsample data
    N0, N = get_start_and_end(seq_dim, u)
    t = t[N0: N].double()
    ang_gt = ang_gt[N0: N].double()
    p_gt = (p_gt[N0: N] - p_gt[N0]).double()
    v_gt = v_gt[N0: N].double()
    u = u[N0: N].double()

    # TODO
    # # add noise
    # if iekf.mes_net.training:
    #     u = dataset.add_noise(u)

    return t, ang_gt, p_gt, v_gt, u, N0


def get_start_and_end(seq_dim, u):
    if seq_dim is None:
        N0 = 0
        N = u.shape[0]
    else:  # training sequence
        N0 = 10 * int(np.random.randint(0, (u.shape[0] - seq_dim) / 10))
        N = N0 + seq_dim
    return N0, N


## TODO: useless function. to be removed
def precompute_lost(Rot, p, list_rpe, N0):
    N = p.shape[0]
    Rot_10_Hz = Rot[::10]
    p_10_Hz = p[::10]
    idxs_0 = torch.Tensor(list_rpe[0]).clone().long() - int(N0 / 10)
    idxs_end = torch.Tensor(list_rpe[1]).clone().long() - int(N0 / 10)
    delta_p_gt = list_rpe[2]
    idxs = torch.Tensor(idxs_0.shape[0]).byte()
    idxs[:] = 1
    idxs[idxs_0 < 0] = 0
    idxs[idxs_end >= int(N / 10)] = 0
    delta_p_gt = delta_p_gt[idxs]
    idxs_end_bis = idxs_end[idxs]
    idxs_0_bis = idxs_0[idxs]
    if len(idxs_0_bis) == 0:
        return None, None
    else:
        delta_p = Rot_10_Hz[idxs_0_bis].transpose(-1, -2).matmul(
            (p_10_Hz[idxs_end_bis] - p_10_Hz[idxs_0_bis]).unsqueeze(-1)).squeeze()
        distance = delta_p_gt.norm(dim=1).unsqueeze(-1)
        return delta_p.double() / distance.double(), delta_p_gt.double() / distance.double()
