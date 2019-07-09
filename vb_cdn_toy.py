import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists
import numpy as np
import input_data
import argparse
from tqdm import tqdm
import time
import math
import os


parser = argparse.ArgumentParser()
parser.add_argument('--n_data', type=int, default=20)
parser.add_argument('--n_points', type=int, default=1000)
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_hidden', type=int, default=100)
parser.add_argument('--n_hidden_hypernet', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--randseed', type=int, default=12)
parser.add_argument('--dataset', type=str, default='cubic')
parser.add_argument('--train_samples', type=int, default=3)
parser.add_argument('--n_iter', type=int, default=10000)
parser.add_argument('--load', default=False, action='store_true')
parser.add_argument('--hetero', default=False, action='store_true')
args = parser.parse_args()

name = 'vbcdn'

os.makedirs('./results/toy/{args.dataset}', exist_ok=True)
os.makedirs('./models/toy/{args.dataset}', exist_ok=True)

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.cuda.manual_seed(args.randseed)

m = args.n_data

if args.dataset == 'cubic':
    # Toy data from (Hernandez-Lobato & Adams, 2015)
    X_train = -4 + 8*torch.rand(m, 1).cuda()  # x ~ Unif(-4, 4)

    if not args.hetero:
        y_train = X_train**3 + 3*torch.randn(m, 1).cuda()  # y = x^3 + eps; eps ~ N(0, 9)
    else:
        y_train = X_train**3
        for i, x in enumerate(X_train):
            if x >= 0:
                y_train[i] += 3*torch.randn(1).cuda()
            else:
                y_train[i] += 15*torch.randn(1).cuda()


class VMGLinear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(VMGLinear, self).__init__()

        in_dim += 1
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.mu = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(in_dim, out_dim)))
        self.logvar_in = nn.Parameter(nn.init.normal_(torch.zeros(in_dim), -3, 1e-6))
        self.logvar_out = nn.Parameter(nn.init.normal_(torch.zeros(out_dim), -3, 1e-6))

    def forward(self, x, randseed=None):
        m = x.size(0)

        W, D_KL = self.sample(m, randseed)

        x = torch.cat([x, torch.ones(m, 1, device='cuda')], 1)
        h = torch.bmm(x.unsqueeze(1), W).squeeze(1)

        return h, D_KL

    def sample(self, m, randseed=None):
        M = self.mu
        logvar_r = self.logvar_in
        logvar_c = self.logvar_out

        # Reparametrization Trick
        var_r = torch.exp(logvar_r)
        var_c = torch.exp(logvar_c)

        if randseed is not None:
            # Sample E (thus \psi) deterministically using randseed, then restore the random seed and state
            torch.manual_seed(randseed)
            E = torch.randn(m, *M.shape, device='cuda')
            torch.manual_seed(np.random.randint(9999999))
        else:
            E = torch.randn(m, *M.shape, device='cuda')

        r, c = self.in_dim, self.out_dim

        # Reparametrization trick
        W = M + torch.sqrt(var_r).view(1, r, 1) * E * torch.sqrt(var_c).view(1, 1, c)

        # KL divergence to prior MVN(0, I, I)
        D_KL = 1/2 * (torch.sum(var_r)*torch.sum(var_c) \
                    + torch.norm(M)**2 \
                    - r*c - c*torch.sum(logvar_r) - r*torch.sum(logvar_c))

        return W, D_KL


class ProbHypernet(nn.Module):

    def __init__(self, in_dim, out_dim, h_dim=10):
        super(ProbHypernet, self).__init__()

        self.in_dim = in_dim + 1
        self.out_dim = out_dim
        self.h_dim = h_dim

        self.M = nn.Parameter(torch.randn(self.in_dim, out_dim))
        torch.nn.init.kaiming_normal_(self.M)

        self.fc_xh = VMGLinear(in_dim, h_dim)
        self.fc_hmu = VMGLinear(h_dim, self.in_dim)
        self.fc_hlogvar_in = VMGLinear(h_dim, self.in_dim)
        self.fc_hlogvar_out = VMGLinear(h_dim, self.out_dim)


    def forward(self, x, psi_randseed=None, output_weight_params=False):
        m = x.shape[0]
        r, c = self.in_dim, self.out_dim

        h, D_KL_hyp1 = self.fc_xh(x)
        h = F.tanh(h)
        mu_scaling, D_KL_hyp2 = self.fc_hmu(h, None if psi_randseed is None else psi_randseed+1)
        logvar_r, D_KL_hyp3 = self.fc_hlogvar_in(h, None if psi_randseed is None else psi_randseed+2)
        logvar_c, D_KL_hyp4 = self.fc_hlogvar_out(h, None if psi_randseed is None else psi_randseed+3)

        # KL-divergence of the Bayesian hypernet
        D_KL_psi = D_KL_hyp1 + D_KL_hyp2 + D_KL_hyp3 + D_KL_hyp4

        M = mu_scaling.view(m, r, 1) * self.M  # Broadcasted: M is (m, r, c)
        var_r = torch.exp(logvar_r)
        var_c = torch.exp(logvar_c)
        E = torch.randn(m, r, c, device='cuda')

        # Reparametrization trick
        W = M + torch.sqrt(var_r).view(m, r, 1) * E * torch.sqrt(var_c).view(m, 1, c)

        x = torch.cat([x, torch.ones(m, 1, device='cuda')], 1)
        h = torch.bmm(x.unsqueeze(1), W).squeeze()

        if output_weight_params:
            return h, D_KL_psi, (M, var_r, var_c)
        else:
            return h, D_KL_psi

    def sample_weights(self, m):
        w1 = self.fc_xh.sample(m)[0].view(m, -1)
        w2 = self.fc_hmu.sample(m)[0].view(m, -1)
        w3 = self.fc_hlogvar_in.sample(m)[0].view(m, -1)
        w4 = self.fc_hlogvar_out.sample(m)[0].view(m, -1)

        return torch.cat([w1, w2, w3, w4], dim=1)


class Model(nn.Module):

    def __init__(self, h_dim=100, h_dim_hypernet=10):
        super(Model, self).__init__()

        self.fc_xh = ProbHypernet(1, h_dim, h_dim_hypernet)
        self.fc_hy = ProbHypernet(h_dim, 1, h_dim_hypernet)

    def forward(self, X, psi_randseed=None):
        h, D_KL1 = self.fc_xh(X, psi_randseed)
        h = F.relu(h)

        y, D_KL2 = self.fc_hy(h, None if psi_randseed is None else psi_randseed+100)
        y = y.unsqueeze(1)

        return y, D_KL1+D_KL2

    def sample_weights(self, m):
        w1 = self.fc_xh.sample_weights(m)
        w2 = self.fc_xh.sample_weights(m)

        return torch.cat([w1, w2], dim=1)

    def get_weight_params_f(self, X):
        h, _, W1_params = self.fc_xh(X, output_weight_params=True)
        h = F.relu(h)
        _, _, W2_params = self.fc_hy(h, output_weight_params=True)
        return W1_params, W2_params


    def get_weight_params_g(self):
        post_vars_1_1 = np.kron(torch.exp(self.fc_xh.fc_xh.logvar_out).cpu().data.numpy(), torch.exp(self.fc_xh.fc_xh.logvar_in).cpu().data.numpy())
        post_vars_1_2 = np.kron(torch.exp(self.fc_xh.fc_hmu.logvar_out).cpu().data.numpy(), torch.exp(self.fc_xh.fc_hmu.logvar_in).cpu().data.numpy())
        post_vars_1_3 = np.kron(torch.exp(self.fc_xh.fc_hlogvar_in.logvar_out).cpu().data.numpy(), torch.exp(self.fc_xh.fc_hlogvar_in.logvar_in).cpu().data.numpy())
        post_vars_1_4 = np.kron(torch.exp(self.fc_xh.fc_hlogvar_out.logvar_out).cpu().data.numpy(), torch.exp(self.fc_xh.fc_hlogvar_out.logvar_in).cpu().data.numpy())

        post_vars_1 = np.concatenate([post_vars_1_1, post_vars_1_2, post_vars_1_3, post_vars_1_4])

        post_vars_2_1 = np.kron(torch.exp(self.fc_hy.fc_xh.logvar_out).cpu().data.numpy(), torch.exp(self.fc_hy.fc_xh.logvar_in).cpu().data.numpy())
        post_vars_2_2 = np.kron(torch.exp(self.fc_hy.fc_hmu.logvar_out).cpu().data.numpy(), torch.exp(self.fc_hy.fc_hmu.logvar_in).cpu().data.numpy())
        post_vars_2_3 = np.kron(torch.exp(self.fc_hy.fc_hlogvar_in.logvar_out).cpu().data.numpy(), torch.exp(self.fc_hy.fc_hlogvar_in.logvar_in).cpu().data.numpy())
        post_vars_2_4 = np.kron(torch.exp(self.fc_hy.fc_hlogvar_out.logvar_out).cpu().data.numpy(), torch.exp(self.fc_hy.fc_hlogvar_out.logvar_in).cpu().data.numpy())

        post_vars_2 = np.concatenate([post_vars_2_1, post_vars_2_2, post_vars_2_3, post_vars_2_4])

        return post_vars_1, post_vars_2


""" Training """
h_dim = args.n_hidden
h_dim_hypernet = args.n_hidden_hypernet
model = Model(h_dim, h_dim_hypernet).cuda()

print(f'Parameter count: {np.sum([value.numel() for value in model.parameters()])}')

S = args.train_samples
m = args.batch_size
lr = args.lr

opt = optim.Adam(model.parameters(), lr)

if args.load:
    model.load_state_dict(torch.load(f'models/toy/{args.dataset}/model_{name}_{h_dim}_{m}_{lr}_{args.n_data}_{S}.bin'))
else:
    for i in range(args.n_iter):
        p = dists.Normal(y_train, 1/math.sqrt(2)*torch.ones_like(y_train))

        log_p_y = []

        # Fix \psi
        # psi_randseed = np.random.randint(9999999)
        psi_randseed = None

        for _ in range(S):
            y_s, D_KL = model.forward(X_train, psi_randseed)
            log_p_y.append(p.log_prob(y_s))

        nll = -torch.mean(torch.logsumexp(torch.stack(log_p_y), 0) - math.log(S))
        loss = nll + 1/args.n_data*D_KL

        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        opt.step()
        opt.zero_grad()

        if i % 200 == 0:
            print('\n==========================================')
            print('Iter-{}'.format(i))
            print('==========================================')
            print('NLL: {:.3f}; D_KL: {:.3f}'.format(nll.data.item(), D_KL.data.item()))


# Test for whole domain
if not args.hetero:
    X_test = torch.linspace(-6, 6, args.n_points, device='cuda').unsqueeze(1)
else:
    X_test = torch.linspace(-4, 4, args.n_points, device='cuda').unsqueeze(1)

z = torch.cat([model.forward(X_test)[0] for _ in range(args.n_samples)], dim=1).t()
y = z.cpu().data.numpy()
np.save(f'results/toy/{args.dataset}/y_test_{name}_{h_dim}_{m}_{lr}_{args.n_data}_{S}.npy', y)

torch.save(model.state_dict(), f'models/toy/{args.dataset}/model_{name}_{h_dim}_{m}_{lr}_{args.n_data}_{S}.bin')
np.save(f'results/toy/{args.dataset}/X_train_{name}_{h_dim}_{m}_{lr}_{args.n_data}.npy', X_train.cpu().data.numpy())
np.save(f'results/toy/{args.dataset}/y_train_{name}_{h_dim}_{m}_{lr}_{args.n_data}.npy', y_train.cpu().data.numpy())


""" ============= Looking at the variance of the mixing distribution and the posterior ============= """

X_test = torch.linspace(-4, 4, args.n_points, device='cuda').unsqueeze(1)


def get_test_weight_params(X_test, m=args.batch_size, region='all'):
    if region == 'all':
        M = X_test.size(0)
        r = range(0, M, m)
    elif region == 'left':
        M = X_test.size(0) // 2
        r = range(0, M, m)
    elif region == 'right':
        M = X_test.size(0) // 2
        r = range(X_test.size(0)//2, X_test.size(0), m)
    else:
        raise Exception('Valid options: "left", "right", "all".')

    avg_var_1, avg_var_2 = 0, 0

    for i in r:
        W1_params_m, W2_params_m = model.get_weight_params_f(X_test[i:i+m])

        W1_vars_A = W1_params_m[1].cpu().data.numpy()
        W1_vars_B = W1_params_m[2].cpu().data.numpy()
        W1_vars = []

        for W1_var_A, W1_var_B in zip(W1_vars_A, W1_vars_B):
            var = np.kron(W1_var_B, W1_var_A)
            W1_vars.append(var)

        W1_vars = np.stack(W1_vars)

        W2_vars_A = W2_params_m[1].cpu().data.numpy()
        W2_vars_B = W2_params_m[2].cpu().data.numpy()
        W2_vars = []

        for W2_var_A, W2_var_B in zip(W2_vars_A, W2_vars_B):
            var = np.kron(W2_var_B, W2_var_A)
            W2_vars.append(var)

        W2_vars = np.stack(W2_vars)

        avg_var_1 += 1/M * W1_vars.sum(0)
        avg_var_2 += 1/M * W2_vars.sum(0)

    return avg_var_1, avg_var_2


avg_var_1, avg_var_2 = 0, 0

for _ in tqdm(range(args.n_samples)):
    _avg_var_1, _avg_var_2 = get_test_weight_params(X_test, region='all')
    avg_var_1 += 1/args.n_samples * _avg_var_1
    avg_var_2 += 1/args.n_samples * _avg_var_2

avg_var_mixing = np.mean(np.concatenate([avg_var_1, avg_var_2]))
print(f'Average variance of mixing distribution: {avg_var_mixing:.3f}')

W1_post_vars, W2_post_vars = model.get_weight_params_g()
avg_var_post = np.mean(np.concatenate([W1_post_vars, W2_post_vars]))
print(f'Average variance on posterior distribution: {avg_var_post:.3f}')
