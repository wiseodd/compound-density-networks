import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists
import numpy as np
import input_data
import argparse
import math


parser = argparse.ArgumentParser()
parser.add_argument('--use_dropout', default=False, action='store_true')
parser.add_argument('--n_data', type=int, default=20)
parser.add_argument('--n_points', type=int, default=1000)
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--lam', type=float, default=1e-3)
parser.add_argument('--n_hidden', type=int, default=100)
parser.add_argument('--n_hidden_hypernet', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--randseed', type=int, default=12)
parser.add_argument('--dataset', type=str, default='cubic')
parser.add_argument('--train_samples', type=int, default=3)
parser.add_argument('--n_iter', type=int, default=10000)
parser.add_argument('--hetero', default=False, action='store_true')
args = parser.parse_args()


name = 'mlcdn'

if args.use_dropout:
    name = 'dropout'

os.makedirs('/results/toy/{args.dataset}', exist_ok=True)
os.makedirs('/models/toy/{args.dataset}', exist_ok=True)

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

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


class ProbHypernet(nn.Module):

    def __init__(self, in_dim, out_dim, h_dim=10):
        super(ProbHypernet, self).__init__()

        self.in_dim = in_dim + 1
        self.out_dim = out_dim
        self.h_dim = h_dim

        self.M = nn.Parameter(torch.randn(self.in_dim, out_dim))
        torch.nn.init.uniform_(self.M, -0.0001, 0.0001)

        self.fc_xh = nn.Linear(in_dim, h_dim)

        self.fc_hmu = nn.Linear(h_dim, self.in_dim)
        self.fc_hlogvar_in = nn.Linear(h_dim, self.in_dim)
        self.fc_hlogvar_out = nn.Linear(h_dim, out_dim)

        self.E = torch.zeros(self.in_dim, out_dim)

    def forward(self, x):
        m = x.shape[0]

        W, D_KL = self.sample(x)

        x = torch.cat([x, torch.ones(m, 1, device='cuda')], 1)
        h = torch.bmm(x.unsqueeze(1), W).squeeze()

        return h, D_KL

    def sample(self, x):
        m = x.shape[0]
        r, c = self.in_dim, self.out_dim

        h = F.relu(self.fc_xh(x))
        mu_scaling = self.fc_hmu(h)
        logvar_r = self.fc_hlogvar_in(h)
        logvar_c = self.fc_hlogvar_out(h)

        # Reparametrization Trick
        M = mu_scaling.view(m, r, 1) * self.M.expand(m, r, c)  # Broadcasted: M is (m, r, c)
        var_r = torch.exp(logvar_r)
        var_c = torch.exp(logvar_c)

        E = torch.randn(m, *self.E.shape, device='cuda')

        # Reparametrization trick
        W = M + torch.sqrt(var_r).view(m, r, 1) * E * torch.sqrt(var_c).view(m, 1, c)

        # KL divergence to prior MVN(0, I, I)
        D_KL = torch.mean(
             1/2 * (torch.sum(var_r, 1)*torch.sum(var_c, 1) \
                    + torch.norm(M.view(m, -1), dim=1)**2 \
                    - r*c - c*torch.sum(logvar_r, 1) - r*torch.sum(logvar_c, 1))
        )

        return W, D_KL


class Model(nn.Module):

    def __init__(self, h_dim=100, h_dim_hypernet=10, use_dropout=False):
        super(Model, self).__init__()

        self.use_dropout = use_dropout

        if not self.use_dropout:
            self.fc_xh = ProbHypernet(1, h_dim, h_dim_hypernet)
            self.fc_hy = ProbHypernet(h_dim, 1, h_dim_hypernet)
        else:
            self.fc_xh = nn.Linear(1, h_dim)
            self.fc_hy = nn.Linear(h_dim, 1)

            if self.use_dropout:
                self.dropout = lambda x: F.dropout(x, p=0.5, training=True)

    def forward(self, X):
        if not self.use_dropout:
            h, D_KL1 = self.fc_xh(X)
            h = F.relu(h)

            y, D_KL2 = self.fc_hy(h)
            y = y.unsqueeze(1)

            return (y, D_KL1+D_KL2) if self.training else y
        else:
            h = F.relu(self.fc_xh(X))

            if self.use_dropout:
                h = self.dropout(h)

            y = self.fc_hy(h)
            return y

    def sample_theta(self, x):
        theta1, _ = self.fc_xh.sample(x)
        h = F.relu(self.fc_xh(x)[0])
        theta2, _ = self.fc_hy.sample(h)

        return theta1, theta2


""" Training """
h_dim = args.n_hidden
h_dim_hypernet = args.n_hidden_hypernet
model = Model(h_dim, h_dim_hypernet, args.use_dropout).cuda()

print(f'Parameter count: {np.sum([value.numel() for value in model.parameters()])}')

S = args.train_samples
m = args.batch_size
lr = args.lr
lam = args.lam

opt = optim.Adam(model.parameters(), lr, weight_decay=args.wd)


for i in range(args.n_iter):
    if not args.use_dropout:
        p = dists.Normal(y_train, 1/math.sqrt(2)*torch.ones_like(y_train))

        log_p_y = []

        for _ in range(S):
            y_s, D_KL = model.forward(X_train)
            log_p_y_s = p.log_prob(y_s)
            log_p_y.append(log_p_y_s)

        loss = -torch.mean(torch.logsumexp(torch.stack(log_p_y), 0) - math.log(S))
        loss += args.lam*D_KL
    else:
        y = model.forward(X_train)
        loss = F.mse_loss(y, y_train)

    loss.backward()
    opt.step()
    opt.zero_grad()

    if i % 200 == 0:
        print('\n==========================================')
        print('Iter-{}'.format(i))
        print('==========================================')
        print('Loss: {:.3f}'.format(loss.data.item()))


model.eval()

# Test for whole domain
if not args.hetero:
    X_test = torch.linspace(-6, 6, args.n_points, device='cuda').unsqueeze(1)
else:
    X_test = torch.linspace(-4, 4, args.n_points, device='cuda').unsqueeze(1)

z = torch.cat([model.forward(X_test) for _ in range(args.n_samples)], dim=1).t()

z = z.cpu().data.numpy()
np.save(f'results/toy/{args.dataset}/y_test_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{args.wd}_{args.n_data}_{S}.npy', z)
np.save(f'results/toy/{args.dataset}/X_train_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{args.wd}_{args.n_data}.npy', X_train.cpu().data.numpy())
np.save(f'results/toy/{args.dataset}/y_train_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{args.wd}_{args.n_data}.npy', y_train.cpu().data.numpy())

print(f'results/toy/{args.dataset}/y_test_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{args.wd}_{args.n_data}_{S}.npy')
