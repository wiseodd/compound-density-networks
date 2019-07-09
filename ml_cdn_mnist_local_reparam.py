import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists
import numpy as np
import scipy.io
import foolbox
import input_data
import argparse
from tqdm import tqdm
import math
import os

import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf


parser = argparse.ArgumentParser()
parser.add_argument('--use_dropout', default=False, action='store_true')
parser.add_argument('--load', default=False, action='store_true')
parser.add_argument('--train_samples', type=int, default=1)
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--lam', type=float, default=1e-5)
parser.add_argument('--n_hidden', type=int, default=100)
parser.add_argument('--n_hidden_hypernet', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--n_iter', type=int, default=20000)
parser.add_argument('--randseed', type=int, default=9999)
parser.add_argument('--adv_samples', type=int, default=1)
parser.add_argument('--info_interval', type=int, default=1000)
args = parser.parse_args()


np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

name = 'mlcdn'

if args.use_dropout:
    name = 'dropout'

os.makedirs('./results/mnist', exist_ok=True)
os.makedirs('./models/mnist', exist_ok=True)

# Load training data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

N_train = mnist.train.images.shape[0]
X_valid = torch.from_numpy(mnist.validation.images).cuda()
t_valid = mnist.validation.labels


def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


class ProbHypernet(nn.Module):

    def __init__(self, in_dim, out_dim, h_dim=50):
        super(ProbHypernet, self).__init__()

        self.in_dim = in_dim + 1
        self.out_dim = out_dim
        self.h_dim = h_dim

        self.M = nn.Parameter(torch.randn(self.in_dim, out_dim))

        self.fc_xh = nn.Linear(in_dim, h_dim)
        nn.init.uniform_(self.fc_xh.weight, -0.0001, 0.0001)

        self.fc_hmu = nn.Linear(h_dim, self.in_dim)
        nn.init.uniform_(self.fc_hmu.weight, -0.0001, 0.0001)

        self.fc_hlogvar_in = nn.Linear(h_dim, self.in_dim)
        nn.init.uniform_(self.fc_hlogvar_in.weight, -0.0001, 0.0001)

        self.fc_hlogvar_out = nn.Linear(h_dim, out_dim)
        nn.init.uniform_(self.fc_hlogvar_out.weight, -0.0001, 0.0001)

    def forward(self, x, output_weight_params=False):
        m = x.shape[0]
        r, c = self.in_dim, self.out_dim

        h = self.fc_xh(x)
        h = F.relu(h)
        mu_scaling = self.fc_hmu(h)
        logvar_r = self.fc_hlogvar_in(h)
        logvar_c = self.fc_hlogvar_out(h)

        M = self.M

        M = mu_scaling.view(m, r, 1) * M  # Broadcasted: M is (m, r, c)
        var_r = torch.exp(logvar_r)
        var_c = torch.exp(logvar_c)

        # KL divergence to prior MVN(0, I, I)
        D_KL = torch.mean(
             1/2 * (torch.sum(var_r, 1)*torch.sum(var_c, 1) \
                    + torch.norm(M.view(m, -1), dim=1)**2 \
                    - r*c - c*torch.sum(logvar_r, 1) - r*torch.sum(logvar_c, 1))
        )

        x = torch.cat([x, torch.ones(m, 1, device='cuda')], 1)
        x_ = x.unsqueeze(1)

        # Recall MN(M, U, V) <=> N(M, V \otimes U).
        # Here, M is (m, c), U is (m, 1, 1),  V is (m, c, c) => we have N(M, U*V) where the multiplication is componentwise.
        mean_h = torch.bmm(x_, M).squeeze()
        XRX = torch.bmm(x_ * var_r.unsqueeze(1), x_.transpose(1, 2)).squeeze()
        var_h = XRX.unsqueeze(-1) * var_c

        eps = torch.randn(m, c, device='cuda')
        h = mean_h + eps * var_h

        if output_weight_params:
            return h, D_KL, (M, var_r, var_c)
        else:
            return h, D_KL


class Model(nn.Module):

    def __init__(self, h_dim=100, h_dim_hypernet=100, use_dropout=False):
        super(Model, self).__init__()

        self.use_dropout = use_dropout

        if not self.use_dropout:
            self.fc_xh = ProbHypernet(784, h_dim, h_dim_hypernet)
            self.fc_hy = ProbHypernet(h_dim, 10, h_dim_hypernet)
        else:
            self.fc_xh = nn.Linear(784, h_dim)
            self.fc_hy = nn.Linear(h_dim, 10)

    def forward(self, X):
        if not self.use_dropout:
            h, D_KL1 = self.fc_xh(X)
            h = F.relu(h)
            y, D_KL2 = self.fc_hy(h)

            return (y, D_KL1+D_KL2) if self.training else y
        else:
            h = F.relu(self.fc_xh(X))
            h = F.dropout(h, p=0.5, training=True)
            y = self.fc_hy(h)

            return y

    def get_weight_params(self, X):
        h, W1_params = self.fc_xh(X, output_weight_params=True)
        h = F.relu(h)
        y, W2_params = self.fc_hy(h, output_weight_params=True)
        return W1_params, W2_params


def validate(m=args.batch_size):
    model.eval()
    val_acc = 0

    for i in range(0, X_valid.size(0), m):
        y_i = model.forward(X_valid[i:i+m])
        val_acc += np.sum(y_i.argmax(dim=1).cpu().numpy() == t_valid[i:i+m])

    model.train()
    return val_acc/X_valid.size(0)


""" Training """
N = mnist.train.labels.shape[0]
S = args.train_samples
m = args.batch_size
lr = args.lr

h_dim = args.n_hidden
h_dim_hypernet = args.n_hidden_hypernet
model = Model(h_dim, h_dim_hypernet, args.use_dropout).cuda()

print(f'Parameter count: {np.sum([value.numel() for value in model.parameters()])}')


if args.load:
    model.load_state_dict(torch.load(f'models/mnist/model_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{args.wd}_{args.lam}_{S}.bin'))
else:
    opt = optim.Adam(model.parameters(), lr, weight_decay=args.wd)
    pbar = tqdm(range(args.n_iter))

    for i in pbar:
        X_mb, t_mb = mnist.train.next_batch(m)
        X_mb, t_mb = torch.from_numpy(X_mb).cuda(), torch.from_numpy(t_mb).long().cuda()

        if not args.use_dropout:
            log_p_y = []

            for _ in range(S):
                y_s, D_KL = model.forward(X_mb)
                log_p_y_s = dists.Categorical(logits=y_s).log_prob(t_mb)
                log_p_y.append(log_p_y_s)

            loss = -torch.mean(torch.logsumexp(torch.stack(log_p_y), 0) - math.log(S))
            loss += args.lam*D_KL
        else:
            y = model.forward(X_mb)
            loss = F.cross_entropy(y, t_mb)

        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        opt.step()
        opt.zero_grad()

        if i % args.info_interval == 0:
            val_acc = validate(m)
            pbar.set_description(f'[Loss: {loss.data.item():.3f}; val acc: {val_acc:.3f}]')

# Save model
if not args.load:
    torch.save(model.state_dict(), f'models/mnist/model_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{args.wd}_{args.lam}_{S}.bin')


""" ======================= Out-of-distribution experiments ======================= """

model.eval()

# Test on MNIST
X_test_MNIST = torch.from_numpy(mnist.test.images).float().cuda()
t_test_MNIST = mnist.test.labels

# Also test on notMNIST
notMNIST = scipy.io.loadmat('notMNIST_small.mat')
notMNIST_img = notMNIST['images'].reshape(784, -1).T / 255  # Normalize
assert notMNIST_img.shape == (notMNIST['images'].shape[-1], 784)

X_test_notMNIST = torch.from_numpy(notMNIST_img).float().cuda()
t_test_notMNIST = notMNIST['labels']


def test(X_test, m=args.batch_size):
    M = X_test.size(0)
    y = []

    for i in range(0, M, m):
        y_i = model.forward(X_test[i:i+m])
        y.append(F.softmax(y_i, dim=1).cpu().data.numpy())

    y = np.concatenate(y, 0)

    return y


def get_test_weight_params(X_test, m=args.batch_size):
    M = X_test.size(0)
    Mean = 0

    for i in range(0, M, m):
        W1_params_m, W2_params_m = model.get_weight_params(X_test[i:i+m])
        Ms = torch.cat([W1_params_m[0].flatten(1), W2_params_m[0].flatten(1)], dim=1)
        Mean += 1/M * Ms.sum(0).cpu().data.numpy()

    return Mean


y_val = 0
y_MNIST = 0
y_notMNIST = 0

for _ in tqdm(range(args.n_samples)):
    y_val += 1/args.n_samples * test(X_valid)
    y_MNIST += 1/args.n_samples * test(X_test_MNIST)
    y_notMNIST += 1/args.n_samples * test(X_test_notMNIST)

# Print accuracy
acc = np.mean(y_val.argmax(1) == t_valid)
print(f'Validation accuracy on MNIST: {acc:.3f}')

acc = np.mean(y_MNIST.argmax(1) == t_test_MNIST)
print(f'Test accuracy on MNIST: {acc:.3f}')

acc = np.mean(y_notMNIST.argmax(1) == t_test_notMNIST)
print(f'Test accuracy on notMNIST: {acc:.3f}')

np.save(f'results/mnist/y_test_mnist_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{args.wd}_{args.lam}_{S}.npy', y_MNIST)
np.save(f'results/mnist/y_test_notmnist_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{args.wd}_{args.lam}_{S}.npy', y_notMNIST)


""" ======================= Adversarial examples experiments ======================= """

model.eval()

# We use tf for evaluation on adversarial data
sess = tf.Session()
x_op = tf.placeholder(tf.float32, shape=(None, 784))

#  Convert pytorch model to a tf_model and wrap it in cleverhans
tf_model_fn = convert_pytorch_model_to_tf(model, out_dims=10)
cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

X_test = mnist.test.images
y_test = mnist.test.labels

M = X_test.shape[0]

adv_accs = []
adv_ents = []


def test_tf(m=100):
    preds = []

    for i in range(0, 1000, m):
        adv_x = np.zeros([m, 784])

        for _ in range(args.adv_samples):
            adv_x += 1/args.adv_samples * sess.run(adv_x_op, feed_dict={x_op: X_test[i:i+m]})

        adv_x = torch.cuda.FloatTensor(adv_x)

        pred = F.softmax(model.forward(adv_x), 1).cpu().data.numpy()
        preds.append(pred)

    preds = np.concatenate(preds, 0)
    return np.nan_to_num(preds)


for eps in np.arange(0, 1.01, 0.1):
    #  Create an FGSM attack
    fgsm_op = FastGradientMethod(cleverhans_model, sess=sess)
    fgsm_params = {'eps': eps,
                   'clip_min': 0.,
                   'clip_max': 1.}

    adv_x_op = fgsm_op.generate(x_op, **fgsm_params)
    adv_preds_op = tf_model_fn(x_op)

    #  Run an evaluation of our model against fgsm
    #  Use M data
    adv_preds = 0
    for _ in tqdm(range(args.n_samples)):
        preds = test_tf()
        adv_preds += 1/args.n_samples * preds

    #  Compute acc and entropy
    acc = (np.argmax(adv_preds, axis=1) == y_test[:1000]).mean()
    ent = (-adv_preds*np.log(adv_preds+1e-8)).sum(1).mean()

    adv_accs.append(acc)
    adv_ents.append(ent)

    print('Adv accuracy: {:.3f}'.format(acc))
    print('Avg entropy: {:.3f}'.format(ent))

sess.close()

# Save data
np.save(f'results/mnist/accs_adv_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{args.wd}_{args.lam}_{S}_{args.adv_samples}.npy', adv_accs)
np.save(f'results/mnist/ents_adv_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{args.wd}_{args.lam}_{S}_{args.adv_samples}.npy', adv_ents)
