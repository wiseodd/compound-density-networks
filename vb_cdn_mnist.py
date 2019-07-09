import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists
import numpy as np
import scipy.io
import input_data
import argparse
import math
from tqdm import tqdm
import os

import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf


parser = argparse.ArgumentParser()
parser.add_argument('--load', default=False, action='store_true', help='Load trained model')
parser.add_argument('--train_samples', type=int, default=1, help='Number of samples of \\theta to be used during training')
parser.add_argument('--n_samples', type=int, default=100, help='Number of joint samples of \\theta and \\psi to be used during testing')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--n_hidden', type=int, default=100, help='Hidden dimension of NN f')
parser.add_argument('--n_hidden_hypernet', type=int, default=100, help='Hidden dimension of NNs (g_l)')
parser.add_argument('--batch_size', type=int, default=200, help='Training batch size')
parser.add_argument('--n_iter', type=int, default=20000, help='Number of training iteration')
parser.add_argument('--randseed', type=int, default=9999, help='Random seed')
parser.add_argument('--adv_samples', type=int, default=1, help='Number of samples used to construct adversarial examples')
parser.add_argument('--info_interval', type=int, default=1000, help='Print info, e.g. loss, every x iterations')
args = parser.parse_args()


np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

name = 'vbcdn'

os.makedirs('./results/mnist', exist_ok=True)
os.makedirs('./models/mnist', exist_ok=True)

# Load data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
N_train = mnist.train.images.shape[0]
X_valid = torch.from_numpy(mnist.validation.images).cuda()
t_valid = mnist.validation.labels


class VMGLinear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(VMGLinear, self).__init__()

        in_dim += 1  # For biases
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_features = out_dim

        self.mu = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(in_dim, out_dim)))
        self.logvar_in = nn.Parameter(nn.init.normal_(torch.zeros(in_dim), -3, 1e-6))
        self.logvar_out = nn.Parameter(nn.init.normal_(torch.zeros(out_dim), -3, 1e-6))

    def forward(self, x, randseed=None):
        M = self.mu
        logvar_r = self.logvar_in
        logvar_c = self.logvar_out
        var_r = torch.exp(logvar_r)
        var_c = torch.exp(logvar_c)

        m = x.size(0)

        if randseed is not None:
            # Sample E (thus \psi) deterministically using a particular randseed, then re-randomize
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

        # Forward pass
        x = torch.cat([x, torch.ones(m, 1, device='cuda')], 1)
        h = torch.bmm(x.unsqueeze(1), W).squeeze()

        return h, D_KL


class ProbHypernet(nn.Module):

    def __init__(self, in_dim, out_dim, h_dim=100):
        super(ProbHypernet, self).__init__()

        self.in_dim = in_dim + 1
        self.out_dim = out_dim
        self.h_dim = h_dim

        self.M = nn.Parameter(torch.randn(self.in_dim, out_dim))
        torch.nn.init.kaiming_normal_(self.M)

        # Use BNN for g
        self.fc_xh = VMGLinear(in_dim, h_dim)
        self.fc_hmu = VMGLinear(h_dim, self.in_dim)
        self.fc_hlogvar_in = VMGLinear(h_dim, self.in_dim)
        self.fc_hlogvar_out = VMGLinear(h_dim, self.out_dim)

    def forward(self, x, psi_randseed=None, output_weight_params=False):
        m = x.shape[0]
        r, c = self.in_dim, self.out_dim

        # Generate parameters of mixing distribution by Bayesian hypernetworks
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

        # Reparametrization trick to generate weights W
        W = M + torch.sqrt(var_r).view(m, r, 1) * E * torch.sqrt(var_c).view(m, 1, c)

        # Actual forward pass on f
        x = torch.cat([x, torch.ones(m, 1, device='cuda')], 1)
        h = torch.bmm(x.unsqueeze(1), W).squeeze()

        if output_weight_params:
            return h, D_KL_psi, (M, var_r, var_c)
        else:
            return h, D_KL_psi


class Model(nn.Module):

    def __init__(self, h_dim=100, h_dim_hypernet=100):
        super(Model, self).__init__()

        self.fc_xh = ProbHypernet(784, h_dim, h_dim_hypernet)
        self.fc_hy = ProbHypernet(h_dim, 10, h_dim_hypernet)

    def forward(self, X, psi_randseed=None):
        h, D_KL_psi1 = self.fc_xh(X, psi_randseed)
        h = F.relu(h)

        y, D_KL_psi2 = self.fc_hy(h, None if psi_randseed is None else psi_randseed+100)

        if self.training:
            return y, D_KL_psi1+D_KL_psi2
        else:
            return y

    def get_weight_params(self, X):
        h, _, W1_params = self.fc_xh(X, output_weight_params=True)
        h = F.relu(h)
        _, _, W2_params = self.fc_hy(h, output_weight_params=True)
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

model = Model(h_dim, h_dim_hypernet).cuda()

print(f'Parameter count: {np.sum([value.numel() for value in model.parameters()])}')


if args.load:
    model.load_state_dict(torch.load(f'models/mnist/model_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{S}.bin'))
else:
    opt = optim.Adam(model.parameters(), lr)
    pbar = tqdm(range(args.n_iter))

    for i in pbar:
        X_mb, t_mb = mnist.train.next_batch(m)
        X_mb, t_mb = torch.from_numpy(X_mb).cuda(), torch.from_numpy(t_mb).long().cuda()

        log_p_y = []

        # Fix \psi, i.e. MC integration only over \theta
        psi_randseed = np.random.randint(9999999)

        for _ in range(S):
            y_s, D_KL_psi = model.forward(X_mb, psi_randseed)
            log_p_y_s = dists.Categorical(logits=y_s).log_prob(t_mb)
            log_p_y.append(log_p_y_s)

        # Use log-sum-exp trick to compute the log expectation for numerical stability
        nll = -torch.mean(torch.logsumexp(torch.stack(log_p_y), 0) - math.log(S))
        # ELBO
        loss = nll + (1/N)*D_KL_psi

        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5)
        opt.step()
        opt.zero_grad()

        if i % args.info_interval == 0:
            val_acc = validate(m)
            pbar.set_description(f'[NLL: {nll.data.item():.3f}; D_KL: {D_KL_psi.data.item():.3f}; val acc: {val_acc:.3f}]')

# Save model
if not args.load:
    torch.save(model.state_dict(), f'models/mnist/model_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{S}.bin')


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

np.save(f'results/mnist/y_test_mnist_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{S}.npy', y_MNIST)
np.save(f'results/mnist/y_test_notmnist_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{S}.npy', y_notMNIST)


# """ ======================= Distance between mean matrices ======================= """

# def get_test_weight_params(X_test, m=args.batch_size):
#     M = X_test.size(0)
#     Mean = 0

#     for i in range(0, M, m):
#         W1_params_m, W2_params_m = model.get_weight_params(X_test[i:i+m])
#         Ms = torch.cat([W1_params_m[0].flatten(1), W2_params_m[0].flatten(1)], dim=1)
#         Mean += 1/M * Ms.sum(0).cpu().data.numpy()

#     return Mean


# X_test_MNIST_1, X_test_MNIST_2 = X_test_MNIST, X_valid
# M_MNIST_1, M_MNIST_2, M_notMNIST = 0, 0, 0

# for _ in tqdm(range(args.n_samples)):
#     # MNIST test
#     A = get_test_weight_params(X_test_MNIST_1)
#     M_MNIST_1 += 1/args.n_samples * A

#     # MNIST valid
#     A = get_test_weight_params(X_test_MNIST_2)
#     M_MNIST_2 += 1/args.n_samples * A

#     # notMNIST test
#     A = get_test_weight_params(X_test_notMNIST)
#     M_notMNIST += 1/args.n_samples * A

# # Check distance between mean matrices to get insight about multimodality depending on within-distribution and OOD test.
# d_M_mnist = np.linalg.norm(M_MNIST_1 - M_MNIST_2)
# d_M_notmnist = np.linalg.norm(M_MNIST_2 - M_notMNIST)

# print(f'd_M_mnist: {d_M_mnist:.3f}; d_M_notmnist: {d_M_notmnist:.3f}')


""" ======================= Adversarial examples experiments ======================= """

model.eval()

# We use tf for evaluation on adversarial data
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
x_op = tf.placeholder(tf.float32, shape=(None, 784))

#  Convert pytorch model to a tf_model and wrap it in cleverhans
tf_model_fn = convert_pytorch_model_to_tf(model)
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
np.save(f'results/mnist/accs_adv_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{S}_{args.adv_samples}.npy', adv_accs)
np.save(f'results/mnist/ents_adv_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{S}_{args.adv_samples}.npy', adv_ents)
