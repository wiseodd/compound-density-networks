import torch
import torchvision
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
import data_loader
import itertools
import math

import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf



parser = argparse.ArgumentParser()
parser.add_argument('--load', default=False, action='store_true')
parser.add_argument('--train_samples', type=int, default=1)
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_hidden', type=int, default=100)
parser.add_argument('--n_hidden_hypernet', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--randseed', type=int, default=9999)
args = parser.parse_args()


np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

name = 'vbcdn'

os.makedirs('/results/cifar', exist_ok=True)
os.makedirs('/models/cifar', exist_ok=True)

# Load training data
trainset, testset = data_loader.load_dataset('cifar10_pretrained')
print(len(trainset.dataset))


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
        torch.nn.init.xavier_normal_(self.M)

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

        self.fc_xh = ProbHypernet(1024, h_dim, h_dim_hypernet)
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
    total = 0

    for x, y in testset:
        x = x.cuda()
        y_i = model.forward(x)
        val_acc += np.sum(y_i.argmax(dim=1).cpu().numpy() == y.numpy())
        total += x.shape[0]

    model.train()
    return val_acc/total


""" Training """
S = args.train_samples
m = args.batch_size
lr = args.lr
N = len(trainset.dataset)

h_dim = args.n_hidden
h_dim_hypernet = args.n_hidden_hypernet
model = Model(h_dim, h_dim_hypernet).cuda()

print(f'Parameter count: {np.sum([value.numel() for value in model.parameters()])}')


if args.load:
    model.load_state_dict(torch.load(f'models/cifar/model_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{S}.bin'))
else:
    opt = optim.Adam(model.parameters(), lr)

    train_losses = []
    val_accs = []

    pbar = tqdm(range(100))
    it = 0

    for i in pbar:
        for x, y in trainset:
            x = x.cuda()
            y = y.cuda()

            log_p_y = []

            # Fix \psi, i.e. MC integration only over \theta
            psi_randseed = np.random.randint(9999999)

            for _ in range(S):
                y_s, D_KL = model.forward(x, psi_randseed)
                log_p_y_s = dists.Categorical(logits=y_s).log_prob(y)
                log_p_y.append(log_p_y_s)

            # Use log-sum-exp trick to compute the log expectation for numerical stability
            nll = -torch.mean(torch.logsumexp(torch.stack(log_p_y), 0) - math.log(S))
            loss = nll + (1/N)*D_KL

            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 5)
            opt.step()
            opt.zero_grad()

        val_acc = validate(m)
        pbar.set_description(f'[Loss: {loss.data.item():.3f}; val acc: {val_acc:.3f}]')

# Save model
if not args.load:
    torch.save(model.state_dict(), f'models/cifar/model_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{S}.bin')


""" ======================= Adversarial examples experiments ======================= """

model.eval()
print(model.training)

input_shape = (None, 1024)

# Assemble the whole net, i.e. concatenate DenseNet121 with the trained model above
input_shape = (None, 3, 32, 32)
trainset, testset = data_loader.load_dataset('cifar10')
pretrained_model = torchvision.models.densenet121(pretrained=True).cuda()
pretrained_model = torch.nn.Sequential(*(list(pretrained_model.children())[:-1]))
pretrained_model.eval()
model = nn.Sequential(pretrained_model, model)
model.eval()

# We use tf for evaluation on adversarial data
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
x_op = tf.placeholder(tf.float32, shape=input_shape)

#  Convert pytorch model to a tf_model and wrap it in cleverhans
tf_model_fn = convert_pytorch_model_to_tf(model, out_dims=10)
cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

adv_accs = []
adv_ents = []


def test_tf(use_adv=True):
    preds = []
    y_test = []
    total = 0

    for x, y in testset:
        x = x.permute(0, 3, 1, 2)

        if use_adv:
            pred = sess.run(adv_preds_op, feed_dict={x_op: x})
            pred = F.softmax(torch.from_numpy(pred), 1).numpy()
        else:
            pred = model.forward(x.cuda())
            pred = F.softmax(pred, 1).cpu().data.numpy()

        preds.append(pred)
        y_test.append(y)
        total += x.shape[0]

        if total >= 1000:
            break

    preds = np.concatenate(preds, 0)
    y_test = np.concatenate(y_test, 0)

    return np.nan_to_num(preds), y_test


if USE_BAYES:
    adv_preds = 0
    for _ in tqdm(range(args.n_samples)):
        preds, y_test = test_tf(False)
        adv_preds += 1/args.n_samples * preds
else:
    adv_preds, y_test = test_tf(False)

#  Compute acc and entropy
acc = (np.argmax(adv_preds, axis=1) == y_test).mean()
ent = (-adv_preds*np.log(adv_preds+1e-8)).sum(1).mean()

adv_accs.append(acc)
adv_ents.append(ent)

print('Adv accuracy: {:.3f}'.format(acc))
print('Avg entropy: {:.3f}'.format(ent))


for eps in np.arange(0.1, 1.01, 0.1):
    #  Create an FGSM attack
    fgsm_op = FastGradientMethod(cleverhans_model, sess=sess)
    fgsm_params = {'eps': eps,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv_x_op = fgsm_op.generate(x_op, **fgsm_params)
    adv_preds_op = tf_model_fn(adv_x_op)

    #  Run an evaluation of our model against fgsm
    #  Use M data
    if USE_BAYES:
        adv_preds = 0
        for _ in tqdm(range(args.n_samples)):
            preds, y_test = test_tf()
            adv_preds += 1/args.n_samples * preds
    else:
        adv_preds, y_test = test_tf()

    #  Compute acc and entropy
    acc = (np.argmax(adv_preds, axis=1) == y_test).mean()
    ent = (-adv_preds*np.log(adv_preds+1e-8)).sum(1).mean()

    adv_accs.append(acc)
    adv_ents.append(ent)

    print('Adv accuracy: {:.3f}'.format(acc))
    print('Avg entropy: {:.3f}'.format(ent))

sess.close()

# Save data
np.save(f'results/cifar/accs_adv_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{S}.npy', adv_accs)
np.save(f'results/cifar/ents_adv_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{S}.npy', adv_ents)
