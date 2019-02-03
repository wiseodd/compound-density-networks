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
import math

import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf


parser = argparse.ArgumentParser()
parser.add_argument('--use_dropout', default=False, action='store_true')
parser.add_argument('--normalize', default=False, action='store_true')
parser.add_argument('--load', default=False, action='store_true')
parser.add_argument('--train_samples', type=int, default=1)
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--lam', type=float, default=1e-7)
parser.add_argument('--n_hidden', type=int, default=100)
parser.add_argument('--n_hidden_hypernet', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--n_iter', type=int, default=100)
parser.add_argument('--randseed', type=int, default=9999)
args = parser.parse_args()


np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

name = 'mlcdn'

if args.use_dropout:
    name = 'dropout'

os.makedirs('/results/cifar', exist_ok=True)
os.makedirs('/models/cifar', exist_ok=True)

# Load training data
trainset, testset = data_loader.load_dataset('cifar10_pretrained')


class ProbHypernet(nn.Module):

    def __init__(self, in_dim, out_dim, h_dim=100):
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

        E = torch.randn(m, r, c, device='cuda')

        # Reparametrization trick
        W = M + torch.sqrt(var_r).view(m, r, 1) * E * torch.sqrt(var_c).view(m, 1, c)

        # KL divergence to prior MVN(0, I, I)
        D_KL = torch.mean(
             1/2 * (torch.sum(var_r, 1)*torch.sum(var_c, 1) \
                    + torch.norm(M.view(m, -1), dim=1)**2 \
                    - r*c - c*torch.sum(logvar_r, 1) - r*torch.sum(logvar_c, 1))
        )

        x = torch.cat([x, torch.ones(m, 1, device='cuda')], 1)
        h = torch.bmm(x.unsqueeze(1), W).squeeze()

        if output_weight_params:
            return h, D_KL, (M, var_r, var_c)
        else:
            return h, D_KL


class Model(nn.Module):

    def __init__(self, h_dim=100, h_dim_hypernet=50, use_dropout=False):
        super(Model, self).__init__()

        self.use_dropout = use_dropout

        if not self.use_dropout:
            self.fc_xh = ProbHypernet(1024, h_dim, h_dim_hypernet)
            self.fc_hy = ProbHypernet(h_dim, 10, h_dim_hypernet)
        else:
            self.fc_xh = nn.Linear(1024, h_dim)
            self.fc_hy = nn.Linear(h_dim, 10)

    def forward(self, X):
        X = X.squeeze()

        if not self.use_dropout:
            h, D_KL1 = self.fc_xh(X)
            h = F.relu(h)
            y, D_KL2 = self.fc_hy(h)

            return (y, D_KL1+D_KL2) if self.training else y
        else:
            h = F.relu(self.fc_xh(X))

            if self.use_dropout:
                h = F.dropout(h, p=0.5, training=True)

            y = self.fc_hy(h)
            return y


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
lam = args.lam

h_dim = args.n_hidden
h_dim_hypernet = args.n_hidden_hypernet
model = Model(h_dim, h_dim_hypernet, args.use_dropout).cuda()

print(f'Parameter count: {np.sum([value.numel() for value in model.parameters()])}')


if args.load:
    model.load_state_dict(torch.load(f'models/cifar/model_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{args.wd}_{args.lam}_{S}.bin'))
else:
    opt = optim.Adam(model.parameters(), lr, weight_decay=args.wd)
    pbar = tqdm(range(args.n_iter))

    for i in pbar:
        for x, y in trainset:
            x = x.cuda()
            y = y.cuda()

            if not args.use_dropout:
                log_p_y = []
                D_KL = 0

                for _ in range(S):
                    y_s, D_KL = model.forward(x)
                    log_p_y_s = dists.Categorical(logits=y_s).log_prob(y)
                    log_p_y.append(log_p_y_s)

                loss = -torch.mean(torch.logsumexp(torch.stack(log_p_y), 0) - math.log(S))
                loss += args.lam*D_KL
            else:
                out = model.forward(x)
                loss = F.cross_entropy(out, y)

            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 5)
            opt.step()
            opt.zero_grad()

        val_acc = validate(m)
        pbar.set_description(f'[Loss: {loss.data.item():.3f}; val acc: {val_acc:.3f}]')

# Save model
if not args.load:
    torch.save(model.state_dict(), f'models/cifar/model_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{args.wd}_{args.lam}_{S}.bin')

""" =============================== Validate ======================================= """

def test():
    model.eval()
    y = []
    t = []

    for x_test, y_test in testset:
        x_test = x_test.cuda()
        y_i = model.forward(x_test)
        y.append(F.softmax(y_i, dim=1).cpu().data.numpy())
        t.append(y_test)

    y = np.concatenate(y, 0)
    t = np.concatenate(t)

    return y, t


y_val = 0

for _ in tqdm(range(args.n_samples)):
    y_s, t = test()
    y_val += 1/args.n_samples*y_s

# Print accuracy
acc = np.mean(y_val.argmax(1) == t)
print(f'Test accuracy on CIFAR-10: {acc:.3f}')



""" ======================= Adversarial examples experiments ======================= """

model.eval()

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


adv_preds = 0
for _ in tqdm(range(args.n_samples)):
    preds, y_test = test_tf(False)
    adv_preds += 1/args.n_samples * preds

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
    adv_preds = 0
    for _ in tqdm(range(args.n_samples)):
        preds, y_test = test_tf()
        adv_preds += 1/args.n_samples * preds

    #  Compute acc and entropy
    acc = (np.argmax(adv_preds, axis=1) == y_test).mean()
    ent = (-adv_preds*np.log(adv_preds+1e-8)).sum(1).mean()

    adv_accs.append(acc)
    adv_ents.append(ent)

    print('Adv accuracy: {:.3f}'.format(acc))
    print('Avg entropy: {:.3f}'.format(ent))

sess.close()

# Save data
np.save(f'results/cifar/accs_adv_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{args.wd}_{args.lam}_{S}.npy', adv_accs)
np.save(f'results/cifar/ents_adv_{name}_{h_dim}_{h_dim_hypernet}_{m}_{lr}_{args.wd}_{args.lam}_{S}.npy', adv_ents)

