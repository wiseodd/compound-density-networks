import torch
import torchvision as tv
import numpy as np
import data_loader
from tqdm import tqdm

os.makedirs('/data', exist_ok=True)

trainset, testset = data_loader.load_dataset('cifar10')

pretrained_model = tv.models.densenet121(pretrained=True).cuda()
model = torch.nn.Sequential(*(list(pretrained_model.children())[:-1]))  # Exclude FC layer


def extract_features(dataset, m=100):
    model.eval()
    features = []
    targets = []

    for x, y in tqdm(dataset):
        x = x.permute(0, 3, 1, 2).cuda()
        feat = model.forward(x)
        features.append(feat.squeeze().cpu().data.numpy())
        targets.append(y)

    model.train()
    return np.concatenate(features, axis=0), np.concatenate(targets)


train_features, train_targets = extract_features(trainset)
test_features, test_targets = extract_features(testset)

print(train_features.shape, test_features.shape, train_targets.shape, test_targets.shape)

np.save('data/cifar10_features_train.npy', train_features)
np.save('data/cifar10_targets_train.npy', train_targets)
np.save('data/cifar10_features_test.npy', test_features)
np.save('data/cifar10_targets_test.npy', test_targets)
