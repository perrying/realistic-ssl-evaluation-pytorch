from torchvision import datasets
import argparse, os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", default=1, type=int, help="random seed")
parser.add_argument("--dataset", "-d", default="svhn", type=str, help="dataset name : [svhn, cifar10]")
parser.add_argument("--nlabels", "-n", default=1000, type=int, help="the number of labeled data")
args = parser.parse_args()

COUNTS = {
    "svhn": {"train": 73257, "test": 26032, "valid": 7326, "extra": 531131},
    "cifar10": {"train": 50000, "test": 10000, "valid": 5000, "extra": 0},
    "imagenet_32": {
        "train": 1281167,
        "test": 50000,
        "valid": 50050,
        "extra": 0,
    },
}

_DATA_DIR = "./data"

def split_l_u(train_set, n_labels):
    # NOTE: this function assume that train_set is shuffled.
    images = train_set["images"]
    labels = train_set["labels"]
    classes = np.unique(labels)
    n_labels_per_cls = n_labels // len(classes)
    l_images = []
    l_labels = []
    u_images = []
    u_labels = []
    for c in classes:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]
        u_images += [c_images[n_labels_per_cls:]]
        u_labels += [np.zeros_like(c_labels[n_labels_per_cls:]) - 1] # dammy label
    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0)}
    return l_train_set, u_train_set

def _load_svhn():
    splits = {}
    for split in ["train", "test", "extra"]:
        tv_data = datasets.SVHN(_DATA_DIR, split, download=True)
        data = {}
        data["images"] = tv_data.data
        data["labels"] = tv_data.labels
        splits[split] = data
    return splits.values()

def _load_cifar10():
    splits = {}
    for train in [True, False]:
        tv_data = datasets.CIFAR10(_DATA_DIR, train, download=True)
        data = {}
        data["images"] = tv_data.data
        data["labels"] = np.array(tv_data.targets)
        splits["train" if train else "test"] = data
    return splits.values()

def gcn(images, multiplier=55, eps=1e-10):
    # global contrast normalization
    images = images.astype(np.float)
    images -= images.mean(axis=(1,2,3), keepdims=True)
    per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
    per_image_norm[per_image_norm < eps] = 1
    return multiplier * images / per_image_norm

def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, height*width*channels)
    image_cov = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
    zca_decomp = np.dot(U, np.dot(np.diag(1/np.sqrt(S + eps)), U.T))
    mean = images.mean(axis=0)
    return mean, zca_decomp

def zca_normalization(images, mean, decomp):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, -1)
    images = np.dot((images - mean), decomp)
    return images.reshape(n_data, height, width, channels)

rng = np.random.RandomState(args.seed)

validation_count = COUNTS[args.dataset]["valid"]

extra_set = None  # In general, there won't be extra data.
if args.dataset == "svhn":
    train_set, test_set, extra_set = _load_svhn()
elif args.dataset == "cifar10":
    train_set, test_set = _load_cifar10()
    train_set["images"] = gcn(train_set["images"])
    test_set["images"] = gcn(test_set["images"])
    mean, zca_decomp = get_zca_normalization_param(train_set["images"])
    train_set["images"] = zca_normalization(train_set["images"], mean, zca_decomp)
    test_set["images"] = zca_normalization(test_set["images"], mean, zca_decomp)
    # N x H x W x C -> N x C x H x W
    train_set["images"] = np.transpose(train_set["images"], (0,3,1,2))
    test_set["images"] = np.transpose(test_set["images"], (0,3,1,2))

# permute index of training set
indices = rng.permutation(len(train_set["images"]))
train_set["images"] = train_set["images"][indices]
train_set["labels"] = train_set["labels"][indices]

if extra_set is not None:
    extra_indices = rng.permutation(len(extra_set["images"]))
    extra_set["images"] = extra_set["images"][extra_indices]
    extra_set["labels"] = extra_set["labels"][extra_indices]

# split training set into training and validation
train_images = train_set["images"][validation_count:]
train_labels = train_set["labels"][validation_count:]
validation_images = train_set["images"][:validation_count]
validation_labels = train_set["labels"][:validation_count]
validation_set = {"images": validation_images, "labels": validation_labels}
train_set = {"images": train_images, "labels": train_labels}

# split training set into labeled data and unlabeled data
l_train_set, u_train_set = split_l_u(train_set, args.nlabels)

if not os.path.exists(os.path.join(_DATA_DIR, args.dataset)):
    os.mkdir(os.path.join(_DATA_DIR, args.dataset))

np.save(os.path.join(_DATA_DIR, args.dataset, "l_train"), l_train_set)
np.save(os.path.join(_DATA_DIR, args.dataset, "u_train"), u_train_set)
np.save(os.path.join(_DATA_DIR, args.dataset, "val"), validation_set)
np.save(os.path.join(_DATA_DIR, args.dataset, "test"), test_set)
if extra_set is not None:
    np.save(os.path.join(_DATA_DIR, args.dataset, "extra"), extra_set)

