import cv2  # OpenCV
import PIL  # Data Augmentation
# Operating System
import os
import time
# Utilities
import pandas as pd  # Handling CSV files
import json
from sklearn import preprocessing
# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from mpi4py import MPI


ROOT_DATA = "E:/project/DM/mini_herbarium/"
DATA_TRAIN = ROOT_DATA + "train/"
DATA_TEST = ROOT_DATA + "test/"
ROOT_OUTPUT = "working/"
META = "metadata.json"
SMALL= "small_metadata.json"
BATCH_SIZE = 512  # number of training examples utilized in one iteration
BATCH_EVAL = 512
SHUFFLE = True
EPOCHS = 1
LEARN_RATE = 4e-4
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = None  # define below
NUM_WORKERS = 4
PRE_TRAINED = False

PATH_SAVE_MODEL = "working/ResNet18_run-04.pth"

ESTIMATED_MAX_TRAINING_TIME = 480  # hours (8h * 60 = 480min, leaving 1h to the test time)


# Dataloaders
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, labels, transform=None):
        self.df = df
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["file_name"].values[idx]
        file_path = DATA_TRAIN + file_name
        img = cv2.imread(file_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels.values[idx]
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["file_name"].values[idx]
        file_path = DATA_TEST + file_name
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]

        return img


# Node 0 loads the Herbarium dataset and gets the list of all the images in it
def node0_withoutMPI():
    all_filenames = []
    # Training dataset
    with open(os.path.join(DATA_TRAIN, META), "r", encoding="ISO-8859-1") as file:
        meta_train = json.load(file)
        print("Number of images (training dataset): ",
              len(meta_train["images"]), )
        for i in list(meta_train.keys()):
            print("  - sample and number of elements in category {}: ".format(i),
                  len(list(meta_train[i])), )
            print("\t[0] ",
                  list(meta_train[i])[0], end="\n")

    for image in meta_train['images']:  # Iterate on images in the JSON file
        all_filenames.append(image['file_name'])
        # HACK: If we got 10K filenames, stop getting filenames, it's enough for a practical session
        if len(all_filenames) == 10000:
            break

    # Process metadata json for training images into a DataFrame
    train_img = pd.DataFrame(meta_train["images"])
    train_ann = pd.DataFrame(meta_train["annotations"]).drop(columns="image_id")
    train_df = train_img.merge(train_ann, on="id")  # Performs a database-style joint

    # Fit the label encoder instance
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(train_df["category_id"])

    # Transform labels to normalized encoding
    label = train_df["category_id"]
    train_df["category_id_le"] = label_encoder.transform(label)

    print("Labels converted to normalized encoding")

    train_data = TrainDataset(train_df, train_df["category_id_le"])

    real_train = []
    for idx in range(len(train_data)):
        img, lab = train_data.__getitem__(idx)
        real_train.append((img, lab))
    return real_train


# node1 preprocess the images.
import torchvision.transforms as T
def node1_withoutMPI(train_data):
    preprocessed_train = []
    for (img, lab) in train_data:
        transform = T.Compose([
            T.Resize([256,256]),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )]
        )
        label = torch.tensor(lab)
        PIL_image = PIL.Image.fromarray(img)
        image = transform(PIL_image)
        preprocessed_train.append((image, label))
    return preprocessed_train


def node2_withoutMPI(train_data):
    resnet18 = torchvision.models.resnet18(pretrained=False)
    fc_inputs = resnet18.fc.in_features
    resnet18.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2),
        nn.LogSoftmax(dim=1)
    )

    resnet18 = resnet18.to('cuda:0')
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(resnet18.parameters())
    num_epochs = 30
    train, val = train_val_split(train_data)
    trained_model, history = train_and_valid(train, val, resnet18, loss_func, optimizer, num_epochs)
    torch.save(history, ROOT_DATA + '_history.pt')


# Here I use 80% images for training and the rest 20% for validation. the function is implemented
# with train_val_split(train_data, ratio=0.8)
def train_and_valid(train_data, valid_data, model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for inputs, labels in train_data:
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(0)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels.unsqueeze(0))

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = inputs.unsqueeze(0)

                outputs = model(inputs)

                loss = loss_function(outputs, labels.unsqueeze(0))

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_data)
        avg_train_acc = train_acc / len(train_data)

        avg_valid_loss = valid_loss / len(valid_data)
        avg_valid_acc = valid_acc / len(valid_data)

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        torch.save(model, ROOT_DATA + '_model_' + str(epoch + 1) + '.pt')
    return model, history


def train_val_split(train_data, ratio=0.8):
    train_size = len(train_data)
    train = train_data[:int(0.8 * train_size)]
    val = train_data[int(0.8 * train_size):]
    return train, val


# split the training data into several dataset.
def split_to_node(train_data, size):
    elements_per_worker = len(train_data) // size
    file_to_scatter = []

    for i in range(size):
        # fr and to: define a range of filenames to give to the i-th worker
        fr = i * elements_per_worker
        to = fr + elements_per_worker

        if i == size - 1:
            # The last worker may have more images to process if <size> does not divide len(all_filenames)
            to = len(train_data)

        file_to_scatter.append(train_data[fr:to])
    print(len(file_to_scatter))
    return file_to_scatter



