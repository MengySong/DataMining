# mpiexec -n 2 python -m mpi4py RunNN.py
#####################################################################
# I feel really sorry for not completing the project with pipelining
# due to my extremly limited memory size. No matter what program I
# try to run it gets an error. Although I've tried several method it
# is still not solved. Thus I am not able to know whether my project
# can run successfully.
# The good news is that I implemented the neural network successfully.
# please check it. However, due to my memory limits I am only able to
# train it with mini database :(, I know that is not enough at all for
# a model. I'm still struggling to run it on Hydra because I need to
# install a lot of libraries on the server. Sorry for that again.

# ================ Outputs ===================
# Rank:  0
# Size:  1
# Device:  cuda
# Number of images (training dataset):  10
#   - sample and number of elements in category annotations:  10
# 	[0]  {'category_id': 1, 'id': 404873, 'image_id': 404873, 'institution_id': 0}
#   - sample and number of elements in category categories:  64500
# 	[0]  {'family': 'Orchidaceae', 'order': 'Asparagales', 'name': 'Aa calceata (Rchb.f.) Schltr.', 'id': 0}
#   - sample and number of elements in category images:  10
# 	[0]  {'file_name': 'images/000/01/404873.jpg', 'height': 1000, 'id': 404873, 'license': 0, 'width': 680}
#   - sample and number of elements in category info:  6
# 	[0]  contributor
#   - sample and number of elements in category licenses:  3
# 	[0]  {'id': 0, 'name': 'Public Domain Dedication', 'url': 'http://creativecommons.org/publicdomain/zero/1.0/'}
#   - sample and number of elements in category institutions:  5
# 	[0]  {'id': 0, 'name': 'New York Botanical Garden'}
# Labels converted to normalized encoding
# Epoch: 1/30
# Epoch: 001, Training: Loss: 1.6421, Accuracy: 75.0000%,
# 		Validation: Loss: 1.6421, Accuracy: 50.0000%, Time: 16.3294s
# Best Accuracy for validation : 0.5000 at epoch 001
# Epoch: 2/30
# Epoch: 002, Training: Loss: 0.7015, Accuracy: 75.0000%,
# 		Validation: Loss: 0.7015, Accuracy: 50.0000%, Time: 2.2053s
# Best Accuracy for validation : 0.5000 at epoch 001
# Epoch: 3/30
# Epoch: 003, Training: Loss: 0.7439, Accuracy: 87.5000%,
# 		Validation: Loss: 0.7439, Accuracy: 50.0000%, Time: 2.1897s
# Best Accuracy for validation : 0.5000 at epoch 001
# Epoch: 4/30
# Epoch: 004, Training: Loss: 0.4413, Accuracy: 75.0000%,
# 		Validation: Loss: 0.4413, Accuracy: 50.0000%, Time: 2.2053s
# Best Accuracy for validation : 0.5000 at epoch 001
# Epoch: 5/30
# Epoch: 005, Training: Loss: 0.2777, Accuracy: 87.5000%,
# 		Validation: Loss: 0.2777, Accuracy: 100.0000%, Time: 2.1988s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 6/30
# Epoch: 006, Training: Loss: 0.6885, Accuracy: 100.0000%,
# 		Validation: Loss: 0.6885, Accuracy: 50.0000%, Time: 2.1937s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 7/30
# Epoch: 007, Training: Loss: 0.6538, Accuracy: 87.5000%,
# 		Validation: Loss: 0.6538, Accuracy: 50.0000%, Time: 2.1983s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 8/30
# Epoch: 008, Training: Loss: 0.3811, Accuracy: 100.0000%,
# 		Validation: Loss: 0.3811, Accuracy: 100.0000%, Time: 2.1897s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 9/30
# Epoch: 009, Training: Loss: 0.3170, Accuracy: 100.0000%,
# 		Validation: Loss: 0.3170, Accuracy: 100.0000%, Time: 2.1982s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 10/30
# Epoch: 010, Training: Loss: 0.3128, Accuracy: 100.0000%,
# 		Validation: Loss: 0.3128, Accuracy: 100.0000%, Time: 2.1838s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 11/30
# Epoch: 011, Training: Loss: 0.3344, Accuracy: 100.0000%,
# 		Validation: Loss: 0.3344, Accuracy: 100.0000%, Time: 2.1897s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 12/30
# Epoch: 012, Training: Loss: 0.3879, Accuracy: 100.0000%,
# 		Validation: Loss: 0.3879, Accuracy: 50.0000%, Time: 2.1897s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 13/30
# Epoch: 013, Training: Loss: 0.4011, Accuracy: 100.0000%,
# 		Validation: Loss: 0.4011, Accuracy: 50.0000%, Time: 2.1906s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 14/30
# Epoch: 014, Training: Loss: 0.4351, Accuracy: 100.0000%,
# 		Validation: Loss: 0.4351, Accuracy: 50.0000%, Time: 2.1824s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 15/30
# Epoch: 015, Training: Loss: 0.4539, Accuracy: 100.0000%,
# 		Validation: Loss: 0.4539, Accuracy: 50.0000%, Time: 2.1819s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 16/30
# Epoch: 016, Training: Loss: 0.4740, Accuracy: 100.0000%,
# 		Validation: Loss: 0.4740, Accuracy: 50.0000%, Time: 2.1896s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 17/30
# Epoch: 017, Training: Loss: 0.4894, Accuracy: 100.0000%,
# 		Validation: Loss: 0.4894, Accuracy: 50.0000%, Time: 2.2053s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 18/30
# Epoch: 018, Training: Loss: 0.5021, Accuracy: 100.0000%,
# 		Validation: Loss: 0.5021, Accuracy: 50.0000%, Time: 2.1930s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 19/30
# Epoch: 019, Training: Loss: 0.5712, Accuracy: 100.0000%,
# 		Validation: Loss: 0.5712, Accuracy: 50.0000%, Time: 2.1896s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 20/30
# Epoch: 020, Training: Loss: 0.4946, Accuracy: 100.0000%,
# 		Validation: Loss: 0.4946, Accuracy: 50.0000%, Time: 2.1832s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 21/30
# Epoch: 021, Training: Loss: 0.4423, Accuracy: 100.0000%,
# 		Validation: Loss: 0.4423, Accuracy: 50.0000%, Time: 2.1898s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 22/30
# Epoch: 022, Training: Loss: 0.4214, Accuracy: 100.0000%,
# 		Validation: Loss: 0.4214, Accuracy: 50.0000%, Time: 2.1963s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 23/30
# Epoch: 023, Training: Loss: 0.4084, Accuracy: 100.0000%,
# 		Validation: Loss: 0.4084, Accuracy: 50.0000%, Time: 2.1826s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 24/30
# Epoch: 024, Training: Loss: 0.4022, Accuracy: 100.0000%,
# 		Validation: Loss: 0.4022, Accuracy: 50.0000%, Time: 2.2056s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 25/30
# Epoch: 025, Training: Loss: 0.3920, Accuracy: 100.0000%,
# 		Validation: Loss: 0.3920, Accuracy: 50.0000%, Time: 2.1897s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 26/30
# Epoch: 026, Training: Loss: 0.3890, Accuracy: 100.0000%,
# 		Validation: Loss: 0.3890, Accuracy: 50.0000%, Time: 2.1955s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 27/30
# Epoch: 027, Training: Loss: 0.3756, Accuracy: 100.0000%,
# 		Validation: Loss: 0.3756, Accuracy: 50.0000%, Time: 2.1897s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 28/30
# Epoch: 028, Training: Loss: 0.3587, Accuracy: 100.0000%,
# 		Validation: Loss: 0.3587, Accuracy: 100.0000%, Time: 2.2053s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 29/30
# Epoch: 029, Training: Loss: 0.3552, Accuracy: 100.0000%,
# 		Validation: Loss: 0.3552, Accuracy: 100.0000%, Time: 2.1983s
# Best Accuracy for validation : 1.0000 at epoch 005
# Epoch: 30/30
# Epoch: 030, Training: Loss: 0.3579, Accuracy: 100.0000%,
# 		Validation: Loss: 0.3579, Accuracy: 100.0000%, Time: 2.1872s
# Best Accuracy for validation : 1.0000 at epoch 005
#
#####################################################################

from projectMeta import node0_withoutMPI, node1_withoutMPI, node2_withoutMPI, split_to_node
import torch
import torchvision.models as models
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print("Rank: ", rank)
size = comm.Get_size()
print("Size: ", size)

# Load (pretrained) NN model
resnet18 = models.resnet18()  # raw model
# resnet18 = models.resnet18(pretrained=True) # with pretrained parameters (data: imagenet 1000)

# Check CUDA feasibility/availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    resnet18 = resnet18.cuda()
print("Device: ", device)

# run the training withoutMPI
def run_withoutMPI():
    training_data = node0_withoutMPI()
    training_data = node1_withoutMPI(training_data)
    node2_withoutMPI(training_data)
run_withoutMPI()


# run with MPI
def node0():
    # preprocess image with node0_withoutMPI and node1_withoutMPI
    training_data = node0_withoutMPI()
    training_data = node1_withoutMPI(training_data)

    # Scatter the file into 2 parts
    if rank == 0:
        file_to_scatter = split_to_node(training_data,2)
    else:
        file_to_scatter = None
    my_file = comm.scatter(file_to_scatter, root=0)
    print('I am Node', rank, 'and I got', len(my_file), 'images to process')
    comm.send((my_file), dest=1)


def node1():
    training_data = comm.recv(source=0)
    node2_withoutMPI(training_data)


# if rank == 0:
#     node0()
# elif rank == 1:
#     node1()
