import os
import pickle

import numpy as np
import torch
from torch import topk
from torch.utils.data import DataLoader, random_split

import cnn_model_wce
import constants as c
import CustomeDataset

num_sublists = c.NUM_SUBLISTS  # number of pickles


def top_k_accuracy(k, proba_pred_y, y_test):
    top_k_pred = proba_pred_y.argsort(axis=1)[:, -k:]
    final_pred = [False] * len(y_test)
    for j in range(len(y_test)):
        final_pred[j] = True if sum(top_k_pred[j] == y_test[j]) > 0 else False
    return np.mean(final_pred)


def made_data_lists(file_name):
    data = {}
    tensors_data = []
    lables_data = []
    #print(f"the filename is : {file_name}")

    with open(file_name, "rb") as f:
        data = pickle.load(f)

    for dir, emission in data.items():
        id = dir.split("/")[2]
        
        emission_tensor = torch.tensor(emission)
        #print(f"BTETS is : {emission_tensor.shape}")
        if emission_tensor.shape[1] != 199:  # 4 seconds -- 551, 3 seconds -- 413

            #print(f"ATETS is : {emission_tensor.shape}")
            continue
        tensors_data.append(emission_tensor)

        lables_data.append(int(id) - 1)  # to start the id from 0

    return tensors_data, lables_data


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    y_test_tag = torch.round(y_test)
    correct_results_sum = (y_pred_tag == y_test_tag).sum().float()
    acc = correct_results_sum / y_test_tag.shape[0]
    acc = torch.round(acc * 100)
    return acc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(str(device) + "\n")

model = cnn_model_wce.Convolutional_Neural_Network().to(device)
# model_path = "1000_speakers_w2v2+aug_4s.pth"
# model = torch.load(model_path)  # Load the model
continue_from_epoch = 0

learning_rate = c.LEARNING_RATE
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

epochs = c.EPOCHS
for epoch in range(continue_from_epoch, epochs):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    index = (epoch % c.NUM_SUBLISTS) + 1
   
    tensors_data, labels_data = made_data_lists(f"data{index}.pickle")

    torch_dataset = CustomeDataset.CustomeDataset(tensors_data, labels_data)

    train_ = int(torch_dataset.__len__() * c.TRAIN_PRE)
    valid_ = torch_dataset.__len__() - train_

    train_data, validation_data = random_split(
        dataset=torch_dataset, lengths=[train_, valid_]
    )

    train_loader = DataLoader(dataset=train_data, batch_size=c.BATCH_SIZE)
    val_loader = DataLoader(dataset=validation_data, batch_size=c.BATCH_SIZE)

    loss_list = []
    for (
        x_batch_train,
        y_batch_train,
    ) in train_loader:  # x_batch is tensors, y_batch is labels
        optimizer.zero_grad()  # initialize
        y_batch_train = y_batch_train.to(device)
        y_batch_train = y_batch_train.long()
        y_pred_train = model(x_batch_train.to(device))  # call forward() function
        y_pred_train = y_pred_train.float()
        loss_train = criterion(
            y_pred_train, y_batch_train
        )  # the difference between the predicaion and truth labels
        y_pred_train = torch.argmax(y_pred_train, 1)
        loss_train.backward()
        acc_train = binary_acc(y_pred_train, y_batch_train)
        optimizer.step()  # update the model parameters in the direction that minimizes the loss function
        epoch_loss += loss_train.item()
        epoch_acc += acc_train.item()

    model.eval()
    valid_loss = 0.0
    valid_acc = 0.0

    final_accuracy = np.array([0, 0, 0, 0], dtype=float)
    for x_batch_val, y_batch_val in val_loader:
        optimizer.zero_grad()
        y_batch_val = y_batch_val.to(device)
        with torch.no_grad():  # disable gradient calculation during the evaluation
            y_pred_val = model(x_batch_val.to(device))
            y_batch_val = y_batch_val.long()
            y_pred_val = y_pred_val.float()
            loss_val = criterion(y_pred_val, y_batch_val)
            # y_pred_val = torch.argmax(y_pred_val, 1)
            accuracy_list = []
            for k in [1, 2, 5, 10]:
                accuracy_list += [top_k_accuracy(k, y_pred_val, y_batch_val)]
            final_accuracy += np.array(accuracy_list)

            valid_loss += loss_val.item()

    print(
        f"Epoch {epoch + 0:03}: | Loss:{epoch_loss / len(train_loader):.5f} | Accuracy:{epoch_acc / len(train_loader):.3f}  | Val Loss:{valid_loss / len(val_loader):.3f}"
    )
    print(
        f"top 1%: {(final_accuracy[0]/len(val_loader))*100:.3f}, top 2%: {(final_accuracy[1]/len(val_loader))*100:.3f}, top 5%: {(final_accuracy[2]/len(val_loader))*100:.3f}, top 10%: {(final_accuracy[3]/len(val_loader))*100:.3f}"
    )
    print()

    #if (epoch % 2) == 0:
    try:
        # path = "F:/SI - wav2vec2/4 seconds/50_speakers_AUG_4s"
        #torch.save(model, model.to_string() + str(epoch) + ".pth")  .state_dict()
        torch.save(model.state_dict(), f"pth_container/Convolutional_Speaker_Identification_Log_Softmax_Model{epoch}.pth")
        print("Successes to save model")
    except:
        print("Not successes to save model")
