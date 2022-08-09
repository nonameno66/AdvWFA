"""
训练一个简单的LSTM网络
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pylab import *
import copy
import time
from sklearn.model_selection import train_test_split
import os, sys

torch.manual_seed(1)                      # reproducible
torch.set_printoptions(threshold=np.inf)  # print all

class GetLoader(torch.utils.data.Dataset):

    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):
        return len(self.data)

class RNN(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        self.out = nn.Linear(128, OUTPUT_SIZE)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        out_trace = self.out(r_out)  # choose r_out at all time steps
        return out, out_trace

# model = RNN(1,2).cuda()
# print(next(model.parameters()).device)  

# exit(0)

def main():
    # ----- Step 1: Set Hyper Parameters ----- #
    """
    PowerCons 数据集包含一年中个体家庭用电量分布在两个季节类别：暖季（1 类）和冷季（2 类），
    这取决于是否在暖季（4 月至 9 月）记录用电量或寒冷的季节（从十月到三月）。
    电力消耗曲线在类别内显着不同。采样率为一年内每十分钟一次。
    """
    is_train = True
    
    model = 'LSTM' # 使用的NN模型
    dataset = 'PowerCons' # 数据集
    NAME = f'{model}-UCR-{dataset}' # 输出名

    epochs = 2000 #200
    BATCH_SIZE = 200  # 150
    LR = 0.001

    TIME_STEP = 144
    INPUT_SIZE = 1
    OUTPUT_SIZE = 2

    train_data_path = f'/content/drive/My Drive/Datasets/UCR/{dataset}/{dataset}_TRAIN.tsv' # 训练数据
    test_data_path = f'/content/drive/My Drive/Datasets/UCR/{dataset}/{dataset}_TEST.tsv' # 测试数据

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # gpu

    print(f"Dataset: {dataset}")

    # ----- Step 2: Dataset Loading and Preprocessing ----- #

    train_text = ''
    test_text = ''

    train_x = [] # 训练集
    train_y = [] 
    test_x = [] # 测试集
    test_y = []

    with open(train_data_path, 'r') as f:
        train_text = f.read()
    train_lines = train_text.split('\n')
    for line in train_lines:
        _list = line.split('\t')
        if len(_list) > 1:
            train_y.append(float(_list[0])-1)
            flo_list = [[float(num)] for num in _list[1:]]
            train_x.append(flo_list)

    with open(test_data_path, 'r') as f:
        test_text = f.read()
    test_lines = test_text.split('\n')
    for line in test_lines:
        _list = line.split('\t')
        if len(_list) > 1:
            test_y.append(float(_list[0])-1)
            flo_list = [[float(num)] for num in _list[1:]]
            test_x.append(flo_list)
            
    train_x = np.array(train_x) # shape=(num_samples, num_feats, 1)
    train_y = np.array(train_y) # shape=(num_samples)
    test_x = np.array(test_x) # shape=(num_samples, num_feats, 1)
    test_y = np.array(test_y) # shape=(num_samples)

    """divide train set to train & valid set, train:valid=0.8:0.2"""
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=777)

    valid_X = torch.from_numpy(valid_x).to(torch.float32).to(device)
    valid_Y = valid_y #torch.from_numpy(valid_y).to(torch.long).to(device)


    """"""
    print('*** Dataset Information ***')
    print(f"train_x: {train_x.shape}, train_y: {train_y.shape}")
    print(f"test_x: {test_x.shape}, train_y: {test_y.shape}")
    print(f"valid_x: {valid_x.shape}, valid_y: {valid_y.shape}")


    train_X = copy.deepcopy(train_x)
    train_X = torch.from_numpy(train_X).to(torch.float32).to(device)
    train_Y = copy.deepcopy(train_y)

    test_X = torch.from_numpy(test_x).to(torch.float32).to(device)
    test_Y = test_y

    train_x = torch.from_numpy(train_x).to(torch.float32).to(device)
    train_y = torch.from_numpy(train_y).to(torch.long).to(device)
    train_data = GetLoader(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    train_Y1 = copy.deepcopy(train_y)


    # ----- Step 3: Create Model Class ----- #



    # ----- Step 4: Instantiate ----- #

    model = RNN(INPUT_SIZE=INPUT_SIZE, OUTPUT_SIZE=OUTPUT_SIZE)

    model = model.to(device)
    print("model device: ", next(model.parameters()).device)
    
    # return
    print('\n*** Model Information ***\n', model, '\n\n*** Training Information ***')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()


    # ----- Step 5: Model Training ----- #

    os.makedirs(f"/content/drive/My Drive/Models/{NAME}/checkpoint", exist_ok=True)
    log_path = f"/content/drive/My Drive/Models/{NAME}/train.log"

    if is_train:
        with open(log_path, 'w') as wf:
            for turns in range(1, epochs+1):
                for batch_idx, (train_x, train_y) in enumerate(train_loader):

                    # print(model.device)
                    # print(train_x.device)
                    train_x = train_x.view(-1, TIME_STEP, INPUT_SIZE)
                    output, _ = model(train_x)
                    loss = loss_fn(output, train_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_output, _ = model(train_X)
                    
                    pred_train_y = torch.max(train_output, 1)[1].data.cpu().numpy()
                    train_accuracy = float((pred_train_y == train_Y).astype(int).sum()) / float(train_Y.size)
                    train_loss = loss_fn(train_output, train_Y1)
                    
                    torch.save(model, f'/content/drive/My Drive/Models/{NAME}/checkpoint/epoch_{turns}_{batch_idx}.pkl')
                    if turns % 10 == 0:
                        print(f"Epoch: {turns}_{batch_idx} | train loss: {train_loss.data.cpu().numpy()} | train accuracy: {train_accuracy}")
                    wf.write(f"Epoch: {turns}_{batch_idx} | train loss: {train_loss.data.cpu().numpy()} | train accuracy: {train_accuracy}\n")

    
    """train"""
    # ------ Step 6: Finding The Optimum Model Automatically ------ #
    turns_chosen, batch_idx_chosen = find_optimum_model(train_X, train_Y, train_Y1, test_X, test_Y, valid_X, valid_Y, BATCH_SIZE, NAME)
    
    # ------ Step 7: RNN - WFA_Extraction ------ #
    final_vector, initial_vector, abst_alphabet_labels, non_prob_transition_matrixes = WFA_Extraction(train_X, train_Y, OUTPUT_SIZE, NAME, turns_chosen, batch_idx_chosen)
    
    # ------ Step 8: Evaluation of Models' Running Time and Accuracy ------ #
    differ_record, rnn_pred_train_y, wfa_pred_train_y, rnn_output, wfa_output = evaluation(model, train_X, train_Y, OUTPUT_SIZE, final_vector, initial_vector, abst_alphabet_labels, non_prob_transition_matrixes, set_name="Train")
    
    # ------ Step 9: find original adversarial samples ------ #
    # find_ori_adversarial_samples(train_X, train_Y, TIME_STEP, OUTPUT_SIZE, differ_record, rnn_pred_train_y, wfa_pred_train_y, rnn_output, wfa_output)
    
    """test"""
    final_vector, initial_vector, abst_alphabet_labels, non_prob_transition_matrixes = WFA_Extraction(test_X, test_Y, OUTPUT_SIZE, NAME, turns_chosen, batch_idx_chosen)
    
    differ_record, rnn_pred_train_y, wfa_pred_train_y, rnn_output, wfa_output = evaluation(model, test_X, test_Y, OUTPUT_SIZE, final_vector, initial_vector, abst_alphabet_labels, non_prob_transition_matrixes, set_name="Test")
    
    """valid"""
    final_vector, initial_vector, abst_alphabet_labels, non_prob_transition_matrixes = WFA_Extraction(valid_X, valid_Y, OUTPUT_SIZE, NAME, turns_chosen, batch_idx_chosen)
    
    differ_record, rnn_pred_train_y, wfa_pred_train_y, rnn_output, wfa_output = evaluation(model, valid_X, valid_Y, OUTPUT_SIZE, final_vector, initial_vector, abst_alphabet_labels, non_prob_transition_matrixes, set_name="Valid")
    

def find_optimum_model(train_X, train_Y, train_Y1, test_X, test_Y, valid_X, valid_Y, BATCH_SIZE, NAME):
    """
    Finding The Optimum Model Automatically
    从训练的所有epoch对应的模型里选最好的模型
    """

    loss_fn = nn.CrossEntropyLoss()


    min_loss = 10000
    max_test_acc = 0
    corresponding_train_acc = 0

    turns_chosen = -1
    batch_idx_chosen = -1

    for turns in range(37, 41):
        batch_number = int(train_X.shape[0]/BATCH_SIZE)
        if train_X.shape[0] % BATCH_SIZE != 0:
            batch_number += 1
        
        for batch_idx in range(batch_number):
            rnn = torch.load(f'/content/drive/My Drive/Models/{NAME}/checkpoint/epoch_{turns}_{batch_idx}.pkl')
            
            train_output, _ = rnn(train_X)
            pred_train_y = torch.max(train_output, 1)[1].data.cpu().numpy()
            train_accuracy = float((pred_train_y == train_Y).astype(int).sum()) / float(train_Y.size)
            train_loss = loss_fn(train_output, train_Y1)
            
            test_output, _ = rnn(test_X)
            pred_test_y = torch.max(test_output, 1)[1].data.cpu().numpy()
            test_accuracy = float((pred_test_y == test_Y).astype(int).sum()) / float(test_Y.size)
            
            
            valid_output, _ = rnn(valid_X)
            pred_valid_y = torch.max(valid_output, 1)[1].data.cpu().numpy()
            valid_accuracy = float((pred_valid_y == valid_Y).astype(int).sum()) / float(valid_Y.size)
            
            test_acc_bias = test_accuracy - max_test_acc
            loss_bias = train_loss.data.cpu().numpy() - min_loss
            if test_acc_bias >= 0.01 or (test_acc_bias >= 0 and test_acc_bias < 0.01 and loss_bias < 0.1):
                max_test_acc = test_accuracy
                min_loss = train_loss.data.cpu().numpy()
                corresponding_train_acc = train_accuracy
                turns_chosen = turns
                batch_idx_chosen = batch_idx

                corresponding_valid_accuracy = valid_accuracy


    print(f'*** The Optimum Model ***\nEpoch Index: {turns_chosen}_{batch_idx_chosen}')
    print('Train Loss: %.6f' % min_loss, 
          '\nTrain Accuracy: %.4f' % corresponding_train_acc, 
          '\nTest Accuracy: %.4f' % max_test_acc,
          '\nValid Accuracy: %.4f' % corresponding_valid_accuracy)
    return turns_chosen, batch_idx_chosen


def WFA_Extraction(train_X, train_Y, OUTPUT_SIZE, NAME, turns_chosen, batch_idx_chosen):
    """
    *** RNN - WFA Extraction ***
    对NN模型 使用WFA进行提取
    """

    def findByRow(mat, row):
        return np.where((mat == row).all(1))[0]


    # Set Hyper-parameter of WFA Establishment
    T = 150
    INPUT_DISTANCE_FACTOR = 0.5  # value range: (0,1], divide tokens in the content that grading size is micro compared with average distance

    if OUTPUT_SIZE >= 3:
        K = 3
    else:
        K = 2


    # ----- Step 1: Input Tokens Abstraction ----- #
    # print(f"*** Step 1: Input Tokens Abstraction ***")

    # Getting Normalized Alphabet
    train_X_numpy = train_X.detach().cpu().numpy()
    alphabet = copy.deepcopy(train_X_numpy)
    min_token = np.min(alphabet)
    alphabet -= min_token
    max_token = np.max(alphabet)
    alphabet /= max_token

    # Finding T_alphabet
    count = 0
    sum_distance = 0
    current_distance = np.zeros(train_X.shape[2])
    for i in range(train_X.shape[0]):
        for j in range(train_X.shape[1]-1):
            for k in range(train_X.shape[2]):
                current_distance[k] += abs(train_X[i, j+1, k] - train_X[i, j, k])
    average_distance = current_distance / (train_X.shape[0] * (train_X.shape[1]-1))
    T_alphabet = 1
    for i in range(train_X.shape[2]):
        T_alphabet *= int(10 / (INPUT_DISTANCE_FACTOR * average_distance[i]))
    # print(f"T_alphabet: {T_alphabet}")

    # Abstracting Input Tokens as k-DCP Format
    k_DCP = np.floor(alphabet*(T_alphabet-1))
    k_DCP = np.reshape(k_DCP, (-1, alphabet.shape[2]))
    uniques = np.unique(k_DCP, axis=0)
    abst_inputs_number = len(uniques)
    # print(f'Number of Abstracted Input Tokens: {abst_inputs_number}')

    # Building Labelling Representation of Abstracted Input Tokens
    abst_alphabet_labels = np.zeros(train_X.shape[0]*train_X.shape[1]).astype(int)
    for i in range(len(uniques)):
        j = findByRow(k_DCP, uniques[i, :])
        abst_alphabet_labels[j] = i
    abst_alphabet_labels = np.reshape(abst_alphabet_labels, (train_X.shape[0], train_X.shape[1]))


    # ----- Step 2: RNN Hidden States Abstraction (Under Sequential Inputs) ----- #
    # print('\n*** Step 2: RNN Hidden States Abstraction ***')

    # Extracting Hidden States from the Optimum Model
    rnn = torch.load(f'../Models/{NAME}/checkpoint/epoch_{turns_chosen}_{batch_idx_chosen}.pkl')
    train_output, train_output_trace = rnn(train_X)

    # Normaling Hidden States to Probability Distribution Format
    train_output_trace = F.softmax(train_output_trace, dim=2)
    states = train_output_trace.detach().cpu().numpy()

    # Abstracting the (Prob) States as k-DCP Format
    # print(f'An Example:\n - Original State: {states[0, 0, :]}')
    sorted_states = -np.sort(-states)[:, :, :K]

    sorted_states_index = np.argsort(-states)[:, :, :K]
    # print(f' - Corresponding Prediction Label: {sorted_states_index[0, 0, :]}')

    sorted_states_t = np.floor(sorted_states*(T-1))
    # print(f' - Corresponding Confidence Level: {sorted_states_t[0, 0, :]}')

    k_DCP = np.append(sorted_states_index, sorted_states_t, axis=2)
    # print(f' - Abstracted State: {k_DCP[0, 0, :]}')

    k_DCP = np.reshape(k_DCP, (-1, 2*K))
    uniques = np.unique(k_DCP, axis=0)
    abst_states_number = len(uniques)
    # print(f'Number of Abstracted States: {abst_states_number}')

    # Building Labelling Representation of the States
    abst_states_labels = np.zeros(train_X.shape[0]*train_X.shape[1]).astype(int)
    for i in range(len(uniques)):
        j = findByRow(k_DCP, uniques[i, :])
        abst_states_labels[j] = i
    abst_states_labels = np.reshape(abst_states_labels, (train_X.shape[0], train_X.shape[1]))
    abst_states_labels = abst_states_labels + 1


    # ----- Step 3: Establishment of the Intinal Vector ----- #
    # print('\n*** Step 3: Intinal Vector Establishment ***')

    initial_vector = np.zeros(abst_states_number+1)
    initial_vector[0] = 1
    # print('Shape of Initial Vector:\n', initial_vector.shape)
    # print('Initial Vector:\n', initial_vector)


    # ----- Step 4: Establishment of the Transition Matrixes ----- #
    # print('\n*** Step 4: Transition Matrixes Establishment ***')

    # Non-probabilistic Transition Matrixes Establishment
    non_prob_transition_matrixes = np.zeros((abst_inputs_number, abst_states_number+1, abst_states_number+1))
    for item in range(abst_inputs_number):
        for i_0 in range(train_X.shape[0]):
            for i_1 in range(train_X.shape[1]):
                if abst_alphabet_labels[i_0, i_1] == item:
                    if i_1 == 0:
                        front_abst_state = 0
                    else:
                        front_abst_state = int(abst_states_labels[i_0, i_1-1])
                    back_abst_state = int(abst_states_labels[i_0, i_1])
                    non_prob_transition_matrixes[item, front_abst_state, back_abst_state] += 1
    # print('Shape of Non_prob_transition_matrixes:\n', non_prob_transition_matrixes.shape)
    # print('Sample of a Non-prob Transition Matrix:\n', non_prob_transition_matrixes[0, :, :])

    # Probabilistic Transition Matrixes Establishment
    for item in range(abst_inputs_number):
        for i in range(non_prob_transition_matrixes.shape[1]):
            i_sum = np.sum(non_prob_transition_matrixes[item, i, :])
            if i_sum != 0:
                non_prob_transition_matrixes[item, i, :] /= i_sum
    # print('Shape of Prob_transition_matrixes:\n', non_prob_transition_matrixes.shape)
    # print('Sample of a Prob Transition Matrix:\n', non_prob_transition_matrixes[0, :, :])
    # print('Sum of the Element of the Sample Prob Transition Matrix:\n', np.sum(non_prob_transition_matrixes[0, :, :]))


    # ----- Step 5: Establishment of the Final Vector ----- #
    # print('\n*** Step 5: Final Vector Establishment ***')

    # (Non-probabilistic) Final Vectors Establishment
    non_prob_final_vector = np.zeros([abst_states_number+1, OUTPUT_SIZE])
    for i_0 in range(train_X.shape[0]):
        for i_1 in range(train_X.shape[1]):
            state_class = np.argsort(-states[i_0, i_1, :])[0]
            abst_label = int(abst_states_labels[i_0, i_1])
            non_prob_final_vector[abst_label, state_class] += 1
    # print('Shape of (Non-prob) Final Vector:\n', non_prob_final_vector.shape)
    # print('(Non-prob) Final Vector:\n', non_prob_final_vector)

    # (Probabilistic) Final Vectors Establishment
    final_vector = np.zeros([abst_states_number+1, OUTPUT_SIZE])
    output_0 = np.zeros(OUTPUT_SIZE)
    output_0_tensor = torch.from_numpy(output_0)
    state_0_tensor = F.softmax(output_0_tensor)
    state_0 = state_0_tensor.detach().cpu().numpy()
    final_vector[0, :] = state_0
    for item in range(1, abst_states_number+1):
        item_classes = non_prob_final_vector[item, :]
        item_sum = np.sum(item_classes)
        final_vector[item, :] = item_classes / item_sum
    # print('Shape of (Prob) Final Vector:\n', final_vector.shape)
    # print('(Prob) Final Vector:\n', final_vector)

    return final_vector, initial_vector, abst_alphabet_labels, non_prob_transition_matrixes

def evaluation(model, train_X, train_Y, OUTPUT_SIZE, final_vector, initial_vector, abst_alphabet_labels, non_prob_transition_matrixes, set_name="Train"):
    """
    Evaluation of Models' Running Time and Accuracy
    计算模型的运行时间和准确率
    """

    # Original RNN Evaluation
    print('*** Evaluation of Original RNN ***')
    time_start = time.time()
    rnn_output, _ = model(train_X)
    time_end = time.time()
    print('Running Time: %fs' % (time_end - time_start))
    rnn_pred_train_y = torch.max(rnn_output, 1)[1].data.cpu().numpy()
    rnn_train_accuracy = float((rnn_pred_train_y == train_Y).astype(int).sum()) / float(train_Y.size)
    print(f'Accuracy ({set_name}): {rnn_train_accuracy}')

    # WFA Evaluation
    time_count = 0.0
    WFA_output = np.zeros([train_X.shape[0], OUTPUT_SIZE])
    for i_0 in range(train_X.shape[0]):
        output = initial_vector
        for i_1 in range(train_X.shape[1]):
            index = int(abst_alphabet_labels[i_0, i_1])
            transition_matrix = non_prob_transition_matrixes[index, :, :]
            time_start = time.time()
            output = np.matmul(output, transition_matrix)
            time_end = time.time()
            time_count += time_end - time_start
        time_start = time.time()
        output = np.matmul(output, final_vector)
        time_end = time.time()
        time_count += time_end - time_start
        WFA_output[i_0, :] = output
    print('\n*** Evaluation of WFA ***')
    print('Running Time: %fs' % time_count)
    wfa_output = torch.from_numpy(WFA_output)
    wfa_pred_train_y = torch.max(wfa_output, 1)[1].data.cpu().numpy()
    wfa_train_accuracy = float((wfa_pred_train_y == train_Y).astype(int).sum()) / float(train_Y.size)
    print(f'Accuracy ({set_name}): {wfa_train_accuracy}')

    # Similarity of the Models
    print('\n*** Similarity ***')
    differ_record = []
    for i_0 in range(train_X.shape[0]):
        if wfa_pred_train_y[i_0] != rnn_pred_train_y[i_0]:
            differ_record.append(i_0)
            # print(f'RNN Output: {rnn_output[i_0, :]}')
            # print(f'WFA Output: {wfa_output[i_0, :]}')
            # print(f'Correct Lable: {train_Y[i_0]}')
    similarity = (1 - len(differ_record) / train_X.shape[0])*100
    print('Similarity between WFA and RNN: %.2f' % similarity, '%')
    
    return differ_record, rnn_pred_train_y, wfa_pred_train_y, rnn_output, wfa_output


def find_ori_adversarial_samples(train_X, train_Y, TIME_STEP, OUTPUT_SIZE, differ_record, rnn_pred_train_y, wfa_pred_train_y, rnn_output, wfa_output):
    """
    Finding Original Adversarial Samples
    寻找原始对抗样本
    """

    # Limitation Setting for Plt
    ylim_low = 65535
    ylim_high = 0
    for item in train_X:
        for ts in range(TIME_STEP):
            if item[ts, 0] > ylim_high:
                ylim_high = item[ts, 0]
            if item[ts, 0] < ylim_low:
                ylim_low = item[ts, 0]
    input_distance = ylim_high - ylim_low
    plot_distance = (input_distance) / 10

    # Finding Original Adversarial Sample
    timestep_record = [i for i in range(1, TIME_STEP+1)]
    for i in differ_record:
        if rnn_pred_train_y[i] != train_Y[i] and wfa_pred_train_y[i] == train_Y[i]:        
            print(f'\nOriginal Adversarial Sample: Input Sample [{i}]')
            print(f'RNN Output: {rnn_output[i, :]}')
            print(f'WFA Output: {wfa_output[i, :]}')
            print(f'Correct Lable: {train_Y[i]}')
            adv_X_record = []
            for item in np.array(train_X[i]):
                adv_X_record.append(item[0])
            plt.plot(timestep_record, adv_X_record)
            plt.ylim((ylim_low-plot_distance/2, ylim_high+plot_distance/2))
            plt.show()
            for j in range(OUTPUT_SIZE):
                for k in range(train_X.shape[0]):
                    if train_Y[k] == j:
                        X_record = []
                        for item in np.array(train_X[k]):
                            X_record.append(item[0]) 
                        plt.plot(timestep_record, X_record, color='#1f77b4')
                plt.plot(timestep_record, adv_X_record, color='orange')
                plt.ylim((ylim_low-plot_distance/2, ylim_high+plot_distance/2))
                plt.show()



if __name__ == "__main__":
    main()