# from datetime import datetime
#
# import numpy as np
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from net.st_gcn import Model
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
#
# ###### **model parameters**
# W = 128  # window size
# TS = 64  # number of voters per test subject
#
# ###### **training parameters**
# LR = 0.001  # learning rate
# batch_size = 32
#
# ###### setup model & data
# net = Model(1, 1, None, True)
# net.to(device)
#
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.0005)
#
# train_data = torch.from_numpy(np.load('data2_tr5/train_data.npy')).float()
# train_label = torch.from_numpy(np.load('data2_tr5/train_label.npy')).float()
# val_data = torch.from_numpy(np.load('data2_tr5/val_data.npy')).float()
# val_label = torch.from_numpy(np.load('data2_tr5/val_label.npy')).float()
#
# train_dataset = TensorDataset(train_data, train_label)
# val_dataset = TensorDataset(val_data, val_label)
#
# # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=12)
# # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=12)
# num_workers = 0  # Windows 建议从 0 开始
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#
#
# print(train_data.shape)
# print(val_data.shape)
#
# ###### start training model
# training_loss = 0.0
# val_loss = 0.0
# best_acc = 0.0
# train_acc = 0.0
#
# val_num = len(val_dataset)
# val_steps = len(val_loader)
# train_steps = len(train_loader)
# train_num = len(train_dataset)
#
# for epoch in range(200):  # number of mini-batches
#     # construct a mini-batch by sampling a window W for each subject
#     for train_data_batch, train_label_batch in train_loader:
#         train_data_batch_dev = train_data_batch.to(device)
#         train_label_batch_dev = train_label_batch.to(device)
#
#         # forward + backward + optimize
#         optimizer.zero_grad()
#         outputs = net(train_data_batch_dev)
#         loss = criterion(outputs.squeeze(-1), train_label_batch_dev)
#         predict = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
#         train_acc += (predict == train_label_batch_dev)[0].sum().item()
#         loss.backward()
#         optimizer.step()
#         torch.cuda.empty_cache()
#
#         # print training statistics
#         training_loss += loss.item()
#
#     # net.eval()
#     acc = 0.0
#     with torch.no_grad():
#         for val_data_batch, val_label_batch in val_loader:
#             val_data_batch_dev = val_data_batch.to(device)
#             val_label_batch_dev = val_label_batch.to(device)
#             outputs = net(val_data_batch_dev)
#             loss = criterion(outputs.squeeze(-1), val_label_batch_dev)
#             outputs = outputs.squeeze(-1)
#             predict = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
#             acc += (predict == val_label_batch_dev).sum().item()
#             val_loss += loss.item()
#
#         val_accurate = acc / val_num
#         val_loss = val_loss / val_steps
#         if best_acc < val_accurate:
#             best_acc = val_accurate
#             torch.save(net.state_dict(), 'output/best_checkpoint.pth')
#
#         results = '[' + str(epoch + 1) + ']' + "train acc " + str(train_acc / train_num) + " training loss:" + str(
#             training_loss / train_steps) + " val acc: " + str(val_accurate) + " val_loss: " + str(val_loss)
#         print(results)
#         with open('output/final.txt', 'a+') as f:
#             time = datetime.now()
#             f.write(str(time) + results)
#             f.write("\n")
#         train_acc = 0.0
#         training_loss = 0.0
# from datetime import datetime
# import os  # 导入 os 库
# import numpy as np
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from net.st_gcn import Model
# from sklearn.manifold import TSNE  # 导入 t-SNE
# import matplotlib.pyplot as plt  # 导入 matplotlib
#
# # 确保输出目录存在
# if not os.path.exists('output'):
#     os.makedirs('output')
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
#
# ###### **model parameters**
# W = 128  # window size
# TS = 64  # number of voters per test subject
#
# ###### **training parameters**
# LR = 0.001  # learning rate
# batch_size = 32
#
# ###### setup model & data
# net = Model(1, 1, None, True)
# net.to(device)
#
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.0005)
#
# train_data = torch.from_numpy(np.load('data1_tr5/train_data.npy')).float()
# train_label = torch.from_numpy(np.load('data1_tr5/train_label.npy')).float()
# val_data = torch.from_numpy(np.load('data1_tr5/val_data.npy')).float()
# val_label = torch.from_numpy(np.load('data1_tr5/val_label.npy')).float()
#
# train_dataset = TensorDataset(train_data, train_label)
# val_dataset = TensorDataset(val_data, val_label)
#
# # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=12)
# # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=12)
# num_workers = 0  # Windows 建议从 0 开始
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#
# print(train_data.shape)
# print(val_data.shape)
#
# ###### start training model
# training_loss = 0.0
# val_loss = 0.0
# best_acc = 0.0
# train_acc = 0.0
#
# val_num = len(val_dataset)
# val_steps = len(val_loader)
# train_steps = len(train_loader)
# train_num = len(train_dataset)
#
# for epoch in range(200):  # number of mini-batches
#     net.train()  # 设置为训练模式
#     # construct a mini-batch by sampling a window W for each subject
#     for train_data_batch, train_label_batch in train_loader:
#         train_data_batch_dev = train_data_batch.to(device)
#         train_label_batch_dev = train_label_batch.to(device)
#
#         # forward + backward + optimize
#         optimizer.zero_grad()
#         # 将 [32, 1, 1, 1] 压缩为 [32]
#         outputs = net(train_data_batch_dev)
#         outputs = outputs.squeeze()
#
#         # 现在形状匹配了，可以计算损失
#         loss = criterion(outputs, train_label_batch_dev)
#
#         # 计算准确率
#         predict = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
#         train_acc += (predict == train_label_batch_dev).sum().item()
#         loss.backward()
#         optimizer.step()
#         torch.cuda.empty_cache()
#
#         # print training statistics
#         training_loss += loss.item()
#
#     net.eval()  # 设置为评估模式
#     acc = 0.0
#     val_features_list = []  # 用于存储特征
#     val_labels_list = []  # 用于存储标签
#     with torch.no_grad():
#         for val_data_batch, val_label_batch in val_loader:
#             val_data_batch_dev = val_data_batch.to(device)
#             val_label_batch_dev = val_label_batch.to(device)
#
#             # 调用新方法获取输出和特征
#             outputs, features = net.extract_feature(val_data_batch_dev)
#
#             # 收集特征和标签
#             val_features_list.append(features.cpu().numpy())
#             val_labels_list.append(val_label_batch.cpu().numpy())
#
#             # 同样地，将 [32, 1, 1, 1] 压缩为 [32]
#             outputs = outputs.squeeze()
#
#             # 计算损失
#             loss = criterion(outputs, val_label_batch_dev)
#
#             # 计算准确率
#             predict = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
#             acc += (predict == val_label_batch_dev).sum().item()
#             val_loss += loss.item()
#
#         val_accurate = acc / val_num
#         val_loss_avg = val_loss / val_steps
#         if best_acc < val_accurate:
#             best_acc = val_accurate
#             torch.save(net.state_dict(), 'output/best_checkpoint.pth')
#
#         # START: t-SNE 可视化代码
#         # 每隔 20 个 epoch 或者在最后一个 epoch 生成一次 t-SNE 图像
#         # if (epoch + 1) % 20 == 0 or epoch == 199:
#         #     print(f"Epoch {epoch + 1}: Generating t-SNE plot...")
#         if epoch == 199:
#             print(f"Epoch {epoch + 1}: Generating t-SNE plot...")
#         # 将 list 转换为 numpy array
#             all_features = np.concatenate(val_features_list, axis=0)
#             all_labels = np.concatenate(val_labels_list, axis=0)
#
#             # 执行 t-SNE 降维
#             tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
#             features_2d = tsne.fit_transform(all_features)
#
#             # 绘图
#             plt.figure(figsize=(10, 8))
#             # unique_labels = np.unique(all_labels)
#             # colors = ['r', 'g'] # 假设是二分类
#
#             # 绘制不同类别的散点
#             for label_val in np.unique(all_labels):
#                 indices = np.where(all_labels == label_val)
#                 plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=f'Class {int(label_val)}',
#                             alpha=0.7)
#
#             plt.title(f't-SNE Visualization in dataset1')
#             plt.xlabel('t-SNE dimension 1')
#             plt.ylabel('t-SNE dimension 2')
#             plt.legend()
#             # 保存图像
#             plt.savefig(f'output/tsne_epoch1_{epoch + 1}.png')
#             plt.close()  # 关闭图像，防止显示
#             print(f"t-SNE plot saved to output/tsne_epoch1_{epoch + 1}.png")
#         # END: t-SNE 可视化代码
#
#         results = '[' + str(epoch + 1) + ']' + "train acc " + str(train_acc / train_num) + " training loss:" + str(
#             training_loss / train_steps) + " val acc: " + str(val_accurate) + " val_loss: " + str(val_loss_avg)
#         print(results)
#         with open('output/final.txt', 'a+') as f:
#             time = datetime.now()
#             f.write(str(time) + results)
#             f.write("\n")
#         train_acc = 0.0
#         training_loss = 0.0
#         val_loss = 0.0  # 重置 val_loss
from datetime import datetime
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from net.st_gcn import Model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 设置全局字体参数，使所有新生成的文本默认加粗黑色
# 但更细粒度的控制可能需要直接在plt.xlabel, plt.ylabel, plt.legend中设置
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['text.color'] = 'black'  # 默认文本颜色

# 确保输出目录存在
if not os.path.exists('output'):
    os.makedirs('output')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

###### **model parameters**
W = 128
TS = 64

###### **training parameters**
LR = 0.001
batch_size = 32

###### setup model & data
net = Model(1, 1, None, True)
net.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.0005)

# 假设您有两个不同的数据集，分别用于生成dataset1和dataset2的图
# 如果您只有一个数据集，您需要手动运行两次，每次加载不同的数据文件来生成两张图
# 这里以 dataset1 和 dataset2 为例，您可能需要修改这里的路径来适配您的实际数据集
# For dataset1:
# train_data = torch.from_numpy(np.load('data1_tr5/train_data.npy')).float()
# train_label = torch.from_numpy(np.load('data1_tr5/train_label.npy')).float()
# val_data = torch.from_numpy(np.load('data1_tr5/val_data.npy')).float()
# val_label = torch.from_numpy(np.load('data1_tr5/val_label.npy')).float()
# current_dataset_name = "dataset1" # 用于文件名和标题

# For dataset2 (your current setup):
train_data = torch.from_numpy(np.load('data2_tr5/train_data.npy')).float()
train_label = torch.from_numpy(np.load('data2_tr5/train_label.npy')).float()
val_data = torch.from_numpy(np.load('data2_tr5/val_data.npy')).float()
val_label = torch.from_numpy(np.load('data2_tr5/val_label.npy')).float()
current_dataset_name = "dataset2"  # 用于文件名和标题

train_dataset = TensorDataset(train_data, train_label)
val_dataset = TensorDataset(val_data, val_label)

num_workers = 0
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

print(f"Loading data from {current_dataset_name}")
print(train_data.shape)
print(val_data.shape)

###### start training model
training_loss = 0.0
val_loss = 0.0
best_acc = 0.0
train_acc = 0.0

val_num = len(val_dataset)
val_steps = len(val_loader)
train_steps = len(train_loader)
train_num = len(train_dataset)

patience = 50
trigger_times = 0
best_val_loss = float('inf')

for epoch in range(200):
    net.train()
    for train_data_batch, train_label_batch in train_loader:
        train_data_batch_dev = train_data_batch.to(device)
        train_label_batch_dev = train_label_batch.to(device)

        optimizer.zero_grad()
        outputs = net(train_data_batch_dev)
        outputs = outputs.squeeze()

        loss = criterion(outputs, train_label_batch_dev)

        predict = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
        train_acc += (predict == train_label_batch_dev).sum().item()

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        training_loss += loss.item()

    net.eval()
    acc = 0.0
    val_features_list = []
    val_labels_list = []
    with torch.no_grad():
        for val_data_batch, val_label_batch in val_loader:
            val_data_batch_dev = val_data_batch.to(device)
            val_label_batch_dev = val_label_batch.to(device)

            outputs, features = net.extract_feature(val_data_batch_dev)
            val_features_list.append(features.cpu().numpy())
            val_labels_list.append(val_label_batch.cpu().numpy())

            outputs = outputs.squeeze()

            loss = criterion(outputs, val_label_batch_dev)

            predict = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
            acc += (predict == val_label_batch_dev).sum().item()

            val_loss += loss.item()

        val_accurate = acc / val_num
        val_loss_avg = val_loss / val_steps
        if best_acc < val_accurate:
            best_acc = val_accurate
            # 保存带有数据集名称的checkpoint
            torch.save(net.state_dict(), f'output/best_checkpoint_{current_dataset_name}.pth')
        else:
            trigger_times += 1
            print(f'EarlyStopping counter: {trigger_times} out of {patience}')
            if trigger_times >= patience:
                print('Early stopping!')
                break
        # if epoch == 199:  # 只在最后一个 epoch 生成图
        #     print(f"Final Epoch {epoch + 1}: Generating t-SNE plot for {current_dataset_name}...")
        #
        #     all_features = np.concatenate(val_features_list, axis=0)
        #     all_labels = np.concatenate(val_labels_list, axis=0)
        #
        #     tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
        #     features_2d = tsne.fit_transform(all_features)
        #
        #     plt.figure(figsize=(10, 8))
        #     for label_val in np.unique(all_labels):
        #         indices = np.where(all_labels == label_val)
        #         # 修改图例标签为 'data 0'/'data 1'
        #         plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=f'class {int(label_val)}', alpha=0.7)
        #
        #     # 修改主标题为更简洁的 "t-SNE Visualization"
        #     plt.title('t-SNE Visualization in dataset2', fontsize=16, fontweight='bold', color='black')  # 标题也加粗加黑
        #
        #     # 设置横纵坐标标签，加粗加黑
        #     plt.xlabel('t-SNE dimension 1', fontsize=14, fontweight='bold', color='black')
        #     plt.ylabel('t-SNE dimension 2', fontsize=14, fontweight='bold', color='black')
        #
        #     # 调整图例字体和位置
        #     plt.legend(fontsize=10, loc='upper right', frameon=True)  # frameon=True 可以让图例有边框
        #
        #     # 保存图片，文件名包含数据集名称
        #     plt.savefig(f'output/tsne_{current_dataset_name}_final.png', dpi=300, bbox_inches='tight')
        #     plt.close()
        #     print(f"Final t-SNE plot for {current_dataset_name} saved to output/tsne_{current_dataset_name}_final.png")

        results = '[' + str(epoch + 1) + ']' + "train acc " + str(train_acc / train_num) + " training loss:" + str(
            training_loss / train_steps) + " val acc: " + str(val_accurate) + " val_loss: " + str(val_loss_avg)
        print(results)
        with open(f'output/final_log_{current_dataset_name}.txt', 'a+') as f:  # 日志文件也加上数据集名称
            time = datetime.now()
            f.write(str(time) + results)
            f.write("\n")