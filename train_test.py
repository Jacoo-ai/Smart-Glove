import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.signal import medfilt

train_df = pd.read_csv("gesture_dataset_320.csv")
test_df = pd.read_csv("gesture_dataset_320_test.csv")



df_all = train_df

label_col = df_all.columns[-1]  # Get the last column name
target_labels = ['0→0', '1→1','2→2','3→3','4→4','5→5','6→6','7→7']  # Change this to your specific labels

# Split the DataFrame based on the label list
df_others = df_all[df_all[label_col].isin(target_labels)]
df_24 = df_all[~df_all[label_col].isin(target_labels)]

# Shuffle each subset
df_others = df_others.sample(frac=1, random_state=42).reset_index(drop=True)
# Sample from each subset
sample_size = 900  # Adjust as needed (or use frac=0.2 for 20% sampling)
df_others = df_others.sample(n=min(sample_size, len(df_others)), random_state=42)



label_col = df_others.columns[-1]

# Replace all values in the last column with a single label (e.g., "new_label")
new_label = "other"  # Change this to the desired label
df_others[label_col] = new_label

#take only 24
# df_data = pd.concat([df_24,df_others])
df_data = df_24


# Split the DataFrame based on the label list
df_others = test_df[test_df[label_col].isin(target_labels)]
df_24 = test_df[~test_df[label_col].isin(target_labels)]
label_col = df_others.columns[-1]

# Replace all values in the last column with a single label (e.g., "new_label")
new_label = "other"  # Change this to the desired label
df_others[label_col] = new_label

#take only 24
# df_test = pd.concat([df_24,df_others])
df_test = df_24
# df_test


def get_data(df):
    # # df = pd.concat([df_dbh,df_zy])
    # # 转换时间序列数据
    feature_columns = df.columns[1:12]  # 去掉 Sample_ID 和 Label
    X = []
    count = 0
    for _, row in df.iterrows():
        sample = [np.array(row[col].split(), dtype=np.float32) for col in feature_columns]
        X.append(np.stack(sample, axis=1))  # (40, num_features)

    X = np.array(X)  # 转换为 (num_samples, 40, num_features)
    print("X 形状:", X.shape)  # 预期 (num_samples, 40, num_features)
    print(X[0])
    # 编码 Label
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Label"])
    print("y 形状:", y.shape)  # (num_samples,)
    print("类别映射:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

    # 进行平滑去噪
    for i in range(X.shape[0]):  # 遍历每个样本
        for j in range(X.shape[2]):  # 遍历每个特征
            if j < 6:  # IMU 数据（ax, ay, az, gx, gy, gz） -> 滑动平均
                X[i, :, j] = np.convolve(X[i, :, j], np.ones(5) / 5, mode='same')
            elif j >= 6 and j < 11:  # Flex 数据（flex1, flex2, ...） -> 中值滤波
                X[i, :, j] = medfilt(X[i, :, j], kernel_size=3)


    # 归一化数据
    scaler = MinMaxScaler()
    X = X.reshape(-1, X.shape[-1])  # (num_samples*window_size, num_features)
    X = scaler.fit_transform(X)  # 归一化
    X = X.reshape(-1, 15, len(feature_columns))  # 还原形状
    print(X[0])
    return X,y,scaler

X,y,scaler = get_data(df_data)
X_test,y_test,scalar= get_data(df_test)
from collections import Counter
# Counter(y)


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Assume your data is stored as NumPy arrays:
# X shape: (5000, 40, 11) where 5000 samples, 40 timesteps, 11 sensor features per timestep.
# y shape: (5000,) with integer class labels (e.g., 0 to num_classes-1).

# --- Step 1: Split the Data ---
# Use an 80/20 split for training and testing.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# --- Step 2: Convert NumPy Arrays to PyTorch Tensors ---
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# --- Step 3: Create TensorDatasets and DataLoaders ---
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Correct hidden state initialization
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # Hidden state
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # Cell state

        # Properly calling LSTM without extra arguments
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        # Extract last timestep's output for classification
        last_timestep_output = output[:, -1, :]  # Shape: (batch_size, hidden_size)
        return self.fc(last_timestep_output)  # Shape: (batch_size, num_classes)



import torch
import torch.nn as nn
import torch.optim as optim
import copy
# 假设已经定义了 LSTMClassifier，并且 train_loader 和 val_loader 已经准备好

# 定义模型
model = LSTMClassifier(input_size=11, hidden_size=64, num_layers=3, num_classes=24, dropout=0.3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load("best_pretrain_24classes_0.7714.pth"))
model.to(device)
patience = 20

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

num_epochs = 100
best_model = copy.deepcopy(model)
# 训练 + 验证循环
train_loss_list = []
train_accuracy_list = []
val_loss_list = []
val_accuracy_list = []
early_stop_counter = 0
best_val_acc = 0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 计算训练集的平均 Loss
    train_loss /= len(train_loader)

    # --- 验证步骤 ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 在验证阶段不计算梯度
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    # 计算验证集的平均 Loss 和准确率
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    train_loss_list.append(train_loss)
    # train_accuracy_list.append(train_accuracy)
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_accuracy)
    print(f'End of Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    # 检查是否是最好的模型
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        early_stop_counter = 0  # 复位 early stopping 计数器
        # torch.save(model.state_dict(), best_model_path)  # 保存最好的模型
        best_model = copy.deepcopy(model)
        print(f"Best model saved with Val Accuracy: {best_val_acc:.2f}%")
    else:
        early_stop_counter += 1  # 计数器 +1

    # 提前停止训练
    if early_stop_counter >= patience and best_val_acc >= 90:
        print(f"Early stopping triggered. No improvement for {patience} epochs.")
        break  # 结束训练

import matplotlib.pyplot as plt

epochs = list(range(1, len(train_loss_list) + 1))

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss_list, label='Train Loss', marker='o')
plt.plot(epochs, val_loss_list, label='Validation Loss', marker='s')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(8, 5))
# plt.plot(epochs, train_accuracy_list, label='', marker='o')
plt.plot(epochs, val_accuracy_list, label='Validation Accuracy', marker='s')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


total_correct = 0
total_samples = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs, 1)
        total_samples += batch_y.size(0)
        total_correct += (predicted == batch_y).sum().item()

test_accuracy = total_correct / total_samples
print(f'Test Accuracy: {test_accuracy:.4f}')
torch.save(model.state_dict(), "best_24classes_0.8263.pth")

