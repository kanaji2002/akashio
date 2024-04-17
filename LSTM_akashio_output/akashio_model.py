import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 正規化されたデータ
normalized = np.array([[0.33783784, 0.25, 1.0, 0.98795181,1],
                      [1.0, 0.125, 0.0, 0.0,0],
                      [0.33783784, 0.25, 1.0, 0.38554217,1],
                      [0.45945946, 0.0, 0.0, 1.0,0],
                      [0.0, 0.25, 1.0, 0.38554217,1],
                      [0.32432432, 0.125, 0.0, 0.75903614,1],
                      [0.0, 1.0, 1.0, 0.26506024,0],
                      [0.32432432, 0.125, 0.0, 0.39759036,1],
                      [0.33783784, 0.25, 1.0, 0.98795181,1],
                      [1.0, 0.125, 0.0, 0.0,0],
                      [0.33783784, 0.25, 1.0, 0.38554217,1],
                      [0.45945946, 0.0, 0.0, 1.0,0],
                      [0.0, 0.25, 1.0, 0.38554217,1],
                      [0.32432432, 0.125, 0.0, 0.75903614,1],
                      [0.0, 1.0, 1.0, 0.26506024,0],
                      [0.32432432, 0.125, 0.0, 0.39759036,1]])

# 説明変数と目的変数を分割
X = normalized[:, :-1]  # 説明変数
y = normalized[:, -1]   # 目的変数

# PyTorchのテンソルに変換
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# LSTMモデル定義
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

# モデル、損失関数、最適化手法の定義
input_size = X.shape[1]
hidden_size = 300
output_size = 1

model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# モデルの学習
num_epochs = 300
#epochs_list = list(range(num_epochs))
epochs_list =[]
loss_list = []

for epoch in range(num_epochs):
    outputs = model(X.view(-1, 1, input_size))
    loss = criterion(outputs.view(-1), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if(epoch==0):
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
        epochs_list.append(epoch)#損失関数を表示させるためのコード
        loss_list.append(loss.item())
        plt.plot(epochs_list, loss_list, color="k")
        
    
    elif(epoch%10==9):
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
        epochs_list.append(epoch)#損失関数を表示させるためのコード
        loss_list.append(loss.item())
        plt.plot(epochs_list, loss_list, color="k")
    
# 損失関数のラベルの指定と表示   
plt.legend()  
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.show()
    
# モデルの評価
# model.eval()
# with torch.no_grad():
#     test_output = model(X.view(-1, 1, input_size))
#     predicted = (test_output.view(-1).numpy() > 0.5).astype(int)
#     print("Predicted labels:", predicted)
    
#ここから変更していく．11/10

# # nissyaryou,tyouryuusokudo , ennbunnnoudo , suionn
# test=[3,43,2,1]
# test_pred=model.predicted(test)
# print(test_pred)

# モデルの評価
# モデルの評価
model.eval()
with torch.no_grad():
    # 学習データに対する予測
    train_output = model(X.view(-1, 1, input_size))
    train_predicted = (train_output.view(-1).numpy() > 0.5).astype(int)

    # 学習データの予測結果をグラフに描画
    plt.figure(figsize=(10, 5))
    plt.plot(y.numpy(), label='Actual labels', marker='o')
    plt.plot(train_predicted, label='Predicted labels', marker='x')
    plt.title('Training Data: Actual vs. Predicted Labels')
    plt.xlabel('Sample')
    plt.ylabel('Label')
    plt.legend()
    plt.show()

    # 新しいデータに対する予測
    new_data = torch.tensor([[3, 43, 2, 1]], dtype=torch.float32)
    test_output = model(new_data.view(-1, 1, input_size))
    test_predicted = (test_output.view(-1).numpy() > 0.5).astype(int)

    # 新しいデータの予測結果を表示
    print("Predicted labels for new data:", test_predicted)


