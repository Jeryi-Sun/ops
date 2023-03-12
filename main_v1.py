import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset.dataset import MyDataset
from config import const
from models.transformer import Transformer, ModelArgs




def train(model, train_loader, val_loader, lr=0.001, epochs=10, device='cpu', save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for hist, label in train_loader:
            hist, label = hist.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(hist)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for hist, label in val_loader:
                hist, label = hist.to(device), label.to(device)
                output = model(hist)
                y_true += label.cpu().numpy().tolist()
                y_pred += output.argmax(dim=-1).cpu().numpy().tolist()
        acc = accuracy_score(y_true, y_pred)
        print(f'Epoch {epoch+1}/{epochs}: val accuracy {acc:.4f}')

        # 保存最优模型参数
        if acc > best_acc and save_path is not None:
            best_acc = acc
            torch.save(model.state_dict(), save_path)

    return model

if __name__ == '__main__':
    # 加载数据
    dataset = MyDataset('train.tsv', 'rec_inter.pickle', max_len=100)

    train_loader = DataLoader(const.train_file, batch_size=64, shuffle=True)
    val_loader = DataLoader(const.valid_file, batch_size=128)

    """
    增加数据类
    """
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size
    )

    # 定义模型
    model = Transformer(input_dim=50, hidden_dim=64, num_layers=2, num_heads=4)

    # 训练模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model = train(model, train_loader, val_loader, lr=0.001, epochs=10, device=device)