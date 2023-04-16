import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import const
import numpy as np
import random
import os
import setproctitle
from models.model_ARIMA import ARIMA_model
from tqdm import tqdm
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
setup_seed(1)

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, help='experiment name', default='default')
parser.add_argument('--description', type=str, help='exp details, used for log name', default='default')
parser.add_argument('--workspace', type=str, default='./workspace')

parser.add_argument('--dataset_name', type=str, default='kuaishou')
parser.add_argument('--use_cpu', dest='use_gpu', action='store_false')
parser.set_defaults(use_gpu=True)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--tb', type=bool, help='whether use tensorboard (record metrics)', default=True)
parser.add_argument('--train_tb', type=bool, help='whether use tensorboard to record loss', default=True)
parser.add_argument('--verbose', type=bool, help='whether save model paremeters in tensorborad', default=False)
parser.add_argument('--model', type=str, help='which model to use', default='basic')
parser.add_argument('--batch_size', type=int, help='training batch_size', default=256)
parser.add_argument('--load_path', type=str, help='parent directory of data', default='../../数据处理/kuaishou')

args = parser.parse_args()

setproctitle.setproctitle("args.description")


def train(model, train_loader, val_loader, lr=0.001, epochs=10, device='cpu', save_path=None):
    """
    Train a model with given train_loader and val_loader.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The DataLoader for training data.
        val_loader (DataLoader): The DataLoader for validation data.
        lr (float, optional): Learning rate. Defaults to 0.001.
        epochs (int, optional): Number of epochs. Defaults to 10.
        device (str, optional): Device to use for training. Defaults to 'cpu'.
        save_path (str, optional): Path to save the best model parameters. Defaults to None.

    Returns:
        model (nn.Module): The trained model.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    best_acc = 0.0
    loss_func = nn.CrossEntropyLoss()
    for epoch in tqdm(range(epochs)):
        model.train()
        for rec_inter_history_s, search_inter_history_s, open_search_inter_history_s, time_features, user_id, label in tqdm(train_loader):
            rec_inter_history_s, search_inter_history_s, open_search_inter_history_s, time_features, user_id, label = \
                rec_inter_history_s.to(device), search_inter_history_s.to(device), open_search_inter_history_s.to(device), time_features.to(device), user_id.to(device), label.to(device).long()
            loss = model.train_(rec_inter_history_s, search_inter_history_s, open_search_inter_history_s, time_features, user_id, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for rec_inter_history_s, search_inter_history_s, open_search_inter_history_s, time_features, user_id, label in val_loader: 
                rec_inter_history_s, search_inter_history_s, open_search_inter_history_s, time_features, user_id, label = \
                    rec_inter_history_s.to(device), search_inter_history_s.to(device), open_search_inter_history_s.to(device), time_features.to(device), user_id.to(device), label.to(device).long()
                output = model.infer_(rec_inter_history_s, search_inter_history_s, open_search_inter_history_s, time_features, user_id)
                y_true += label.cpu().numpy().tolist()
                y_pred += output.argmax(dim=-1).cpu().numpy().tolist()
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            print(f'Epoch {epoch+1}/{epochs}: val accuracy {acc:.4f}, precision {precision:.4f}, recall {recall:.4f}, F1-score {f1:.4f}')

        # 保存最优模型参数
        if acc > best_acc and save_path is not None:
            best_acc = acc
            torch.save(model.state_dict(), save_path)

    return model

if __name__ == '__main__':
    # 加载数据
    valid_data = pd.read_csv(const.valid_file, sep='\t')
    rec_inter_history = pickle.load(open(const.recommendation_index, 'rb'))
    search_inter_history = pickle.load(open(const.search_index, 'rb'))
    open_actions = pickle.load(open(const.open_actions, 'rb'))  
    # 定义模型
    y_true = []
    y_pred = []
    score_final = []
    model = ARIMA_model()
    # 训练模型
    for idx in tqdm(range(len(valid_data))):
        user_id, time, label = valid_data.loc[idx, ['user_id', 'request_time_ms', 'label']]
        open_action_list = [1-a for a in open_actions[(user_id, time)]]
        y_true.append(1-label)
        y_pred.append(model.predict(open_action_list))
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'val accuracy {acc:.4f}, precision {precision:.4f}, recall {recall:.4f}, F1-score {f1:.4f}')


    