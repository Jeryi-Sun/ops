import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import fbeta_score
from dataset.dataset import MyDataset, my_collate_fn
from config import const
from models.transformer import Transformer, ModelArgs
import numpy as np
import random
import os
import setproctitle
from models.model_mean_weight import basic_model_weight_mean
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

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
            f_0_5 = fbeta_score(y_true, y_pred, beta=0.5)
            print(f'Epoch {epoch+1}/{epochs}: val accuracy {acc:.4f}, precision {precision:.4f}, recall {recall:.4f}, F1-score {f1:.4f}, F_0_5-score {f_0_5:.4f}')

        # 保存最优模型参数
        if acc > best_acc and save_path is not None:
            best_acc = acc
            torch.save(model.state_dict(), save_path)

    return model

if __name__ == '__main__':
    # 加载数据
    train_dataset = MyDataset(const.train_file,  max_len_reco=const.max_seq_len_reco, max_len_search=const.max_seq_len_search, max_len_open_search=const.max_seq_len_open_search)
    val_dataset = MyDataset(const.valid_file, max_len_reco=const.max_seq_len_reco, max_len_search=const.max_seq_len_search, max_len_open_search=const.max_seq_len_open_search)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=my_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=128, collate_fn=my_collate_fn)

    """
    增加数据类
    """
    device = 'cuda'+args.gpu_id if torch.cuda.is_available() else 'cpu'

    
    model_args_reco: ModelArgs = ModelArgs(
        max_seq_len=const.max_seq_len_reco, max_batch_size=args.batch_size, vocab_size=const.item_id_num, dim=const.item_id_dim, device=device
    )
    model_args_open_search: ModelArgs = ModelArgs(
        max_seq_len=const.max_seq_len_open_search, max_batch_size=args.batch_size, vocab_size=const.query_id_num, dim=const.query_id_dim, device=device
    )

    model_args_search: ModelArgs = ModelArgs(
        max_seq_len=const.max_seq_len_search, max_batch_size=args.batch_size, vocab_size=const.query_id_num,dim=const.query_id_dim, device=device
    )


    # 定义模型
    model = basic_model_weight_mean(model_args_reco, model_args_search, model_args_open_search)

    # 训练模型

    train(model, train_loader, val_loader, lr=args.lr, epochs=args.epochs, device=device)