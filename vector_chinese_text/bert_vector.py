import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import pickle
from tqdm import tqdm
import setproctitle
setproctitle.setproctitle("dataprocess.jeryi.sunzx")
root_path = "/Users/sunzhongxiang/Desktop/科研/搜推联合/数据处理/kuaishou/raw_data/"
bert_path = "/Users/sunzhongxiang/Desktop/bert-chinese"
# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_path)
model = BertModel.from_pretrained(bert_path)

# 设定设备，GPU加速更快，如果没有GPU则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载数据
df = pd.read_csv(root_path+'session_info_.tsv',sep='\t')
df['keyword_segment'].apply(lambda x:"".join(eval(x)))
search_session_id_list = df['search_session_id'].tolist()
keyword_segment_list = df['keyword'].tolist()

# 用BERT模型编码文本并保存句向量
search_session2embedding = {}
for search_session_id, keyword_segment in tqdm(zip(search_session_id_list, keyword_segment_list)):
    input_ids = torch.tensor(tokenizer.encode(keyword_segment, add_special_tokens=True)).unsqueeze(0).to(device)
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # 只取最后一层输出
        sentence_embedding = torch.mean(last_hidden_states, dim=1).squeeze().cpu().numpy()  # 取词向量的平均值作为句向量
    search_session2embedding[search_session_id] = sentence_embedding

# 保存句向量
with open(root_path+'search_session2embedding.pickle', 'wb') as f:
    pickle.dump(search_session2embedding, f)