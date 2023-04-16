
import numpy as np
class rule_model:
    def __init__(self) -> None:
        pass
    def predict(self, reco_list, search_list, open_action_list, threshold1, threshold2):
        reco_count = len(reco_list)
        search_count = len(search_list)
        total_count = reco_count + search_count

        # 计算搜索和浏览推荐视频的比例
        search_ratio = search_count / total_count
        open_search_ratio = np.count_nonzero(np.array(open_action_list) == 1) / len(open_action_list)


        # 如果搜索次数与总行为次数的比例大于阈值，则预测用户下一次会进行搜索；反之，则预测用户下一次会浏览推荐视频
        if search_ratio > threshold1 or open_search_ratio>threshold2:
            return 1
        else:
            return 0