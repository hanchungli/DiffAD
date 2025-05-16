import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, mean_squared_error

def generate_adversarial_pgd(model, original_sr, target_ori, epsilon=0.05, alpha=0.01, iterations=20):
    """
    基于PGD生成对抗样本
    :param model: 扩散模型
    :param original_sr: 原始低分辨率输入 [B, C, H, W]
    :param target_ori: 目标高分辨率数据（用于计算损失）
    :param epsilon: 扰动最大幅度（L∞范数约束）
    :param alpha: 单步更新步长
    :param iterations: 迭代次数
    :return: 对抗样本
    """
    adversarial_sr = original_sr.clone().requires_grad_(True)   # 初始化时启用梯度
    
    for _ in range(iterations):
        # 清零梯度
        if adversarial_sr.grad is not None:
            adversarial_sr.grad.zero_()
        model.train()  # 切换模型为训练模式
        # 强制启用梯度计算
        with torch.enable_grad():
            model_output = model.super_resolution(
                adversarial_sr,
                min_num=target_ori.min().item(),
                max_num=target_ori.max().item(),
                continous=False
            )
            loss = torch.nn.functional.l1_loss(model_output, target_ori.view_as(model_output))
        
        # 反向传播获取梯度
        loss.backward()
        grad = adversarial_sr.grad.data
        # 打印梯度和损失变化
        print(f"Iter {_+1}/{iterations} | Loss: {loss.item():.4f} | Grad Norm: {grad.norm().item():.4f}")
        # 更新对抗样本（直接操作 .data 避免断开计算图）
        adversarial_sr.data = adversarial_sr.data + alpha * grad.sign()
        
        # 单步更新对抗样本
        perturbed_data = adversarial_sr.data + alpha * grad.sign()
        delta = torch.clamp(perturbed_data - original_sr.data, min=-epsilon, max=epsilon)
        adversarial_sr.data = original_sr.data + delta
        adversarial_sr.data.clamp_(0, 1)
    
    return adversarial_sr.detach()
def calculate_attack_impact(clean_df, attacked_df):
    """
    计算对抗攻击对异常检测的影响
    :param clean_df: 原始测试结果DataFrame
    :param attacked_df: 对抗攻击后的测试结果DataFrame
    :return: 攻击指标字典（F1下降比例、误报率变化等）
    """
    # 计算原始F1
    clean_f1 = f1_score(clean_df['label'], clean_df['differ'] > clean_df['differ'].quantile(0.95))
    
    # 计算攻击后F1
    attacked_f1 = f1_score(attacked_df['label'], attacked_df['differ'] > attacked_df['differ'].quantile(0.95))
    
    # 计算下降比例
    drop_ratio = (clean_f1 - attacked_f1) / clean_f1
    
    return {
        'clean_f1': clean_f1,
        'attacked_f1': attacked_f1,
        'f1_drop_ratio': drop_ratio
    }


def squeeze_tensor(tensor):
    return tensor.squeeze().cpu()


def update_csv_col_name(all_datas):
    df = all_datas.copy()
    df.columns = [0, 1, 2, 3]

    return df


def tensor2allcsv(visuals, col_num):
    df = pd.DataFrame()
    sr_df = pd.DataFrame(squeeze_tensor(visuals['SR']))
    ori_df = pd.DataFrame(squeeze_tensor(visuals['ORI']))
    lr_df = pd.DataFrame(squeeze_tensor(visuals['LR']))
    inf_df = pd.DataFrame(squeeze_tensor(visuals['INF']))

    if col_num != 1:
        for i in range(col_num, sr_df.shape[1]):
            sr_df.drop(labels=i, axis=1, inplace=True)
            ori_df.drop(labels=i, axis=1, inplace=True)
            lr_df.drop(labels=i, axis=1, inplace=True)
            inf_df.drop(labels=i, axis=1, inplace=True)

    df['SR'] = sr_df.mean(axis=1)
    df['ORI'] = ori_df.mean(axis=1)
    df['LR'] = lr_df.mean(axis=1)
    df['INF'] = inf_df.mean(axis=1)

    df['differ'] = (ori_df - sr_df).abs().mean(axis=1)
    df['label'] = squeeze_tensor(visuals['label'])

    differ_df = (sr_df - ori_df)

    return df, sr_df, differ_df


def merge_all_csv(all_datas, all_data):
    all_datas = pd.concat([all_datas, all_data])
    return all_datas


def save_csv(data, data_path):
    data.to_csv(data_path, index=False)


def get_mean(df):
    mean = df['value'].astype('float32').mean()
    normal_mean = df['value'][df['label'] == 0].astype('float32').mean()
    anomaly_mean = df['value'][df['label'] == 1].astype('float32').mean()

    return mean, normal_mean, anomaly_mean


def get_val_mean(df):
    mean_dict = {}

    ori = 'ORI'
    ori_mean = df[ori].astype('float32').mean()
    ori_normal_mean = df[ori][df['label'] == 0].astype('float32').mean()
    ori_anomaly_mean = df[ori][df['label'] == 1].astype('float32').mean()

    gen_mean = df['SR'].astype('float32').mean()
    gen_normal_mean = df['SR'][df['label'] == 0].astype('float32').mean()
    gen_anomaly_mean = df['SR'][df['label'] == 1].astype('float32').mean()

    mean_dict['MSE'] = mean_squared_error(df[ori], df['SR'])

    mean_dict['ori_mean'] = ori_mean
    mean_dict['ori_normal_mean'] = ori_normal_mean
    mean_dict['ori_anomaly_mean'] = ori_anomaly_mean

    mean_dict['gen_mean'] = gen_mean
    mean_dict['gen_normal_mean'] = gen_normal_mean
    mean_dict['gen_anomaly_mean'] = gen_anomaly_mean

    mean_dict['mean_differ'] = ori_mean - gen_mean
    mean_dict['normal_mean_differ'] = ori_normal_mean - gen_normal_mean
    mean_dict['anomaly_mean_differ'] = ori_anomaly_mean - gen_anomaly_mean

    mean_dict['ori_no-ano_differ'] = ori_normal_mean - ori_anomaly_mean
    mean_dict['ori_mean-no_differ'] = ori_mean - ori_normal_mean
    mean_dict['ori_mean-ano_differ'] = ori_mean - ori_anomaly_mean

    mean_dict['gen_no-ano_differ'] = gen_normal_mean - gen_anomaly_mean
    mean_dict['gen_mean-no_differ'] = gen_mean - gen_normal_mean
    mean_dict['gen_mean-ano_differ'] = gen_mean - gen_anomaly_mean

    return mean_dict


def relabeling_strategy(df, params):
    y_true = []
    best_N = 0
    best_f1 = -1
    best_thred = 0
    best_predictions = []
    thresholds = np.arange(params['start_label'], params['end_label'], params['step_label'])

    df_sort = df.sort_values(by="differ", ascending=False)
    df_sort = df_sort.reset_index(drop=False)

    for t in thresholds:
        # if (t - 1) % params['step_t'] == 0:
        #     print("t: ", t)
        y_true, y_pred, thred = predict_labels(df_sort, t)
        for i in range(len(y_true)):
            if y_pred[i] == 1 and y_true[i] == 1:
                j = i - 1
                while j >= 0 and y_true[j] == 1 and y_pred[j] == 0:
                    y_pred[j] = 1
                    j -= 1
                j = i + 1
                while j < len(y_pred) and y_true[j] == 1 and y_pred[j] == 0:
                    y_pred[j] = 1
                    j += 1

        f1 = calculate_f1(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_N = t
            best_thred = thred
            best_predictions = y_pred

    accuracy = calculate_accuracy(y_true, best_predictions)
    precision = calculate_precision(y_true, best_predictions)
    recall = calculate_recall(y_true, best_predictions)

    return best_f1


def predict_labels(df_sort, num):
    df_sort['pred_label'] = 0
    df_sort.loc[0:num - 1, 'pred_label'] = 1
    thred = df_sort.loc[num - 1, 'differ']

    df_sort = df_sort.set_index('index')
    df_sort = df_sort.sort_index()

    y_true = df_sort['label'].tolist()
    y_pred = df_sort['pred_label'].tolist()

    return y_true, y_pred, thred


def calculate_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def calculate_precision(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    return precision


def calculate_recall(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    return recall


def calculate_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    return f1
