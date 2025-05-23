import numpy as np
import pandas as pd

import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, mean_squared_error

def generate_adversarial_pgd(model, original_sr, target_ori, epsilon, alpha, iterations, 
                          momentum_decay=0.9, multiscale_weights=None, sensitivity=1.5):
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
    # 记录原始数据范围
    raw_min = original_sr.min().item()
    raw_max = original_sr.max().item()                      
    # 1. 归一化原始输入
    x_norm, min_val, max_val = minmax_normalize(original_sr)
    
    # 2. 在归一化空间初始化对抗样本
    adversarial_norm = x_norm.clone().requires_grad_(True)
    accumulated_grad = torch.zeros_like(adversarial_norm)
    
    # 3. 预计算原始差异（基于归一化数据）
    with torch.no_grad():
        clean_output = model.super_resolution(
            minmax_denormalize(x_norm, min_val, max_val),
            min_num=raw_min, 
            max_num=raw_max
        )
        clean_differ = (target_ori - clean_output).abs().mean(dim=1, keepdim=True)
    
    # 4. PGD攻击循环
    for _ in range(iterations):
        adversarial_norm.grad = None
        
        # 前向计算（注意反归一化输入模型）
        adv_output = model.super_resolution(
            minmax_denormalize(adversarial_norm, min_val, max_val),
            min_num=raw_min,
            max_num=raw_max
        )
        loss = adversarial_loss(adv_output, target_ori, clean_differ)
        loss.backward()
        
        # 梯度动量计算
        grad = adversarial_norm.grad.data
        accumulated_grad = momentum_decay * accumulated_grad + (1 - momentum_decay) * grad
        
        # 多尺度扰动生成
        base_delta = alpha * accumulated_grad.sign()
        if multiscale_weights is not None:
            base_delta += 0.3 * generate_multiscale_perturbation(base_delta, multiscale_weights)
        
        # 自适应扰动约束
        current_epsilon = adaptive_epsilon_mask(clean_differ, epsilon, sensitivity)
        perturbed_norm = adversarial_norm.data + base_delta
        delta = torch.clamp(perturbed_norm - x_norm, -current_epsilon, current_epsilon)
        
        # 更新并裁剪到[0,1]
        adversarial_norm.data = torch.clamp(x_norm + delta, 0, 1)
    
    # 5. 反归一化得到最终对抗样本
    adversarial_sr = minmax_denormalize(adversarial_norm.detach(), min_val, max_val)
    return adversarial_sr
                            
def minmax_normalize(x: torch.Tensor, eps=1e-8) -> tuple:
    """最大最小归一化到[0,1]范围"""
    min_val = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # [B,C,1,1]
    max_val = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    x_norm = (x - min_val) / (max_val - min_val + eps)
    return x_norm, min_val, max_val

def minmax_denormalize(x_norm: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """从[0,1]范围反归一化到原始尺度"""
    return x_norm * (max_val - min_val) + min_val 
  
def adversarial_loss(model_output, target_ori, clean_differ, lambda_anomaly=2.0):
    """
    组合损失函数：
    - 正常区域：最大化重构误差
    - 异常区域：最小化差异
    """
    current_differ = (target_ori - model_output).abs()
    
    # 正常区域掩码（原始差异低于95%分位数）
    normal_mask = (clean_differ < clean_differ.quantile(0.95)).float()
    loss_normal = - (current_differ * normal_mask).mean()  # 最大化误差
    
    # 异常区域掩码（原始差异高于95%分位数）
    anomaly_mask = (clean_differ >= clean_differ.quantile(0.95)).float()
    loss_anomaly = (current_differ * anomaly_mask).mean()  # 最小化误差
    
    return loss_normal + lambda_anomaly * loss_anomaly
    
def generate_multiscale_perturbation(delta, scales=[0.1, 0.3, 0.5], weights=[0.4, 0.3, 0.3]):
    """
    生成多尺度频域扰动：
    - scales: 频带比例（0-1）
    - weights: 各尺度扰动权重
    """
    device = delta.device  # 获取输入张量的设备信息
    delta_fft = torch.fft.fftn(delta, dim=(-2, -1))
    perturb = torch.zeros_like(delta)
    
    for scale, weight in zip(scales, weights):
        # 创建频域掩码
        h, w = delta.shape[-2:]
        mask = torch.zeros((h, w), device=device)
        cx, cy = h//2, w//2
        radius = int(min(h, w) * scale)
        mask[cx-radius:cx+radius, cy-radius:cy+radius] = 1
        
        # 应用频域滤波
        filtered_fft = delta_fft * mask
        filtered = torch.fft.ifftn(filtered_fft).real
        perturb += weight * filtered
    
    return perturb / sum(weights)
    
def adaptive_epsilon_mask(clean_differ, base_epsilon=0.1, sensitivity=1.5):
    """
    生成自适应扰动幅度掩码：
    - 高差异区域（异常）允许更小扰动
    - 低差异区域（正常）允许更大扰动
    """
    norm_differ = clean_differ / clean_differ.amax(dim=(1,2,3), keepdim=True)
    epsilon_mask = base_epsilon * (1 + sensitivity * (1 - norm_differ))
    return epsilon_mask.clamp(max=2*base_epsilon)




def squeeze_tensor(tensor):
    return tensor.squeeze().cpu()


def update_csv_col_name(all_datas):
    df = all_datas.copy()
    df.columns = [0, 1, 2, 3]

    return df


def tensor2allcsv(visuals, col_num, attack_delta=None):
    """将张量转换为包含完整时间序列的DataFrame"""
    df = pd.DataFrame()
    
    # 提取张量数据并转换为DataFrame
    sr_df = pd.DataFrame(squeeze_tensor(visuals['SR']))
    ori_df = pd.DataFrame(squeeze_tensor(visuals['ORI']))
    lr_df = pd.DataFrame(squeeze_tensor(visuals['LR']))
    inf_df = pd.DataFrame(squeeze_tensor(visuals['INF']))
    
    # 裁剪多余列（确保ori_df和sr_df列数一致）
    if col_num != 1:
        for i in range(col_num, sr_df.shape[1]):
            sr_df.drop(columns=i, axis=1, inplace=True)
            ori_df.drop(columns=i, axis=1, inplace=True)
            lr_df.drop(columns=i, axis=1, inplace=True)
            inf_df.drop(columns=i, axis=1, inplace=True)
    
    # 计算各列均值（标量值）
    df['SR'] = sr_df.mean(axis=1)
    df['ORI'] = ori_df.mean(axis=1)
    df['LR'] = lr_df.mean(axis=1)
    df['INF'] = inf_df.mean(axis=1)
    
    # 计算差异均值（确保ori_df和sr_df维度一致）
    df['differ'] = (ori_df - sr_df).abs().mean(axis=1) 
    # 标签字段
    df['label'] = squeeze_tensor(visuals['label'])
    
    # 生成完整差异DataFrame
    differ_df = ori_df - sr_df
    
    return df, sr_df, differ_df


def merge_all_csv(all_datas, all_data):
    return pd.concat([all_datas, all_data], ignore_index=True)


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
