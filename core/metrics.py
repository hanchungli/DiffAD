import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    adversarial_sr = original_sr.clone().requires_grad_(True)   # 初始化时启用梯度
    print(f"初始对抗样本梯度追踪状态: {adversarial_sr.requires_grad}")  # 应为True
    accumulated_grad = torch.zeros_like(adversarial_sr)     
    # 预计算原始差异（用于动态调整目标）
    with torch.no_grad():
        # 计算min和max
        min_val = target_ori.min().item()
        max_val = target_ori.max().item()
        clean_output = model.super_resolution(original_sr,
             min_num=target_ori.min().item(),
             max_num=target_ori.max().item(),
             continous=False
             )
        clean_differ = (target_ori - clean_output).abs()       # [B, C, H, W]                 
    for _ in range(iterations):
        # 清零梯度
        adversarial_sr.grad = None
        model.train()  # 切换模型为训练模式
        # 前向计算
        adv_output = model.super_resolution(
            adversarial_sr, 
            min_num=min_val, 
            max_num=max_val
        )
        loss = adversarial_loss(adv_output, target_ori, clean_differ)
        
        # 反向传播获取梯度
        loss.backward()
        grad = adversarial_sr.grad.data
      
        # 梯度归一化（改用L2范数）
        grad_norm = grad / (grad.norm(p=2, dim=(1,2,3), keepdim=True) + 1e-8)

        # === 动量累积 ===
        accumulated_grad = momentum_decay * accumulated_grad + (1 - momentum_decay) * grad
        
        # === 多尺度扰动生成 ===
        base_delta = alpha * accumulated_grad.sign()
        if multiscale_weights is not None:
            multiscale_perturb = generate_multiscale_perturbation(
                base_delta, 
                scales=[0.1, 0.3, 0.5], 
                weights=multiscale_weights
            )
            total_delta = base_delta + 0.3 * multiscale_perturb  # 比例可调
        else:
            total_delta = base_delta
        
        # === 自适应约束应用 ===
        current_epsilon = adaptive_epsilon_mask(
            clean_differ, 
            base_epsilon=epsilon, 
            sensitivity=sensitivity
        )  # [B, 1, 1, 1]
        
        # 扰动裁剪与更新
        perturbed_data = adversarial_sr.data + total_delta
        delta = torch.clamp(perturbed_data - original_sr.data,
                           min=-current_epsilon, 
                           max=current_epsilon)
        adversarial_sr.data = original_sr.data + delta.detach()
        adversarial_sr.data.clamp_(0, 1)
        
        # 日志输出
        print(f"Iter {_+1}: Loss={loss.item():.2f} | Max Delta={delta.abs().max().item():.4f}")
    
    return adversarial_sr.detach()
    
def adversarial_loss(model_output, target_ori, clean_differ, lambda_anomaly=0.7):
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

def calculate_attack_impact(clean_df, attacked_df, attack_params=None):
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
    # 计算MSE
    mse_clean = mean_squared_error(clean_df['ORI'], clean_df['SR'])
    mse_attacked = mean_squared_error(attacked_df['ORI'], attacked_df['SR'])
    return {
        'clean_f1': clean_f1,
        'attacked_f1': attacked_f1,
        'f1_drop_ratio': drop_ratio,
        'mse_clean': mse_clean,
        'mse_attacked': mse_attacked,
        'attack_params': attack_params
    }

def plot_attack_comparison(clean_data, attacked_data, index=0, save_dir='results/visualization'):
    """绘制原始/对抗样本对比图"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查索引有效性
    if index >= len(clean_data) or index >= len(attacked_data):
        raise ValueError(f"索引{index}超出数据范围 (clean长度={len(clean_data)}, attacked长度={len(attacked_data)})")
    
    plt.figure(figsize=(15, 5))
    
    # 提取时间序列数据
    clean_ori = np.array(clean_data.iloc[index]['ORI'])
    clean_sr = np.array(clean_data.iloc[index]['SR'])
    attacked_ori = np.array(attacked_data.iloc[index]['ORI'])
    attacked_sr = np.array(attacked_data.iloc[index]['SR'])
    
    # 子图1: 原始样本对比
    plt.subplot(131)
    plt.plot(clean_ori, label='Original', color='blue', linewidth=1)
    plt.plot(clean_sr, label='Clean SR', linestyle='--', color='green', linewidth=1)
    plt.title("Clean Sample")
    plt.legend()
    
    # 子图2: 对抗样本对比
    plt.subplot(132)
    plt.plot(attacked_ori, label='Original', color='blue', linewidth=1)
    plt.plot(attacked_sr, label='Adversarial SR', linestyle='--', color='red', linewidth=1)
    plt.title("Adversarial Sample")
    plt.legend()
    
    # 子图3: 异常分数对比
    plt.subplot(133)
    plt.plot(clean_data['differ'].iloc[index], label='Clean Differ', alpha=0.7)
    plt.plot(attacked_data['differ'].iloc[index], label='Attacked Differ', alpha=0.7)
    plt.title("Anomaly Score Comparison")
    plt.legend()
    
    # 保存并关闭
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'attack_comparison_idx_{index}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # 显式关闭图形释放内存
    return save_path
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
