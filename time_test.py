import argparse
import logging
import os

import pandas as pd
import torch
from tensorboardX import SummaryWriter

import core.logger as Logger
import core.metrics as Metrics
import data as Data
import model as Model
from decimal import Decimal
import matplotlib.pyplot as plt



def time_test(params, strategy_params, temp_list):
    # 从 params 中获取 test_loader
    test_loader = params['test_loader']
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    opt = params['opt']
    logger = params['logger']
    logger_test = params['logger_test']
    model_epoch = params['model_epoch']

    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    logger.info('Begin Model Evaluation.')
    # ========================================================
    # 1. 运行正常测试（无攻击）
    # ========================================================
    logger.info('Running clean evaluation (no attack)...')
    clean_all_datas = pd.DataFrame()
    with torch.no_grad():
        for _, test_data in enumerate(test_loader):
            diffusion.feed_data(test_data)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals()
            all_data, _, _ = Metrics.tensor2allcsv(visuals, params['col_num'])
            clean_all_datas = Metrics.merge_all_csv(clean_all_datas, all_data)

    # ========================================================
    # 2. 运行对抗攻击测试
    # ========================================================
    logger.info('Running adversarial evaluation...')
    attacked_all_datas = pd.DataFrame()
    # 定义默认攻击参数
    default_attack_params = {
      'epsilon': 0.1,      # 默认扰动幅度
      'alpha': 0.01,       # 默认单步更新步长
      'iterations': 10     # 默认PGD迭代次数
    }

    # 获取用户配置的攻击参数（处理None情况）
    user_attack_params = opt.get('attack_params', {}) or {}

    # 合并参数：用户配置覆盖默认值
    attack_params = {**default_attack_params, **user_attack_params}
    with torch.enable_grad():
        for _, test_data in enumerate(test_loader):
            # 备份原始数据
            original_sr = test_data['SR'].clone().to(diffusion.device)
            target_ori = test_data['ORI'].to(diffusion.device)
            
            # 生成对抗样本
            diffusion.netG.train()  # 强制模型进入训练模式以允许梯度计算
            adversarial_sr = Metrics.generate_adversarial_pgd(
                diffusion.netG, 
                original_sr, 
                target_ori,
                epsilon=attack_params['epsilon'],
                alpha=attack_params['alpha'],
                iterations=attack_params['iterations']
            )
            diffusion.netG.eval()   # 恢复评估模式
            # 替换测试数据中的SR
            test_data['SR'] = adversarial_sr.clamp(0, 1)
            diffusion.feed_data(test_data)
            
            # 关闭梯度以执行正常测试
            with torch.no_grad():
                diffusion.test(continous=False)
                visuals = diffusion.get_current_visuals()
            
            # 保存攻击后数据
            all_data, _, _ = Metrics.tensor2allcsv(visuals, params['col_num'])
            attacked_all_datas = Metrics.merge_all_csv(attacked_all_datas, all_data)

    # ========================================================
    # 3. 后处理与评估
    # ========================================================
    # 截断冗余数据
    for df in [clean_all_datas, attacked_all_datas]:
      if df.shape[0] > params['row_num']:
        df.drop(index=df.index[params['row_num']:], inplace=True)
    
    # 计算F1并输出攻击效果
    clean_f1 = Metrics.relabeling_strategy(clean_all_datas, strategy_params)
    attacked_f1 = Metrics.relabeling_strategy(attacked_all_datas, strategy_params)
    
    # 计算攻击指标
    attack_metrics = Metrics.calculate_attack_impact(clean_all_datas, attacked_all_datas)
    print(
        f"[Attack Impact] Clean F1: {attack_metrics['clean_f1']:.4f} | "
        f"Attacked F1: {attack_metrics['attacked_f1']:.4f} | "
        f"F1 Drop Ratio: {attack_metrics['f1_drop_ratio'] * 100:.2f}%"
    )
    
    # 可视化对比
    Metrics.plot_attack_comparison(clean_all_datas, attacked_all_datas, index=0)
    
    # 打印最终结果
    temp_f1 = Decimal(attacked_f1).quantize(Decimal("0.0000"))
    print('Final F1-score (attacked): ', float(temp_f1))
    print(
    "\n[攻击效果总结]\n"
    f"Clean F1: {stats['clean_f1']:.4f}\n"
    f"Attacked F1: {stats['attacked_f1']:.4f}\n"
    f"F1下降比例: {stats['f1_drop_ratio']*100:.2f}%\n"
    f"Clean平均MSE: {stats['avg_clean_mse']:.4f}\n"
    f"Attacked平均MSE: {stats['avg_attacked_mse']:.4f}\n"
    f"攻击参数: {stats['attack_params']}"
)


# evaluate model performance
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/smap_time_test.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train ', 'val', 'test'],
                        help='Run either train(training) or val(generation)', default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    temp_list = []
    model_epoch = 100

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args, model_epoch)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    logger_name = 'test' 
    # logging
    Logger.setup_logger(logger_name, opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger(logger_name)
    logger.info(Logger.dict2str(opt))
    # 验证日志配置
    logger.info("===== 日志系统已激活 =====")
    logger.info(f"日志文件: {os.path.abspath(os.path.join(opt['path']['log'], 'test.log'))}")
    logger.info(f"当前工作目录: {os.getcwd()}")
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    test_set = Data.create_dataset(opt['datasets']['test'], 'test')

    test_loader = Data.create_dataloader(test_set, opt['datasets']['test'], 'test')
    logger.info('Initial Dataset Finished')
    logger_test = logging.getLogger(logger_name)  # test logger

    start_label = opt['model']['beta_schedule']['test']['start_label']
    end_label = opt['model']['beta_schedule']['test']['end_label']
    step_label = opt['model']['beta_schedule']['test']['step_label']
    step_t = opt['model']['beta_schedule']['test']['step_t']
    strategy_params = {
        'start_label': start_label,
        'end_label': end_label,
        'step_label': step_label,
        'step_t': step_t
    }

    params = {
        'opt': opt,
        'logger': logger,
        'logger_test': logger_test,
        'model_epoch': model_epoch,
        'row_num': test_set.row_num,
        'col_num': test_set.col_num,
        'test_loader': test_loader
    }

    time_test(params, strategy_params, temp_list)
    logging.shutdown()
