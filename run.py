# run.py - PyTorch Version
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 导入自定义模块
from utils import load_and_preprocess_data
from CNN import PWVCNN

def rmse_metric(predictions, targets):
    """计算RMSE"""
    return torch.sqrt(torch.mean((predictions - targets)**2))

def main():
    # 数据文件路径 - 已修改为新的路径
    data_filepath = 'E:/PMO/PWV/RadioSondeCNN/AMSR2_NAGNSS_MatchResult_Interpolated_New_with_Temp_Height_Celsius.csv'
    batch_size = 64
    epochs = 3000 # 论文中提到不到3000次迭代

    # 1. 加载和预处理数据
    train_loader, test_loader, scaler_X, scaler_y, X_train_shape, X_test_shape = \
        load_and_preprocess_data(data_filepath, batch_size)

    # 确定输入特征数量
    # X_train_shape 是 (samples, channels, features_length)
    input_features = X_train_shape[2] # 对应sequence_length

    # 2. 构建PyTorch CNN模型
    model = PWVCNN(input_features=input_features)
    print("\nModel Architecture:")
    print(model)

    # 3. 定义损失函数和优化器
    criterion = nn.MSELoss() # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam优化器

    # 4. 训练模型
    print("\nStarting model training...")
    train_losses = []
    val_losses = []
    train_rmses = []
    val_rmses = []

    for epoch in range(epochs):
        model.train() # 设置模型为训练模式
        running_loss = 0.0
        running_rmse = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad() # 梯度清零
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward() # 反向传播
            optimizer.step() # 更新权重

            running_loss += loss.item() * inputs.size(0)
            running_rmse += rmse_metric(outputs, targets).item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_rmse = running_rmse / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_rmses.append(epoch_rmse)

        # 在测试集上评估 (作为验证)
        model.eval() # 设置模型为评估模式
        val_running_loss = 0.0
        val_running_rmse = 0.0
        with torch.no_grad(): # 不计算梯度
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_rmse += rmse_metric(outputs, targets).item() * inputs.size(0)

        val_loss = val_running_loss / len(test_loader.dataset)
        val_rmse = val_running_rmse / len(test_loader.dataset)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)

        if (epoch + 1) % 100 == 0: # 每100个epoch打印一次
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {epoch_loss:.4f}, Train RMSE: {epoch_rmse:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
    print("Model training complete.")

    # 5. 评估模型 (在测试集上进行最终评估)
    model.eval() # 设置模型为评估模式
    all_predictions_scaled = []
    all_targets_scaled = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            all_predictions_scaled.append(outputs.cpu().numpy())
            all_targets_scaled.append(targets.cpu().numpy())

    y_pred_scaled = np.concatenate(all_predictions_scaled)
    y_test_scaled = np.concatenate(all_targets_scaled)

    # 反归一化
    y_test_original = scaler_y.inverse_transform(y_test_scaled)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)

    # 计算R2, RMSE, MAE (使用原始尺度数据)
    r2 = r2_score(y_test_original, y_pred_original)
    rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae_original = mean_absolute_error(y_test_original, y_pred_original)

    print(f"\n--- Final Evaluation on Original Scale (Test Set) ---")
    print(f"R^2 Score: {r2:.4f}")
    print(f"RMSE (Original Scale): {rmse_original:.4f} mm")
    print(f"MAE (Original Scale): {mae_original:.4f} mm")

    # 6. 可视化训练历史
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Test Loss') # 在PyTorch中，通常将测试集作为验证集
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_rmses, label='Train RMSE')
    plt.plot(val_rmses, label='Test RMSE')
    plt.title('Model RMSE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 7. 绘制预测值与真实值的散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_original, y_pred_original, alpha=0.5, s=10)
    plt.plot([min(y_test_original), max(y_test_original)],
             [min(y_test_original), max(y_test_original)],
             '--r', linewidth=2, label='1:1 line')
    plt.xlabel('True PWV (mm)')
    plt.ylabel('Predicted PWV (mm)')
    plt.title('True vs. Predicted PWV')
    plt.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse_original:.3f} mm',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box') # 保持x轴和y轴比例一致
    plt.show()

if __name__ == '__main__':
    main()