# utils.py - PyTorch Version
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PWVDataset(Dataset):
    """
    自定义PyTorch Dataset，用于加载和处理PWV数据。
    """
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_and_preprocess_data(filepath, batch_size=32, pwv_min=0, pwv_max=80):
    """
    加载CSV数据，进行预处理，并创建PyTorch DataLoader。
    添加了PWV值筛选功能。

    Args:
        filepath (str): CSV文件的路径。
        batch_size (int): DataLoader的批次大小。
        pwv_min (float): PWV的最小值阈值。
        pwv_max (float): PWV的最大值阈值。

    Returns:
        tuple: 包含DataLoader、Scaler的元组
               (train_loader, test_loader, scaler_X, scaler_y, X_train_shape, X_test_shape)。
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        exit()

    # 1. 特征和标签选择 - 根据新的表头进行修改
    features_cols = [
        'suominet_lat', 'suominet_lon', 'suominet_height', 'suominet_temp',
        'tb_h_18GHz', 'tb_v_18GHz',
        'tb_h_23GHz', 'tb_v_23GHz',
        'tb_h_36GHz', 'tb_v_36GHz'
        # 注意：这里没有89GHz的数据，因为新表头中没有。
        # 如果需要，请确保数据文件包含这些列。
        # 如果您希望总特征数保持10个，可以移除经纬度或海拔/温度中的一个。
        # 目前是10个特征。
    ]
    label_col = 'pwv' # 标签列名改为 'pwv'

    # 检查所有必需的列是否存在
    missing_cols = [c for c in features_cols + [label_col] if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in CSV: {missing_cols}")
        exit()

    # 2. 处理缺失值 (这里简单地删除包含缺失值的行)
    initial_rows = len(df)
    df_cleaned = df.dropna(subset=features_cols + [label_col])
    if len(df_cleaned) < initial_rows:
        print(f"Removed {initial_rows - len(df_cleaned)} rows due to missing values.")

    # 3. 筛选PWV值在指定范围内的行
    pwv_initial_rows = len(df_cleaned)
    df_filtered = df_cleaned[(df_cleaned[label_col] >= pwv_min) & (df_cleaned[label_col] <= pwv_max)]
    if len(df_filtered) < pwv_initial_rows:
        print(f"Removed {pwv_initial_rows - len(df_filtered)} rows because PWV was outside [{pwv_min}, {pwv_max}] range.")

    X = df_filtered[features_cols].copy()
    y = df_filtered[label_col].copy()

    # 4. 数据归一化 (MinMaxScaler将数据缩放到[0, 1]范围)
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # 5. 数据集划分 (训练集和测试集)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.1, random_state=42
    )

    # 6. Reshape数据以适应CNN输入 (batch_size, channels, sequence_length)
    # 这里的X_train.shape[1]现在是10个特征 (经纬度海拔温度 + 6个亮温度)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # 记录形状，以便在模型构建时使用
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape

    print(f"Total samples in original file: {initial_rows}")
    print(f"Samples after cleaning NaN and filtering PWV: {len(df_filtered)}")
    print(f"X_train shape (for PyTorch Conv1d): {X_train.shape}")
    print(f"X_test shape (for PyTorch Conv1d): {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # 7. 创建PyTorch Dataset和DataLoader
    train_dataset = PWVDataset(X_train, y_train)
    test_dataset = PWVDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler_X, scaler_y, X_train_shape, X_test_shape

if __name__ == '__main__':
    # 示例用法
    data_filepath = 'E:/PMO/PWV/RadioSondeCNN/AMSR2_NAGNSS_MatchResult_Interpolated_New_with_Temp_Height_Celsius.csv'
    train_loader, test_loader, scaler_X, scaler_y, _, _ = load_and_preprocess_data(data_filepath)
    print("\nData loading and preprocessing complete. DataLoaders created.")

    # 检查DataLoader的输出形状
    for batch_X, batch_y in train_loader:
        print(f"Batch X shape: {batch_X.shape}")
        print(f"Batch y shape: {batch_y.shape}")
        break