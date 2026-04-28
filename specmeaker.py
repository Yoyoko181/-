import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
import torch
import scipy.io

def footstep_concatenation(x_train, y_train, footsteps_num):
    """
    Concatenates multiple footsteps into a single training sample.

    Parameters:
    - x_train: np.array
        Input feature matrix of shape (samples, features).
    - y_train: np.array
        Corresponding labels (e.g., participant IDs).
    - footsteps_num: int
        Number of footsteps to concatenate into one sample.

    Returns:
    - dict with keys:
        'data_set': torch.Tensor of concatenated input data.
        'labels_set': torch.Tensor of corresponding labels.
    """

    trn_smpl = x_train.shape[0]
    nm_trn = (trn_smpl // footsteps_num) * footsteps_num
    y_train = y_train[:nm_trn]

    trn_idx = np.arange(nm_trn)
    trn_rsz_nm = nm_trn // footsteps_num
    trn_resized = trn_idx.reshape(trn_rsz_nm, footsteps_num)

    # Prepare labels matrix to check consistency
    y_trn_comb = np.zeros([trn_rsz_nm, footsteps_num])
    for col in range(footsteps_num):
        y_trn_comb[:, col] = y_train[trn_resized[:, col]]

    # Remove rows with mixed labels
    mismatched_elmnts = [
        row_idx for row_idx, row in enumerate(y_trn_comb)
        if len(np.unique(row)) > 1
    ]
    y_trn_comb = np.delete(y_trn_comb, mismatched_elmnts, axis=0)
    trn_resized = np.delete(trn_resized, mismatched_elmnts, axis=0)

    # Concatenate footsteps horizontally
    x_train = torch.Tensor(x_train)
    train_set = x_train[trn_resized[:, 0]]
    for i in range(1, footsteps_num):
        train_set = torch.cat((train_set, x_train[trn_resized[:, i]]), dim=1)

    # Use the first label from consistent rows
    y_train = torch.Tensor(y_trn_comb[:, 0]).type(torch.LongTensor)

    return {'data_set': train_set, 'labels_set': y_train}


def generate_cwt_images(file_path, output_path, footsteps_num=1, scales=np.arange(1, 257), wavelet='morl'):
    """
    Loads footstep data, concatenates footsteps, applies CWT, and saves the resulting images.

    Parameters:
    - file_path: str
        Path to the .mat input file containing footstep features.
    - output_path: str
        Directory to save the CWT image outputs.
    - footsteps_num: int
        Number of footsteps to concatenate per sample.
    - scales: np.array
        Scales used for the wavelet transform.
    - wavelet: str
        Wavelet type (e.g., 'morl', 'cmor').
    """

    # === Load dataset from .mat file ===
    dataset = scipy.io.loadmat(file_path)
    
    # 自动识别变量名
    available_keys = [k for k in dataset.keys() if not k.startswith('__')]
    if 'footstep_feat' in available_keys:
        footstep_dataset = dataset['footstep_feat']
    elif 'person_feat' in available_keys:
        # 如果是 person_feat，它可能是一个 cell array (object array in python)
        raw_data = dataset['person_feat']
        if raw_data.dtype == object:
            print("检测到 cell array (person_feat)，正在自动转换并合并数据...")
            all_feats = []
            all_labels = []
            for i in range(raw_data.shape[1]):
                person_data = raw_data[0, i]
                if person_data.size > 0:
                    all_feats.append(person_data)
                    # 假设 label 是 1-indexed (i+1)
                    all_labels.append(np.full((person_data.shape[0],), i + 1))
            
            features = np.vstack(all_feats)
            labels = np.concatenate(all_labels) - 1
            print(f"转换完成: 合计 {features.shape[0]} 个样本，{len(np.unique(labels))} 个类别。")
        else:
            footstep_dataset = raw_data
            features = footstep_dataset[:, :-1]
            labels = footstep_dataset[:, -1] - 1
    elif len(available_keys) > 0:
        print(f"警告: 未找到标准变量名，尝试使用变量: '{available_keys[0]}'")
        footstep_dataset = dataset[available_keys[0]]
        features = footstep_dataset[:, :-1]
        labels = footstep_dataset[:, -1] - 1
    else:
        raise KeyError("在 .mat 文件中未找到任何有效数据变量。")

    # === Concatenate footsteps ===
    data = footstep_concatenation(features, labels, footsteps_num)
    X, y = data['data_set'], data['labels_set']

    # === Generate CWT images ===
    classes = np.unique(y)
    for class_label in classes:
        class_folder = os.path.join(output_path, str(class_label))
        os.makedirs(class_folder, exist_ok=True)

        class_data = X[y == class_label]
        for index, signal in enumerate(class_data):
            signal = signal.numpy()
            coefficients, _ = pywt.cwt(signal, scales, wavelet)

            plt.imshow(coefficients, cmap='jet', aspect='auto')
            plt.axis('off')
            image_path = os.path.join(class_folder, f'cwt_image_{class_label}_{index}.png')
            print(f"Saving: {image_path}")
            plt.savefig(image_path, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()


if __name__ == "__main__":
    # === Configuration ===
    # 自动获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 指向 model_train 目录下的 .mat 文件
    file_path = os.path.join(current_dir, 'model_train', 'footstep_feat.mat')
    
    # 输出目录设置为 model_train/cwt_images
    output_path = os.path.join(current_dir, 'model_train', 'cwt_images')
    
    footsteps_num = 1  # 默认拼接步数

    print(f"输入文件路径: {file_path}")
    print(f"输出目录路径: {output_path}")

    # === Run the full processing pipeline ===
    if not os.path.exists(file_path):
        print(f"错误: 找不到输入文件 {file_path}")
    else:
        generate_cwt_images(file_path, output_path, footsteps_num)
