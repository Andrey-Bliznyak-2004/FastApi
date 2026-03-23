import laspy
from logging.handlers import RotatingFileHandler
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import KNNGraph
from model import DGCNN_seg
import os
import time

K = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BLOCK_SIZE = 8192
CHECKPOINT_PATH = 'last_model.pth' 

def load_model():
    model = DGCNN_seg(in_channels=3, out_channels=4, k=K)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f'Checkpoint not found: {CHECKPOINT_PATH}')
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

_model = None
_knn_transform = KNNGraph(k=K)

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

def read_las(file_path, load_points=True):
    """
    Читает LAS/LAZ файл.
    Возвращает точки, RGB (нормализованные 0-1) и метаданные.
    """
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    
    
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        rgb_raw = np.vstack((las.red, las.green, las.blue)).T
        if rgb_raw.max() > 255:
            rgb = rgb_raw.astype(np.float32) / 65535.0
        else:
            rgb = rgb_raw.astype(np.float32) / 255.0
    else:
        rgb = np.zeros((len(points), 3), dtype=np.float32)

    metadata = {
        'filename': os.path.basename(file_path),
        'num_points': len(points),
        'bounds': {
            'x': [float(points[:,0].min()), float(points[:,0].max())],
            'y': [float(points[:,1].min()), float(points[:,1].max())],
            'z': [float(points[:,2].min()), float(points[:,2].max())],
        }
    }
    return points, rgb, metadata

def process_block(xyz, rgb):
    """Обрабатывает один блок точек и возвращает метки."""
    if len(xyz) < 512:
        return np.zeros(len(xyz), dtype=np.int64)

    centroid = xyz.mean(axis=0)
    xyz_norm = xyz - centroid
    max_dist = np.linalg.norm(xyz_norm, axis=1).max()
    if max_dist > 0:
        xyz_norm /= max_dist

    pos = torch.from_numpy(xyz_norm).float()
    x = torch.from_numpy(rgb).float()

    data = Data(x=x, pos=pos)
    data = _knn_transform(data)
    batch = Batch.from_data_list([data]).to(DEVICE)

    model = get_model()
    with torch.no_grad():
        out = model(batch)
        pred = out.argmax(dim=1).cpu().numpy()
    return pred

def segment_point_cloud(points, rgb, progress_callback=None):
    """Выполняет сегментацию всего облака и возвращает метки и статистику."""
    n_points = len(points)
    labels = np.zeros(n_points, dtype=np.uint8)

    start_time = time.time() 

    for start in range(0, n_points, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, n_points)
        block_labels = process_block(points[start:end], rgb[start:end])
        labels[start:end] = block_labels

        if progress_callback:
            processed = end
            progress_callback(processed, n_points)  

    unique, counts = np.unique(labels, return_counts=True)
    class_stats = {int(cls): int(count) for cls, count in zip(unique, counts)}

    for cls in range(4):
        if cls not in class_stats:
            class_stats[cls] = 0

    return labels, class_stats

def save_segmented_las(file_path, points, rgb, labels, output_path):
    """Сохраняет сегментированное облако с RGB-раскраской и классификацией"""
    
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scale = [0.01, 0.01, 0.01]
    header.offset = [points[:,0].min(), points[:,1].min(), points[:,2].min()]
    
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
   
    colors = np.zeros((len(labels), 3), dtype=np.uint16)
    colors[labels == 0] = [65535, 0, 0]      
    colors[labels == 1] = [0, 65535, 0]      
    colors[labels == 2] = [0, 0, 65535]      
    colors[labels == 3] = [65535, 65535, 0]  
    
    las.red = colors[:, 0]
    las.green = colors[:, 1]
    las.blue = colors[:, 2]
    
<<<<<<< HEAD
=======
    # Сохраняем метки сегментации в поле user_data
    las.add_extra_dim(laspy.ExtraBytesParams(name="classification", type="u1"))
>>>>>>> 9f1a543dad5b36b2cf35f0a8e297b7bb055c31bf
    las.classification = labels.astype(np.uint8)
    
    las.write(output_path)
    return output_path