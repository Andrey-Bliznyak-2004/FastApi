import laspy
from logging.handlers import RotatingFileHandler
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import KNNGraph
from model import DGCNN_seg
import os
import time
import plotly.graph_objs as go
import io 
import logging
from logging.handlers import RotatingFileHandler
K = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BLOCK_SIZE = 8192
CHECKPOINT_PATH = 'last_model.pth' 
from minio import Minio
from minio.error import S3Error
logger = logging.getLogger("Minio")
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    "app.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

def load_model_from_minio(bucket_name='models', object_name='last_model.pth'):
    client = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
    
    try:
        
       
        if not client.bucket_exists(bucket_name):
            logger.info(f"Bucket '{bucket_name}' does not exist")
            raise FileNotFoundError(f"Bucket '{bucket_name}' does not exist")
        logger.info(f"Bucket '{bucket_name}' exists")
        
        # Проверка наличия объекта
        try:
            stats = client.stat_object(bucket_name, object_name)
            logger.info(f"Object found: size={stats.size} bytes")
        except S3Error as e:
            logger.error(f"Object '{object_name}' not found: {e}")
            raise
        
        # Загрузка
        logger.info(f"Downloading {bucket_name}/{object_name}...")
        response = client.get_object(bucket_name, object_name)
        model_bytes = response.read()
        response.close()
        response.release_conn()
        logger.info(f"Downloaded {len(model_bytes)} bytes")
        
        # Загрузка в PyTorch
        logger.info("Loading model from bytes...")
        checkpoint = torch.load(io.BytesIO(model_bytes), map_location=DEVICE)
        
        # Проверка содержимого
        if 'model_state_dict' in checkpoint:
            logger.info("Found 'model_state_dict' in checkpoint")
        else:
            logger.info("No 'model_state_dict' in checkpoint, using as is")
        
        model = DGCNN_seg(in_channels=3, out_channels=4, k=K)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(DEVICE)
        model.eval()
        
        logger.info(f"Model successfully loaded from MinIO")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model from MinIO: {type(e).__name__}: {e}")
        raise

def load_model():
    """Загрузка из локального файла (как fallback)."""
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
        try:
            _model = load_model_from_minio()
        except Exception:
            logger.error("MinIO model not available, trying local file...")
            _model = load_model()
    return _model

def read_las(file_path, load_points=True):
    """
    Читает LAS/LAZ файл.
    Возвращает точки, RGB (нормализованные 0-1) и метаданные.
    """
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)

    # Чтение RGB
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        rgb_raw = np.vstack((las.red, las.green, las.blue)).T
        if rgb_raw.max() > 255:
            rgb = rgb_raw.astype(np.float32) / 65535.0
        else:
            rgb = rgb_raw.astype(np.float32) / 255.0
    else:
        rgb = np.zeros((len(points), 3), dtype=np.float32)
        logger.info('Warning: No RGB data found, using zeros')

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
    """Сохраняет сегментированное облако в LAS/LAZ файл"""
    import laspy

    # Создаем новый LAS файл
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scale = [0.01, 0.01, 0.01]
    header.offset = [points[:,0].min(), points[:,1].min(), points[:,2].min()]

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # Сохраняем RGB
    colors = np.zeros((len(labels), 3), dtype=np.uint16)
    colors[labels == 0] = [65535, 0, 0]      
    colors[labels == 1] = [0, 65535, 0]      
    colors[labels == 2] = [0, 0, 65535]      
    colors[labels == 3] = [65535, 65535, 0]  
    las.red = colors[:, 0]
    las.green = colors[:, 1]
    las.blue = colors[:, 2]
    # Сохраняем метки сегментации в поле user_data
    # (или в дополнительное поле, если нужно)
    las.add_extra_dim(laspy.ExtraBytesParams(name="classification", type="u1"))
    las.classification = labels.astype(np.uint8)

    las.write(output_path)
def generate_plotly_html(file_path, max_points=500000):
    try:
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).T
        
        colors = np.vstack((las.red, las.green, las.blue)).T
        
        if colors.max() > 255:
            colors = (colors / 256).astype(np.uint8)
            
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            colors = colors[indices]

        marker_colors = [f'rgb({c[0]}, {c[1]}, {c[2]})' for c in colors]

        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=marker_colors,
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title="Segmented Point Cloud",
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='data' # Keeps the true proportions of the point cloud
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig.to_html(include_plotlyjs='cdn', full_html=True)
        
    except Exception as e:
        logger.error(f"Error generating Plotly visualization: {e}")
        return None