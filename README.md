# LAS/LAZ segmentation API
FastAPI-based service for segmenting LiDAR point clouds (.las/.laz) using DGCNN neural network.
## This service segment only 4 clases:
0. background
1. road
2. buildings
3. vehicles

### Before using this, you need:
- Python 3.8+
- redis server on port 6379

### How to install
* Create virtual environment
python -m venv venv
venv\Scripts\activate

* Install dependencies
pip install -r requirements.txt

* Start Redis

# Run project (recomend use 2 terminals)
### First terminal
- python worker.py
### Second terminal
- uvicorn 1:app --reload
### Work with file
- open in browser <http://127.0.0.1:8000/docs>
- upload you .las/.laz file
- copy the 'task_id'
- past 'task_id' in second block
- after finishing segmentsnion, past 'task_id' in third block to download result file