import io, json, os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from PIL import Image

RUN_DIR = os.environ.get('RUN_DIR', '/content/drive/MyDrive/visual-search')
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

sess = ort.InferenceSession(f'{RUN_DIR}/model.onnx', providers=['CPUExecutionProvider'])
class_names = json.load(open(f'{RUN_DIR}/splits.json'))['class_names']
gallery = np.load(f'{RUN_DIR}/gallery.npz')
g_emb, g_y, g_idx = gallery['emb'], gallery['y'], gallery['idx']

app = FastAPI(title='Marketplace Visual Search')

def preprocess(data: bytes):
    img = Image.open(io.BytesIO(data)).convert('RGB')
    s = max(img.size)
    canvas = Image.new('RGB', (s, s), (255, 255, 255))
    canvas.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
    arr = np.asarray(canvas.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None]          # HWC -> 1xCxHxW

def run(data: bytes):
    logits, emb = sess.run(None, {'image': preprocess(data)})
    return logits[0], emb[0]

@app.post('/predict')
async def predict(file: UploadFile = File(...), top_k: int = 3):
    logits, _ = run(await file.read())
    probs = np.exp(logits - logits.max()); probs /= probs.sum()
    best = probs.argsort()[::-1][:top_k]
    return {'predictions': [{'label': class_names[i], 'prob': round(float(probs[i]), 4)} for i in best]}

@app.post('/similar')
async def similar(file: UploadFile = File(...), k: int = 5):
    _, emb = run(await file.read())
    sims = g_emb @ emb
    best = sims.argsort()[::-1][:k]
    return {'results': [{'dataset_index': int(g_idx[i]), 'label': class_names[g_y[i]], 'score': round(float(sims[i]), 4)} for i in best]}
