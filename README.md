# Marketplace Visual Search

Product-photo classifier and "find similar items" search, built on a fine-tuned ResNet-50 and served with ONNX Runtime + FastAPI.

Data: [`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small) (44,072 images, 60x80 px, upscaled to 224x224). I kept the 20 most common `articleType` classes, capped at 1,500 images per class, with a stratified 80/10/10 split (seed 42): 17,647 train / 2,199 val / 2,199 test.

## Results (held-out test set)

| Metric | Value |
|---|---|
| Classification accuracy (20 classes) | 0.955 |
| Macro F1 | 0.961 |
| Best validation accuracy | 0.950 (epoch 10) |
| Recall@1 / @5 / @10 | 0.949 / 0.973 / 0.980 |
| Precision@5 | 0.950 |
| ONNX vs PyTorch, max output difference | 3.1e-6 |
| CPU latency (Colab CPU, ONNX Runtime) | ~84 ms / image |

A retrieval hit means the result has the same category as the query, not that it is the same product.

## Approach

- **Model:** `timm` ResNet-50 backbone (ImageNet weights) + Dropout/Linear head. First epoch with the backbone frozen, then the whole network fine-tuned.
- **Training:** 10 epochs on a Colab T4 (~108 s/epoch), AdamW, warmup + cosine LR, mixed precision, label smoothing 0.1, early stopping (patience 3).
- **Preprocessing:** pad to square with white (no cropping), resize to 224, ImageNet normalisation. Augmentation (random resized crop, flip, colour jitter) on train only.
- **Search:** the 2,048-d backbone output is L2-normalised, so cosine similarity is a dot product against the 17,647 training embeddings.
- **Serving:** the model is exported to ONNX (opset 17, dynamic batch) with two outputs, class logits and the normalised embedding. `app.py` runs it with `onnxruntime` only.

## Where the model is wrong

The weakest classes are the ones that look alike:

- Casual Shoes vs Sports Shoes (most errors): similar silhouettes, the difference is often a style label rather than shape.
- Tshirts vs Tops: both are plain upper-body garments.
- Flip Flops vs Sandals: heavy overlap in appearance.

## API

`app.py` expects `model.onnx`, `splits.json` and `gallery.npz` in `RUN_DIR` (default `/content/drive/MyDrive/visual-search`, override with the `RUN_DIR` environment variable). These files are not in git (see `.gitignore`).

```bash
pip install -r requirements.txt
RUN_DIR=/path/to/visual-search uvicorn app:app --port 8000
```

- `POST /predict` (multipart `file`, optional `top_k`): top-k classes with probabilities.
- `POST /similar` (multipart `file`, optional `k`): the k most similar gallery items (`dataset_index`, `label`, cosine `score`).

`dataset_index` is the row number in the Hugging Face dataset, so the image can be looked up with `ds[dataset_index]['image']`.

## Limits

- Tested through FastAPI's in-process `TestClient`; not deployed to a server.
- Retrieval quality is judged by category match, so colour and pose can pull in same-category items that are not the same product.
- The images are 60x80 px, so fine detail (logos, textures) is not available to the model.
