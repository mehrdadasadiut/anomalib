import torch
# Import the required modules
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.loggers import AnomalibTensorBoardLogger
# Initialize the datamodule, model and engine
datamodule = MVTec(task='classification',num_workers=0,seed=42)
tensorboard_logger = AnomalibTensorBoardLogger(save_dir='tblogs')
model = Patchcore()
engine = Engine(image_metrics=["AUROC","PRO","F1Score"],task=TaskType.CLASSIFICATION,logger=tensorboard_logger)
# Train the model
engine.fit(datamodule=datamodule, model=model)

all_embeddings = torch.cat(model.embeddings, dim=0).cpu().numpy()
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
tsne_result = tsne.fit_transform(all_embeddings)
# Step 4: Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
plt.title("t-SNE visualization of latent space embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

#results = engine.test(model=model, datamodule=datamodule)
#predictions = engine.predict(
#    datamodule=datamodule,
#    model=model,
    #I am enabling return_prediction
#    return_predictions = True,
#    ckpt_path="results/Patchcore/MVTec/bottle/v5/weights/lightning/model.ckpt",
#)