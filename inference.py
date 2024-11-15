from anomalib.engine import Engine
from anomalib.data import MVTec
from anomalib.models import Patchcore

# Assuming the datamodule, model and engine is initialized from the previous step,
# a prediction via a checkpoint file can be performed as follows:
datamodule = MVTec()
engine = Engine()
model = Patchcore()
predictions = engine.predict(
    datamodule=datamodule,
    model=model,
    #I am enabling return_prediction
    return_predictions = True,
    ckpt_path="results/Patchcore/MVTec/bottle/v0/weights/lightning/model.ckpt",
)
print(predictions[0]['pred_scores'])