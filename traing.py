from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.loggers import AnomalibTensorBoardLogger
import matplotlib
matplotlib.use('TkAgg')

# Initialize the datamodule, model and engine
datamodule = MVTec(task='detection',num_workers=0,seed=42)
tensorboard_logger = AnomalibTensorBoardLogger(save_dir='tblogs')
model = Patchcore()
engine = Engine(image_metrics=["AUROC","PRO","F1Score"],task=TaskType.DETECTION,logger=tensorboard_logger)
# Train the model
engine.fit(datamodule=datamodule, model=model)
#results = engine.test(model=model, datamodule=datamodule)
predictions = engine.predict(
    datamodule=datamodule,
    model=model,
    return_predictions = True,
    ckpt_path="results/Patchcore/MVTec/bottle/v10/weights/lightning/model.ckpt",
)


