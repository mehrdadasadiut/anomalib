import pdb
# Import the required modules
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
#results = engine.test(model=model, datamodule=datamodule)
predictions = engine.predict(
    datamodule=datamodule,
    model=model,
    #I am enabling return_prediction
    return_predictions = True,
    ckpt_path="results/Patchcore/MVTec/bottle/v5/weights/lightning/model.ckpt",
)


def label_images(predictions, indices):
    """Interactive loop to label uncertain images."""
    for i in indices:
        idx = i.item()
        img_tensor = predictions[0]['image'][idx]
        score = predictions[0]['score'][idx]
        image_path = predictions[0]['image_path'][idx]
        
        # Display image
        show_image(img_tensor, score, image_path)
        
        # Get user input for labeling
        
def unnormalize(img_tensor, mean, std):
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)  # Unnormalize by multiplying with std and adding mean
    return img_tensor

#Use the predictions to add uncertain samples for labeling. Which samples? those which model is not confident enough.
def show_image(img_tensor,i,score):
    # Unnormalize the image
    mean = torch.tensor([0.485, 0.456, 0.406])  # ImageNet mean
    std = torch.tensor([0.229, 0.224, 0.225])  # ImageNet std
    img_tensor = unnormalize(img_tensor.clone(), mean, std)
    # Convert from (C, H, W) to (H, W, C) for displaying
    img_tensor = img_tensor.permute(1, 2, 0)
    # Clip the values to be in range [0, 1] for display
    img_tensor = torch.clamp(img_tensor, 0, 1)
    plt.imshow(img_tensor)
    plt.title(f'Image {i}:, Normallity Score: {score}')
    plt.axis('off')
    plt.show(block=False)
    label = input(f"Label this image with Normallity Score: {score}) as [1] Good or [0] Not Normal: ")
    while label not in ['0', '1']:
            label = input("Invalid input. Please enter ''1'' for Good or ''0'' for Not Normal: ")
    plt.close()
    #plt.imsave('image {}.png'.format(i), img_tensor.numpy())

def confusion_matrix(predictions):
    label = predictions[0]['label']
     # Convert pred_labels from boolean to integer
    pred_labels = predictions[0]['pred_labels'].int()

    # Calculate components of confusion matrix
    TP = ((label == 1) & (pred_labels == 1)).sum().item() #True Positive - Correctly found as analomous
    TN = ((label == 0) & (pred_labels == 0)).sum().item() #True Negative - Correctly found as normal
    FP = ((label == 0) & (pred_labels == 1)).sum().item() #Incorrectly found as Anolomous
    FN = ((label == 1) & (pred_labels == 0)).sum().item() #Incorrectly found as normal

    # Display the confusion matrix
    confusion_matrix = {
        "True Positive": TP,
        "True Negative": TN,
        "False Positive": FP,
        "False Negative": FN
    }
    return confusion_matrix

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
confusion_m = confusion_matrix(predictions)
print(confusion_m)
#pdb.set_trace()
scores = predictions[0]['pred_scores']
print(predictions[0]['label'])
print(scores)
#pdb.set_trace()
#values, indices = torch.topk(scores, 5, largest=False)
#print("Smallest values:", values)
#print("Indices", indices)
#for i in indices:
#    print(i.item())
#    print('ImagePaths:',predictions[0]['image_path'][i.item()])
#    show_image(predictions[0]['image'][i.item()],i.item(),predictions[0]['pred_scores'][i.item()])



