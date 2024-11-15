import torch
from anomalib.metrics import AUROC
import matplotlib.pyplot as plt
true = torch.tensor([0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
pred = torch.tensor([0.98, 0.10, 0.10, 0.80, 0.50, 0.12, 0.10, 0.05, 0.35, 0.10])

auroc = AUROC()
auroc(pred, true)
fig, title = auroc.generate_figure()
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#FigureCanvas(fig)
#plt.show()
print(fig)