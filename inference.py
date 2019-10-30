from PIL import Image
import torchvision.transforms.functional as TF
from torch import load as tload
from torch import max as tmax
from torchvision.transforms import Scale, CenterCrop, ToTensor, Normalize
from torch.autograd import Variable

# Paths
model_path = './CIR_estimator.mdl'
image_path = './example_images/CIR4_example.jpg'

# Load the model
cir_est=tload(model_path).cpu()
cir_est.train(False)

# Load the image
image = Image.open(image_path)
image = Scale(256)(image)
image = CenterCrop(224)(image)
image = ToTensor()(image)
image = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
image.unsqueeze_(0)

# Calculate the CIR pestimation
_, cir_score = tmax(cir_est(Variable(image)).data, 1)
cir_score = cir_score.numpy()[0]+1
print('Estimated CIR score is %d' %cir_score)