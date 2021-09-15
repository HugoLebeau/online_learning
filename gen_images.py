import argparse
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample, save_as_images)

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=10, metavar="BATCHSIZE")
parser.add_argument('--n-images', type=int, default=10, metavar="NIMAGES")
parser.add_argument('--im-class', type=str, default='mushroom', metavar="CLASS")
parser.add_argument('--truncation', type=float, default=1, metavar="TRUNCATION")
parser.add_argument('--file', type=str, default='.', metavar="FILE")
parser.add_argument('--seed', type=int, default=1, metavar="SEED")
args = parser.parse_args()

path = args.file+'/'+args.im_class+"_{}"
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

model_gan = BigGAN.from_pretrained('biggan-deep-256')
model_vgg = models.vgg19(pretrained=True)
if use_cuda:
    model_gan.to('cuda')
    model_vgg.to('cuda')
model_gan.eval()
model_vgg.eval()

features_file = open(args.file+'/'+args.im_class+'_features.csv', 'w')

scale = transforms.Resize((224, 224)) # VGG input size
norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

class_vector = one_hot_from_names(args.im_class, batch_size=args.batch_size)
class_vector = torch.from_numpy(class_vector)

for ite in tqdm(range(args.n_images//args.batch_size)):
    noise_vector = truncated_noise_sample(truncation=args.truncation, batch_size=args.batch_size)
    noise_vector = torch.from_numpy(noise_vector)
    
    if use_cuda:
        noise_vector = noise_vector.to('cuda')
        class_vector = class_vector.to('cuda')
    
    # Generate an image
    with torch.no_grad():
        image = model_gan(noise_vector, class_vector, args.truncation)
        features = model_vgg.features(norm(scale(image)))
    
    if use_cuda:
        image = image.to('cpu')
        features = features.to('cpu')
    
    # Save results
    save_as_images(image, file_name=path.format(ite))
    np.savetxt(features_file, torch.flatten(features, 1, -1).numpy())

features_file.close()
