import torch
import os
import numpy as np
from models.shufflenet_v2 import shufflenet_v2_x1_0
from torchvision.transforms import transforms
from PIL import Image
import collections

# 2D-Plot
plot_style = dict(marker='o',
                markersize=4,
                linestyle='-',
                lw=2)

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
            'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
            'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
            'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
            'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
            'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
            'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
            'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
            'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
            }

def get_data(i, dataset):
    img_name = dataset[0][i]
    label    = dataset[1][i]
    label = np.array(label)
    return img_name, label

def NMEloss(output, label):
    dis = output-label
    dis = np.sqrt(np.sum(np.square(dis),1))
    loss = np.mean(dis) / 384
    return loss

def hm2coor(hm):
    coor = []
    test = np.array(hm)
    for i in range(68):
        coor.append(np.where(test[0][i] == np.max(test[0][i]))[0][0])
        coor.append(np.where(test[0][i] == np.max(test[0][i]))[1][0])
    coor = np.array(coor).reshape(68,2)*4
    return coor

def result(input, hm2coor_method = 0):
    input2 = test_transform(input).view(1,3,384,384)
    input2 = input2.to(device)
    output = model(input2).detach()
    output = output.cpu().numpy()

    if hm2coor_method == 0:
        output2 = hm2coor(output)

    return output2.astype('float32')

path = 'save_dir/ShuffleNet/best_model16_4.pt'
model = shufflenet_v2_x1_0()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.load_state_dict(torch.load(path))
model.to(device)
model.eval()

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
test_transform = transforms.Compose([
            transforms.GaussianBlur(3, 2),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])

dir_path = 'data/aflw_test/'
dirs = os.listdir(dir_path)

for i,img in enumerate(dirs):

    image = Image.open(dir_path+'/'+img)
    output = result(image, hm2coor_method= 0)
    
    output = output.reshape((136))


    txt_path = 'result.txt'
    f = open(txt_path, 'a')
    f.write(str(img))
    f.write(' ')
    for i in range(135):
        f.write(str(output[i]))
        f.write(' ')
    f.write(str(output[135]))
    f.write('\n')
    f.close()