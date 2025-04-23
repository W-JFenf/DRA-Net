import PIL
import torch, torchvision
from torch.utils.data import DataLoader,Dataset
import numpy as np
from torch import nn
from skimage import io
import os
import argparse
from PIL import Image
from .utils.utils  import AverageMeter
from tqdm import tqdm
from .utils.metrics import iou_score2,dice_score,f1_scorex2
from collections import OrderedDict
from model.DRA_NET import DRA_net





predimg = []
predimg_color = []
labelimg = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="UNET",
                        help='model name: Modified_UNET', choices=["DRA-net"])
    config = parser.parse_args()
    return config


def add_alpha_channel(img,fac):
    img = Image.open(img)
    img = img.convert('RGBA')
    # 更改图像透明度
    factor = fac
    img_blender = Image.new('RGBA', img.size, (0, 0, 0, 0))
    img = Image.blend(img_blender, img, factor)
    return img


def image_together(image, layer, save_path, save_name):
    layer = layer
    base = image
    # bands = list(layer.split())
    heigh, width = layer.size
    for i in range(heigh):
        for j in range(width):
            r, g, b, a = layer.getpixel((i, j))
            if r == 0 and g == 0 and b == 0:
                layer.putpixel((i, j), (0, 0, 0, 0))
            if r == 255 and g == 0 and b == 0:
                layer.putpixel((i, j), (255, 0, 0, 0))
            if r == 0 and g == 255 and b == 0:
                layer.putpixel((i, j), (0, 255, 0, 0))
            if r == 0 and g == 0 and b == 255:
                layer.putpixel((i, j), (0, 0, 255, 0))
    base.paste(layer, (0, 0), layer)  # 贴图操作
    base.save(save_path + "/" + save_name + ".png")  # 图片保存

class MyData(Dataset):
    def __init__(self, root_dir, label_dir, transformers = None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.image_path = os.listdir(self.root_dir)
        self.label_path = os.listdir(self.label_dir)
        self.transformers = transformers
    def __getitem__(self, idx):  #如果想通过item去获取图片，就要先创建图片地址的一个列表
        img_name = self.image_path[idx]
        label_name = self.label_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)  # 每个图片的位置
        label_item_path = os.path.join(self.label_dir, label_name)
        image = io.imread(img_item_path)/255
        image = torch.from_numpy(image)
        label = io.imread(label_item_path)
        label = torch.from_numpy(label)
        return image,label
    def __len__(self):
        return len(self.image_path)


def testdate(test_loader, model):
    avg_meters = {#'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'f1-score':AverageMeter()
    }
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader))
        for input, target in test_loader:
            input = input.float().cuda()
            target = target.long().cuda()
            b, h, w, c = input.size()
            input = input.reshape(b, c, h, w)
            output = model(input)

            preds = torch.softmax(output, dim=1).cpu()
            preds = torch.argmax(preds.data, dim=1)
            predimg_color.append(preds)
            preds = torch.squeeze(preds)
            predimg.append(preds)
            labelimg.append(target.cpu())

            iou = iou_score2(output, target)
            dice = dice_score(output, target)
            f1_score = f1_scorex2(output, target)
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['f1-score'].update(f1_score, input.size(0))
            postfix = OrderedDict([
                #('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('f1-score', avg_meters['f1-score'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    return OrderedDict([#('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('f1-score',avg_meters['f1-score'].avg)])


colormap1 = [[0,0,0], [255,0,0],[0,255,0],[0,255,255],[255,0,255],[255,255,0],[0,255,255]]
#{红、绿、青、粉、黄、紫}

def label2image(prelabel,colormap):
    #预测的标签转化为图像，针对一个标签图
    _,h,w = prelabel.shape
    prelabel = prelabel.reshape(h*w,-1)
    image = np.zeros((h*w,3),dtype=np.uint8)
    for i in range(len(colormap)):
        index = np.where(prelabel == i)
        image[index,:] = colormap[i]
    return image.reshape(h, w, 3)

forecast_label = 'F:\\Medical_imageseg_yes_baixibao\\test_mydata\\unet_1\\forecast_label'
forecast_label_npy = 'F:\\Medical_imageseg_yes_baixibao\\test_mydata\\unet_1\\forecast_label_npy'
labelnpytoimg_dir = 'F:\\Medical_imageseg_yes_baixibao\\test_mydata\\unet_1\\labelnpytoimg_dir'
imgnpytoimg_dir = "F:\\Medical_imageseg_yes_baixibao\\test_mydata\\unet_1\imgnpytoimg_dir"
img_dir = 'F:\\Medical_imageseg_yes_baixibao\\baixibao_mydata\\test\\image'
label_dir = 'F:\\Medical_imageseg_yes_baixibao\\baixibao_mydata\\test\\label'
imgandlabel = "F:\\Medical_imageseg_yes_baixibao\\test_mydata\\unet_1\\imgandlabel"
imgandlabel2 = "F:\\Medical_imageseg_yes_baixibao\\test_mydata\\unet_1\\imgandlabel2"

img_read = os.listdir(img_dir)

dataset = MyData(img_dir, label_dir,transformers=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

config = vars(parse_args())
print("=> creating model" )
if config['name'] == "DRA-net":
    net = DRA_net(3, 7)
else:
    raise ValueError("Wrong Parameters")
net.load_state_dict(torch.load(("F:\Medical_imageseg_yes_baixibao\checpoint\\newdata\\UNET_base\\bestmodel_baixibao_0.0001_final.pth").format(config["name"])))
net.cuda()


if __name__ == "__main__":
    test_log = testdate(test_loader,net)
    print('testdata IOU:{:.4f}, testdata dice:{:.4f}, testdata f1-score:{:.4f}'.format(test_log['iou'],
                                                        test_log['dice'],test_log['f1-score']))
    for i in range(len(predimg)):
        pre = predimg[i]#预测npy
        preimg2 = label2image(predimg_color[i],colormap=colormap1)#预测label图片
        label = label2image(labelimg[i],colormap=colormap1)#原始label图片
        x = io.imread(img_dir+"\\"+img_read[i])
        if i < 10:
            test_pre_name = "000{}.png".format(i)
            test_pre_np = "000{}.npy".format(i)

            clip_image_path2 = os.path.join(imgnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path2,x)
            imgx = add_alpha_channel(clip_image_path2,0.85)
            imgx2 = add_alpha_channel(clip_image_path2,0.85)

            clip_image_path = os.path.join(labelnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path, label)
            labelx = add_alpha_channel(clip_image_path,0.85)

            image_together(imgx,labelx,imgandlabel,test_pre_name)

            clip_label_path = os.path.join(forecast_label, test_pre_name)
            io.imsave(clip_label_path, preimg2)
            prelabelx = add_alpha_channel(clip_label_path, 0.85)
            image_together(imgx2, prelabelx, imgandlabel2, test_pre_name)

            pre_np = os.path.join(forecast_label_npy, test_pre_np)
            np.save(pre_np, pre)

        if i >= 10 and i < 100:
            test_pre_name = "00{}.png".format(i)
            test_pre_np = "00{}.npy".format(i)

            clip_image_path2 = os.path.join(imgnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path2, x)
            imgx = add_alpha_channel(clip_image_path2, 0.85)
            imgx2 = add_alpha_channel(clip_image_path2, 0.85)

            clip_image_path = os.path.join(labelnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path, label)
            labelx = add_alpha_channel(clip_image_path, 0.85)

            image_together(imgx, labelx, imgandlabel, test_pre_name)

            clip_label_path = os.path.join(forecast_label, test_pre_name)
            io.imsave(clip_label_path, preimg2)
            prelabelx = add_alpha_channel(clip_label_path, 0.85)
            image_together(imgx2, prelabelx, imgandlabel2, test_pre_name)

            pre_np = os.path.join(forecast_label_npy, test_pre_np)
            np.save(pre_np, pre)

        if i>=100 and i < 1000:
            test_pre_name = "0{}.png".format(i)
            test_pre_np = "0{}.npy".format(i)

            clip_image_path2 = os.path.join(imgnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path2, x)
            imgx = add_alpha_channel(clip_image_path2, 0.85)
            imgx2 = add_alpha_channel(clip_image_path2, 0.85)

            clip_image_path = os.path.join(labelnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path, label)
            labelx = add_alpha_channel(clip_image_path, 0.85)

            image_together(imgx, labelx, imgandlabel, test_pre_name)

            clip_label_path = os.path.join(forecast_label, test_pre_name)
            io.imsave(clip_label_path, preimg2)
            prelabelx = add_alpha_channel(clip_label_path, 0.85)
            image_together(imgx2, prelabelx, imgandlabel2, test_pre_name)

            pre_np = os.path.join(forecast_label_npy, test_pre_np)
            np.save(pre_np, pre)

        if i>=1000:
            test_pre_name = "{}.png".format(i)
            test_pre_np = "{}.npy".format(i)

            clip_image_path2 = os.path.join(imgnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path2, x)
            imgx = add_alpha_channel(clip_image_path2, 0.85)
            imgx2 = add_alpha_channel(clip_image_path2, 0.85)

            clip_image_path = os.path.join(labelnpytoimg_dir, test_pre_name)
            io.imsave(clip_image_path, label)
            labelx = add_alpha_channel(clip_image_path, 0.85)

            image_together(imgx, labelx, imgandlabel, test_pre_name)

            clip_label_path = os.path.join(forecast_label, test_pre_name)
            io.imsave(clip_label_path, preimg2)
            prelabelx = add_alpha_channel(clip_label_path, 0.85)
            image_together(imgx2, prelabelx, imgandlabel2, test_pre_name)

            pre_np = os.path.join(forecast_label_npy, test_pre_np)
            np.save(pre_np, pre)



