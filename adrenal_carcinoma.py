
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import monai
import torch
import pydicom
import nibabel as nib
import os
from glob import glob
import dicom2nifti
from celluloid import Camera
from IPython.display import HTML
import SimpleITK as sitk

from monai.apps.tcia import TCIA_LABEL_DICT
from monai.config import print_config
from monai.networks.nets import UNet , BasicUnetPlusPlus
from monai.networks.layers import Norm
from monai.metrics import DiceMetric , get_confusion_matrix
from monai.losses import DiceLoss , DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch , ArrayDataset
from monai.apps import TciaDataset
from monai.config import print_config , KeysCollection
from monai.utils import first , set_determinism
from monai.transforms import (
    Compose,
    LoadImage,
    LoadImaged,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    ToTensor,
    ToTensord,
    ScaleIntensityRange,
    ScaleIntensityRanged,
    ThresholdIntensity,
    ThresholdIntensityd,
    SaveImaged,
    Spacingd,
    CropForegroundd,
    Orientationd,
    AsDiscrete,
    RandCropByPosNegLabeld,
    DivisiblePadd,
    Resized,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd




)

print_config()

HOME = os.getcwd()
HOME

new_dir = 'Data'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)


data_root = '/Data/Adrenal-ACC-Ki67-Seg'
#for removing zip files after extraction
zip_file = glob(os.path.join(data_root , '*.zip'))

for i in zip_file:
    os.remove(i)

import shutil
shutil.rmtree('/kaggle/working/Data/Adrenal-ACC-Ki67-Seg/raw')

paths = glob(os.path.join(data_root,'*','*','*' , '*.dcm'))

paths[1].split('/')[-4]

#making dataframe from paths for better handling and splitting
df = pd.DataFrame(paths , columns = ['f_path'])
df['case_name'] = [i.split('/')[-4] for i in df['f_path']]
df['type'] = [i.split('/')[-2] for i in df['f_path']]
df['root'] = [i.split('/')[-5] for i in df['f_path']]
df

# len(folders) , len(img_list) , len(seg_list)

#new dirs to recieve the output nifti files
imagesTr = 'imageTr'
labelsTr = 'labelTr'
if not os.path.exists(imagesTr):
    os.mkdir(imagesTr)
if not os.path.exists(labelsTr):
    os.mkdir(labelsTr)

imgs_out_path = os.path.join(HOME , 'imageTr')
segs_out_path = os.path.join(HOME , 'labelTr')

df.head(1)

images_df = df.loc[df['type'] == 'image']
labels_df = df.loc[df['type'] == 'seg']

len(labels_df)


#transformng dicom stcked labels to nifti
for x,i in enumerate(labels_df['f_path']):
    label = i
    print(label)
    img = sitk.ReadImage(label)
    imgd = sitk.GetArrayFromImage(img)
    sitk.WriteImage(img , os.path.join(segs_out_path , f"{str(i.split('/')[-4])}.nii"))

images_df['case_name'].nunique()

images_list = glob(os.path.join(data_root , '*' , '*' , 'image'))
images_list

images_path_list = []
default_list = []
#transform images dirs from dicom_dirs to nifti
for x,i in enumerate(images_list):
    cnt = 0
    img_dir = i
    try:
        dicom2nifti.dicom_series_to_nifti(i , os.path.join(imgs_out_path , str(i.split('/')[-3]) + '.nii') , reorient_nifti = False )
    #handling exceptions for corrupted images
    except Exception:
        print(str(i.split('/')[-3]))
        default_list.append(str(i.split('/')[-3]))
        pass
    cnt+=x
    print(cnt)

# len(os.listdir(segs_out_path))
    
#image/labels matching
for i in os.listdir(segs_out_path):
    cnt = 0
    print(i)
    if i.split('.')[0] == default_list[0]:
        os.remove(os.path.join(HOME , segs_out_path,i))
        cnt+=1
        print(f'{cnt} files removed')
        print(os.listdir(segs_out_path))

len(os.listdir(segs_out_path))

train_images = sorted(glob(os.path.join('/kaggle/working/imageTr', "*.nii")))
train_labels = sorted(glob(os.path.join('/kaggle/working/labelTr', "*.nii")))
# train_labels = sorted(glob(os.path.join(data_root, "*", "300" , 'seg' , '*.dcm')))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files, val_files = data_dicts[:-8], data_dicts[-8:]

len(train_files) , len(val_files)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import random
r = random.randint(0, len(train_images))


r_img = nib.load(train_images[r]).get_fdata()
print(r_img.shape , np.max(r_img) , np.min(r_img))
r_label = nib.load(train_labels[r]).get_fdata()
print(r_label.shape , r_label.min() , r_label.max() , np.max(r_label),np.min(r_label))


plt.figure(figsize = (8,5))
plt.subplot(121)
plt.imshow(r_img[:,:,20], cmap = 'gray')
plt.colorbar()
plt.subplot(122)
plt.imshow(r_label[:,:,20])
plt.show()

#checking labels consistency
from tqdm.auto import tqdm
#checking labels consistancy
for i in tqdm(train_labels):
    label = nib.load(i).get_fdata()
    # print(i,len(np.unique(label)))
    if len(np.unique(label)) > 2:
        print(f'default file {i}')

#setting piplines for train and validation

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200.0,
            a_max=200.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
                ScaleIntensityRanged(
            keys=["label"],
            a_min=0.0,
            a_max=255.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),

            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64, 64, 64),
                pos=1,
                neg=1,
                num_samples=6,
                image_key="image",
                image_threshold=0,
            ),
        #DivisiblePadd(keys=["image", "label"], k = 64),
     DivisiblePadd(keys = ['image' , 'label'] , k = 32),
#             Resized(
#                 keys=["image", "label"],
#                 spatial_size=(128, 128, 128)
#             ),
            RandFlipd(
                keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        )
          ])


val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200.0,
            a_max=200.0,
            b_min=0.0,
            b_max=1.0,
            clip=True),
                ScaleIntensityRanged(
            keys=["label"],
            a_min=0.0,
            a_max=255.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        CropForegroundd(keys=["image", "label"], source_key="image"),

        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    DivisiblePadd(keys = ['image' , 'label'] , k = 32)])
# Resized(keys=["image", "label"], spatial_size = (128,128,128))
#  DivisiblePadd(keys = ['image' , 'label'] , k = 32)

check_ds = Dataset(data=train_files, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
image, label = (check_data["image"][0][0], check_data["label"][0][0])
print(f"image shape: {image.shape}, label shape: {label.shape} , label max: {np.max(label)}")
# for i in range(label.shape[2]):
#     if label[i].max() == 1:
#         k = random.randint(0,i)
# print(k)
plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :,80 ], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, 80])
plt.show()

check_ds = Dataset(data=val_files, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
image, label = (check_data["image"][0][0], check_data["label"][0][0])
print(f"image shape: {image.shape}, label shape: {label.shape}, label max: {np.max(label)}")
# for i in range(label.shape[2]):
#     if label[i].max() == 1:
#         k = random.randint(0,i)
# print(k)
plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :, 80], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, 80])
plt.show()

# Dataloaders - Train , val

#train
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=1 )
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1 , drop_last = True )

#val
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2 )

#Model
import torch

device = torch.device("cuda")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(32, 64, 128, 256 , 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True )
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=True, reduction="mean")

HOME

max_epochs = 300
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
#         print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    #np.save(epoch_loss_values[0],os.path.join(HOME , 'epoch_loss.npy'))
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.inference_mode():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (64, 64, 64)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model )
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
#                 print(val_labels.shape , val_outputs.shape)

                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                #np.save(metric_values,os.path.join(HOME , 'metric_values.npy'))
                torch.save(model.state_dict(), os.path.join(HOME, "best_metric_model.pth") )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.show()

model.load_state_dict(torch.load(os.path.join(HOME, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        roi_size = (64, 64, 64)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)



        # plot the slice [:, :, 80]
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {i}")
        plt.imshow(val_data["image"][0, 0, :, :,80], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {i}")
        plt.imshow(val_data["label"][0, 0, :, :, 80])
        plt.subplot(1, 3, 3)
        plt.title(f"output {i}")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 80])
        plt.show()
        #val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
        val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
        nib.save(nib.Nifti1Image(val_outputs,val_data['image'][0,0,:,:,:].affine), os.path.join(HOME, 'Modelpred_'+str(i)+'_.nii.gz'))

        if i == 10:
            break

CT = nib.load('/kaggle/working/imageTr/Adrenal_Ki67_Seg_053.nii').get_fdata()
CT = np.clip(CT , a_min = -200 , a_max = 200 )
msk = nib.load('/kaggle/working/labelTr/Adrenal_Ki67_Seg_053.nii').get_fdata()


fig = plt.figure()
camera = Camera(fig)

for i in range(CT.shape[2]):
    plt.imshow(CT[:,:,i] , cmap = 'bone')
    mask = np.ma.masked_where(msk[:,:,i] == 0 , msk[:,:,i])
    plt.imshow(mask,alpha = 0.5 )
    camera.snap()
animation = camera.animate()

HTML(animation.to_html5_video())

