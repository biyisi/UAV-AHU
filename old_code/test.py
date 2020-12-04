import os

import pretrainedmodels
import torch
import torchvision

# model = torchvision.models.resnet18(pretrained=True)
# model = torchvision.models.vgg16_bn(pretrained=True)
# print(model)


# print(pretrainedmodels.model_names)
# print(pretrainedmodels.pretrained_settings['resnet18'])
# model_name = 'resnet152'
# model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
# print(model)

transform_train_list = [
    # torchvision.transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    torchvision.transforms.Resize((384, 384), interpolation=3),
    torchvision.transforms.Pad(10, padding_mode='edge'),
    torchvision.transforms.RandomCrop((384, 384)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
transform_val_list = [
    torchvision.transforms.Resize(size=(384, 384), interpolation=3),  # Image.BICUBIC
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
data_transforms = {
    'train': torchvision.transforms.Compose(transform_train_list),
    'val': torchvision.transforms.Compose(transform_val_list)
}
data_transforms = data_transforms

image_datasets = {}
image_datasets['view'] = torchvision.datasets.ImageFolder(os.path.join("./data/train", 'test_view'),
                                                          data_transforms['train'])
image_datasets['google'] = torchvision.datasets.ImageFolder(os.path.join("./data/train", 'google'),
                                                            data_transforms['train'])

dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=2,
                                   pin_memory=True)
    for x in ['view', 'google']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['view', 'google']}

num = dataloaders['view']
print(num)

import os

print(os.path.abspath(os.curdir))
def get_file_num(path):
    # os.chdir(path)
    ret = os.listdir(path)
    print("len(ret) =", len(ret))
    # os.chdir()ni

if __name__ == '__main__':
    path = "./data/train/test_view"
    get_file_num(path)
    print(os.path.abspath(os.curdir))

# for data in dataloaders['view']:
#     inputs, labels = data
#     print(labels)
#     break

# for data, data1 in zip(dataloaders['view'], dataloaders['google']):
#     temp = dataloaders['view']
#     inputs, labels = data
#     print(inputs)
#     print(labels)
#     print("----------------------------")
#     print("----------------------------")
#     break
