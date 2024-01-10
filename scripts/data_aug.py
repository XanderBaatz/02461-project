# Libraries
import torchvision.transforms as transforms
import train

# Auto augmentations
tf_a0 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

tf_a1 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.AutoAugment(),
    transforms.ToTensor(),
])

tf_a2 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandAugment(),
    transforms.ToTensor(),
])

tf_a3 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.AugMix(),
    transforms.ToTensor(),
])

tf_a4 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
])

# Assemble the transform variations into a list
#transforms_list = [tf_1, tf_2, tf_3, tf_4, tf_5, tf_6, tf_7, tf_8, tf_9, tf_10]
transforms_list = [tf_a0, tf_a1, tf_a2, tf_a3, tf_a4]

for i, t in enumerate(transforms_list, start=1):
    train.train_model(mn_append=f"a{i}", train_transform=t)

#for transform in tf_10.transforms:
#    print(transform)