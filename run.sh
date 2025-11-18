# CIFAR-10 (tự động chọn allconv, nin, vgg16)
python exp.py --dataset cifar10

# ImageNet (tự động chọn alexnet, chỉ non-targeted)
python exp.py --dataset imagenet_folder --imagenet_folder data/imagenet_227

# Hoặc chỉ định model cụ thể
python exp.py --dataset cifar10 --models allconv nin