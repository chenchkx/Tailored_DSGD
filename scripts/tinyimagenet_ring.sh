
# Resnet18
python main.py --dataset_name "TinyImageNet" --image_size 56 --batch_size 64 --mode "ring" --size 16  --lr 0.2 --model "ResNet18" --warmup_step 20 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --device 0
python main.py --dataset_name "TinyImageNet" --image_size 56 --batch_size 512 --mode "ring" --size 16  --lr 1.6 --model "ResNet18" --warmup_step 20 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --device 0

# AlexNet
python main.py --dataset_name "TinyImageNet" --image_size 64 --batch_size 64 --mode "ring" --size 16  --lr 0.01 --model "AlexNet" --warmup_step 20 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --device 0
python main.py --dataset_name "TinyImageNet" --image_size 64 --batch_size 512 --mode "ring" --size 16  --lr 0.08 --model "AlexNet" --warmup_step 20 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --device 0


# DenseNet
python main.py --dataset_name "TinyImageNet" --image_size 56 --batch_size 64 --mode "ring" --size 16  --lr 0.1 --model "DenseNet" --warmup_step 20 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --device 0
python main.py --dataset_name "TinyImageNet" --image_size 56 --batch_size 512 --mode "ring" --size 16  --lr 0.8 --model "DenseNet" --warmup_step 20 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --device 0
