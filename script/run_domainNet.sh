#!/bin/sh
python train.py --exp_name domainNet-run1 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_painting.txt --target ./txt/domainNet/target_real.txt --gpu $1
python train.py --exp_name domainNet-run1 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_painting.txt --target ./txt/domainNet/target_sketch.txt --gpu $1
python train.py --exp_name domainNet-run1 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_real.txt --target ./txt/domainNet/target_sketch.txt --gpu $1
python train.py --exp_name domainNet-run1 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_real.txt --target ./txt/domainNet/target_painting.txt --gpu $1
python train.py --exp_name domainNet-run1 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_sketch.txt --target ./txt/domainNet/target_painting.txt --gpu $1
python train.py --exp_name domainNet-run1 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_sketch.txt --target ./txt/domainNet/target_real.txt --gpu $1

python train.py --exp_name domainNet-run2 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_painting.txt --target ./txt/domainNet/target_real.txt --gpu $1
python train.py --exp_name domainNet-run2 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_painting.txt --target ./txt/domainNet/target_sketch.txt --gpu $1
python train.py --exp_name domainNet-run2 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_real.txt --target ./txt/domainNet/target_sketch.txt --gpu $1
python train.py --exp_name domainNet-run2 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_real.txt --target ./txt/domainNet/target_painting.txt --gpu $1
python train.py --exp_name domainNet-run2 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_sketch.txt --target ./txt/domainNet/target_painting.txt --gpu $1
python train.py --exp_name domainNet-run2 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_sketch.txt --target ./txt/domainNet/target_real.txt --gpu $1

python train.py --exp_name domainNet-run3 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_painting.txt --target ./txt/domainNet/target_real.txt --gpu $1
python train.py --exp_name domainNet-run3 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_painting.txt --target ./txt/domainNet/target_sketch.txt --gpu $1
python train.py --exp_name domainNet-run3 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_real.txt --target ./txt/domainNet/target_sketch.txt --gpu $1
python train.py --exp_name domainNet-run3 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_real.txt --target ./txt/domainNet/target_painting.txt --gpu $1
python train.py --exp_name domainNet-run3 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_sketch.txt --target ./txt/domainNet/target_painting.txt --gpu $1
python train.py --exp_name domainNet-run3 --config ./configs/domainNet-train-config.yaml --source ./txt/domainNet/source_sketch.txt --target ./txt/domainNet/target_real.txt --gpu $1
