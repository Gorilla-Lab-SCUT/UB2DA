#!/bin/sh
python train.py --exp_name office31-run1 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_amazon.txt --target ./txt/office31/target_dslr.txt --gpu $1
python train.py --exp_name office31-run1 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_amazon.txt --target ./txt/office31/target_webcam.txt --gpu $1
python train.py --exp_name office31-run1 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_dslr.txt --target ./txt/office31/target_webcam.txt --gpu $1
python train.py --exp_name office31-run1 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_dslr.txt --target ./txt/office31/target_amazon.txt --gpu $1
python train.py --exp_name office31-run1 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_webcam.txt --target ./txt/office31/target_amazon.txt --gpu $1
python train.py --exp_name office31-run1 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_webcam.txt --target ./txt/office31/target_dslr.txt --gpu $1

python train.py --exp_name office31-run2 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_amazon.txt --target ./txt/office31/target_dslr.txt --gpu $1
python train.py --exp_name office31-run2 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_amazon.txt --target ./txt/office31/target_webcam.txt --gpu $1
python train.py --exp_name office31-run2 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_dslr.txt --target ./txt/office31/target_webcam.txt --gpu $1
python train.py --exp_name office31-run2 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_dslr.txt --target ./txt/office31/target_amazon.txt --gpu $1
python train.py --exp_name office31-run2 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_webcam.txt --target ./txt/office31/target_amazon.txt --gpu $1
python train.py --exp_name office31-run2 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_webcam.txt --target ./txt/office31/target_dslr.txt --gpu $1

python train.py --exp_name office31-run3 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_amazon.txt --target ./txt/office31/target_dslr.txt --gpu $1
python train.py --exp_name office31-run3 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_amazon.txt --target ./txt/office31/target_webcam.txt --gpu $1
python train.py --exp_name office31-run3 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_dslr.txt --target ./txt/office31/target_webcam.txt --gpu $1
python train.py --exp_name office31-run3 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_dslr.txt --target ./txt/office31/target_amazon.txt --gpu $1
python train.py --exp_name office31-run3 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_webcam.txt --target ./txt/office31/target_amazon.txt --gpu $1
python train.py --exp_name office31-run3 --config ./configs/office31-train-config.yaml --source ./txt/office31/source_webcam.txt --target ./txt/office31/target_dslr.txt --gpu $1