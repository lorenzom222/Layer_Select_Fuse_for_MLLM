#!/bin/bash

# Base directory
BASE_DIR="/home/aac/ssrivas9/Layer_Select_Fuse_for_MLLM/playground/data"

# Create necessary directories
#mkdir -p $BASE_DIR/{coco/train2017,gqa/images,textvqa/train_images,vg/{VG_100K,VG_100K_2}}

# # COCO Dataset
# echo "Downloading COCO dataset (this is a large download)..."
# cd $BASE_DIR/coco
# wget http://images.cocodataset.org/zips/train2017.zip
# unzip train2017.zip
# rm train2017.zip

# GQA Dataset
echo "Downloading GQA dataset..."
cd $BASE_DIR/gqa
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip
rm images.zip

# TextVQA Dataset
echo "Downloading TextVQA dataset..."
cd $BASE_DIR/textvqa
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip
mv train_val_images/* train_images/
rmdir train_val_images
rm train_val_images.zip

# VisualGenome Dataset
echo "Downloading VisualGenome dataset (this may take a while)..."
cd $BASE_DIR/vg
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images.zip -d VG_100K
unzip images2.zip -d VG_100K_2
rm images.zip images2.zip

echo "All downloads completed!"