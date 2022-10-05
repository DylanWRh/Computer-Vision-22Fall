# Hybrid Image

This program reads in two images and produces a hybrid image, 
and is implemented according to the requirements of the 
[Problem Set](https://pku.vision/course/22fall/Problem.Set.1.pdf).

## Usage

Run the command below to produce the hybrid image of [dog.bmp](1_dog.bmp) and [cat.bmp](1_cat.bmp).
```
python hybrid.py
```
Low pass image, high pass image and hybrid image will be saved
as `left.jpg`, `right.jpg`, `hybrid.jpg`.

If you want to process your own images, run
```
python hybrid.py --left image1_path --right image2_path
```
