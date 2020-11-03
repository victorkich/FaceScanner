<h1 align="center">FaceScanner</h1>

<p align="center"> 
  <img src="https://img.shields.io/badge/PyTorch-v1.6.0-blue"/>
  <img src="https://img.shields.io/badge/OpenCV-v4.4.0.42-blue"/>
  <img src="https://img.shields.io/badge/Tensorflow-v2.3.0-blue"/>
  <img src="https://img.shields.io/badge/Torchvision-v0.8.1-blue"/>
  <img src="https://img.shields.io/badge/Scipy-v1.5.3-blue"/>
  <img src="https://img.shields.io/badge/Matplotlib-v3.3.2-blue"/>
  <img src="https://img.shields.io/badge/Pandas-v1.1.2-blue"/>
  <img src="https://img.shields.io/badge/Tqdm-v4.49.0-blue"/>
  <img src="https://img.shields.io/badge/Numpy-v1.19.2-blue"/>
</p>
<br/>

## Objective
<p align="justify"> 
  <a>Implementation of multiples recent articles for detect age, gender, region, using or not using mask.</a>  
</p>
  

## Setup

<p align="justify"> 
 <a>Make sure after your git clone, create a folder called output. To do it, use:</a>
</p>

```shell
git clone https://github.com/victorkich/FaceScanner/
```

```shell
cd FaceScanner
```

```shell
mkdir output
```

<p align="justify"> 
 <a>All of requirements is show in the badgets above, but if you want to install all of them, enter the repository and execute the following line of code:</a>
</p>

```shell
pip3 install -r requirements.txt
```

<p align="justify"> 
 <a>For test in your image paste the file inside tests folder and put the following code:</a>
</p>

```shell
python3 predict.py --img your_img.type
```

<p align="justify"> 
 <a>The bouding box image and the log.csv will be appear inside the output folder.</a>
</p>

## Examples

<p align="center"> 
  <img src="media/example1.jpg" alt="FaceScanner"/>
  <img src="media/example2.jpg" alt="FaceScanner"/>
</p>  

## References
<p align="justify"> 
  This project uses a lot of already published articles, and you can see the references to the articles below:
  
  <a href="https://github.com/dchen236/FairFace">FairFace</a>
  
  <a href="https://github.com/biubug6/Pytorch_Retinaface">RetinaFace</a>
  
  <a href="https://github.com/chandrikadeb7/Face-Mask-Detection">Face-Mask-Detection</a>
</p>

<p align="center"> 
  <i>If you liked this repository, please don't forget to starred it!</i>
  <img src="https://img.shields.io/github/stars/victorkich/FaceScanner?style=social"/>
</p>
