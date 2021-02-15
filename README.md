## 目的
* ニューラルネットワークで線形回帰を行う
* **M1 Macでは実行できません**

## Quick Start
1. Install Git and Docker on your computer.
2. Clone the repository:
```shell
$ git clone https://github.com/takerukyono/Neural_network.git
```
3. Create an image:
```shell
$ docker build -t tensorflow:0.1 .
```
4. Check the image:
```shell
$ docker images
```
5. Run the container:
```shell
$ docker run -it tensorflow:0.1 bash
```
6. Run the program:
```shell
python ./nn-hidden_sigmoid.py
```
7. Let's check the results. The images will also be saved in the images directory.

## Task
- [ ] Change the number of learning (epochs).
- [ ] Change the number of units in the hidden layer(num_hidden).
