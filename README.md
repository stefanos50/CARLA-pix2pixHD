# Pix2PixHD for CARLA Simulator
This is a modfied version of the [Pix2PixHD project](https://github.com/NVIDIA/pix2pixHD) that enables it's real-time execution for the CARLA simulator.
<br>

<div align="center">
  <img src="https://drive.google.com/thumbnail?id=1BJWDS6YfXTViW6Iv5rBlQ9fLWY76nhRW&sz=w1000" alt="Image" width="400px" height="auto">
  <img src="https://drive.google.com/thumbnail?id=1jvY1qv7yFByHrjH5D7paDo5I5DVFqpkh&sz=w1000" alt="Image" width="400px" height="auto">
</div>

## Features

* Code for running Pix2PixHD in CARLA simulator in real-time.
* Code for extracting a synthetic dataset in synchronous mode (RGB Frame, Synthesized Frame, Instance Segmentation, and Semantic Segmentation).
* Support for synchronous and asynchronous modes.
* Parameterization from a yaml config file rather than directly interacting with the code.

## Prerequisites
- Linux or macOS or Windows
- [CARLA version 0.9.14](https://carla.org/2022/12/23/release-0.9.14/) or higher (older versions do not follow the Cityscapes labeling scheme).
- Python 2 or 3
- NVIDIA GPU (20G memory or larger) + CUDA cuDNN

> 📝 **Note**: The code was tested with an RTX 4090 GPU, Windows 11 operating system, CARLA 0.9.14, Python 3.8, and PyTorch with CUDA 11.8.

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install python libraries [dominate](https://github.com/Knio/dominate).
```bash
pip install dominate
pip install numpy
pip install carla
pip install opencv-python
pip install pillow
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- Clone this repo:
```bash
git clone https://github.com/stefanos50/CARLA-pix2pixHD
cd CARLA-pix2pixHD
```

### Real-Time Execution
- Please download the pre-trained Cityscapes model from [here](https://drive.google.com/file/d/1h9SykUnuZul7J3Nbms2QGH1wa85nbN2-/view?usp=sharing) (google drive link), and put it under `./checkpoints/label2city_1024p/`
- Execute tha CARLA.exe (executable) and wait until the world is initialized.
- Execute the following command:
```bash
python test.py --name label2city_1024p --netG local --ngf 32 --resize_or_crop none
```

### Configuration

The yaml configuration file can be located in `./CARLA-pix2pixHD/carla_settings.yaml/` and it contains the following parameters:

```bash
connection:
 ip: 127.0.0.1 #server ip
 port: 2000 #server port
 timeout: 10.0 #timeout wait time

world:
 town: Town01 #the selected carla town (Town01, Town10HD, Town02, ..., etc.)
 fixed_delta_seconds: 0.05
 synchronous_mode: True #Synchronous or Asynchronous execution (True or False)
 weather_preset: ClearNoon #The selected predefined weather presets from the documentations (ClearNoon, ClearSunet, ... , etc.)

general:
 vehicle: vehicle.tesla.model3 (the selected vehicle from the documentation catalogue)
 cam_width: 2048 #camera image width
 cam_height: 1024 #camera image heigh
 cam_x: 1.5 #x coordinate (location) of the camera
 cam_z: 1.4 #z coordinate (location) of the camera
 visualize_results: True #visualize the sensor data and results (original frame, synthesized frame, semantic, and instance segmentation)
 colorize_masks: False #colorize the grayscale semantic segmentation mask (it will drop the performance)
 no_instance: False #do not use instance maps during the inference
 scale_instance: True #scale the instance segmentation back to uint numbers
 use_label_as_instance: False #use the semantic segmentation mask as instance (Cityscapes for most objects uses the same id as in the label mask in comparison with the CARLA instance segmentation sensor)
 map_instances: False #map the labels for vehicles,pederestrian etc. to specific instances from Cityscapes (used if use_label_as_instance = False).
 cityscapes_label: [24,26] #cityscapes classes
 cityscapes_instance: [24001,26005] #instances mapping
 run_model_every_n: 3 #run the moden every 'n' ticks of the world. If set to 1 then it will run for every frame. This can increase the performance (FPS).

dataset:
 export_data: True #export data on the disk (export location: \datasets\carla\)
 export_step: 20 #export every 'step' frames
 capture_when_static: True #export frames when the vehicle is not moving (True or False)
 speed_threshold: 0.1 #export frames only when a speed threshold is reached

pygame:
 window_width: 1024 #the width of the pygame window
 window_height: 512 #the height of the pygame window
```

### Spawning Traffic and other functionalities

The code works with most of the samples that CARLA already provides in the `\CarlaEXE\PythonAPI\examples` directory. If you want to spawn a variety of vehicles and NPCs in the world, you can easily execute the provided `generate_traffic.py` script. The same is also applicable for dynamic weather via `dynamic_weather.py` and any other functionality that is already provided by the CARLA team.


### Visualization (Real-Time) Results

![Screenshot 2024-02-22 153923](https://github.com/stefanos50/CARLApix2pixHD/assets/36155283/d77ce26b-ad30-46d7-8220-bf1994501255)

![Screenshot 2024-02-22 160354](https://github.com/stefanos50/CARLApix2pixHD/assets/36155283/40f74d44-7f2b-470f-865c-493c2bf6d05a)

### Dataset
- We use the Cityscapes dataset. To train a model on the full dataset, please download it from the [official website](https://www.cityscapes-dataset.com/) (registration required).
After downloading, please put it under the `datasets` folder in the same way the example images are provided.


### Training
- Train a model at 1024 x 512 resolution (`bash ./scripts/train_512p.sh`):
```bash
#!./scripts/train_512p.sh
python train.py --name label2city_512p
```
- To view training results, please checkout intermediate results in `./checkpoints/label2city_512p/web/index.html`.
If you have tensorflow installed, you can see tensorboard logs in `./checkpoints/label2city_512p/logs` by adding `--tf_log` to the training scripts.

### Multi-GPU training
- Train a model using multiple GPUs (`bash ./scripts/train_512p_multigpu.sh`):
```bash
#!./scripts/train_512p_multigpu.sh
python train.py --name label2city_512p --batchSize 8 --gpu_ids 0,1,2,3,4,5,6,7
```
Note: this is not tested and we trained our model using single GPU only. Please use at your own discretion.

### Training with Automatic Mixed Precision (AMP) for faster speed
- To train with mixed precision support, please first install apex from: https://github.com/NVIDIA/apex
- You can then train the model by adding `--fp16`. For example,
```bash
#!./scripts/train_512p_fp16.sh
python -m torch.distributed.launch train.py --name label2city_512p --fp16
```
In our test case, it trains about 80% faster with AMP on a Volta machine.

### Training at full resolution
- To train the images at full resolution (2048 x 1024) requires a GPU with 24G memory (`bash ./scripts/train_1024p_24G.sh`), or 16G memory if using mixed precision (AMP).
- If only GPUs with 12G memory are available, please use the 12G script (`bash ./scripts/train_1024p_12G.sh`), which will crop the images during training. Performance is not guaranteed using this script.

### Training with your own dataset
- If you want to train with your own dataset, please generate label maps which are one-channel whose pixel values correspond to the object labels (i.e. 0,1,...,N-1, where N is the number of labels). This is because we need to generate one-hot vectors from the label maps. Please also specity `--label_nc N` during both training and testing.
- If your input is not a label map, please just specify `--label_nc 0` which will directly use the RGB colors as input. The folders should then be named `train_A`, `train_B` instead of `train_label`, `train_img`, where the goal is to translate images from A to B.
- If you don't have instance maps or don't want to use them, please specify `--no_instance`.
- The default setting for preprocessing is `scale_width`, which will scale the width of all training images to `opt.loadSize` (1024) while keeping the aspect ratio. If you want a different setting, please change it by using the `--resize_or_crop` option. For example, `scale_width_and_crop` first resizes the image to have width `opt.loadSize` and then does random cropping of size `(opt.fineSize, opt.fineSize)`. `crop` skips the resizing step and only performs random cropping. If you don't want any preprocessing, please specify `none`, which will do nothing other than making sure the image is divisible by 32.

## More Training/Test Details
- Flags: see `options/train_options.py` and `options/base_options.py` for all the training flags; see `options/test_options.py` and `options/base_options.py` for all the test flags.
- Instance map: we take in both label maps and instance maps as input. If you don't want to use instance maps, please specify the flag `--no_instance`.


## Citation

If you find this useful for your research, please use the following.

```
@inproceedings{wang2018pix2pixHD,
  title={High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs},
  author={Ting-Chun Wang and Ming-Yu Liu and Jun-Yan Zhu and Andrew Tao and Jan Kautz and Bryan Catanzaro},  
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

## Acknowledgments
This code borrows heavily from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
