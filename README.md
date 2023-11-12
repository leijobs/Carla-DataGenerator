# Carla-DataGenerator

this is a almost-forgotten project, for carla data generation

I modified the [Original Project](https://github.com/mmmmaomao/DataGenerator) as follows:
* support road segmentation task
* support kitti-style object label
* support kitti-based data visualization
- [ ] Todo: support fisheye camera

## SimpleRun
1. download carla `0.9.12` or newer pre-build version
2. run carla simulator
3. run as follows:
```python
python generator.py
```

## Visualization
1. modify data path as your path in following .py
2. install mayavi and vtk in advance
3. run follow .py script:
```python
python kitti_img_vis.py
python kitti_lidar_vis.py
python kitti_plot_3d_boxes.py
```

## Kitti-style object filter
in `` you can find:
```python

```


