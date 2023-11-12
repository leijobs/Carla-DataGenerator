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

## Road Seg
in file `DataSave` you can find:
```python
SEM_COLORS = {

    6: (157, 234, 50), # ROAD LINE
    7: (128, 64, 128), # ROAD

}
```
here is the list of segmentation you can modify:
```

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
in `data_utils.py` you can find this, you can change object grouping scheme
```python
vehicles = ["vehicle.audi.a2", "vehicle.audi.etron", "vehicle.audi.tt", "vehicle.bmw.grandtourer", "vehicle.citroen.c3", "vehicle.dodge.charger_2020", "vehicle.dodge.charger_police", "vehicle.dodge.charger_police_2020",
            "vehicle.ford.crown", "vehicle.ford.mustang", "vehicle.jeep.wrangler_rubicon", "vehicle.lincoln.mkz_2017", "vehicle.lincoln.mkz_2020", "vehicle.mercedes.coupe", "vehicle.mercedes.coupe_2020", "vehicle.micro.microlino",
            "vehicle.mini.cooper_s", "vehicle.mini.cooper_s_2021", "vehicle.nissan.micra", "vehicle.nissan.patrol", "vehicle.nissan.patrol_2021", "vehicle.seat.leon", "vehicle.tesla.model3"]
cycles = ["vehicle.harley-davidson.low_rider", "vehicle.kawasaki.ninja", "vehicle.vespa.zx125", " vehicle.yamaha.yzf", "vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"]
van = ["vehicle.ford.ambulance", "vehicle.mercedes.sprinter", "vehicle.volkswagen.t2", "vehicle.volkswagen.t2_2021"]
bus = ["vehicle.mitsubishi.fusorosa"]
truck = ["vehicle.carlamotors.firetruck", "vehicle.carlamotors.carlacola", "vehicle.carlamotors.european_hgv"]
```



