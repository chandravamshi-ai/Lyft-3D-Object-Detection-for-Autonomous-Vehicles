# Lyft-3D-Object-Detection-for-Autonomous-Vehicles
Build and optimize a model to detect 3d objects like car, buse etc.., based on a large-scale dataset. Dataset features the raw sensor camera inputs as perceived by a fleet of multiple, high-end, autonomous vehicles in a restricted geographic area.
![animated_car](https://level5.lyft.com/wp-content/uploads/2019/07/Top-down-sensor-diagram-smaller.png)
![real_car](https://level5.lyft.com/wp-content/uploads/2019/07/Copy-of-BP9I1484-1.jpg)

# Getting the data
* [Lyft](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data) - Dataset Link
# Dataset Content
* Dataset consist of train_data, test_data, train_images, test_images, train_lidar, test_lidar, test_maps, train_maps, train.csv, sample_submission.csv
* All are important folders but most important folder is train_data folder.
# Dataset Description
* scene - Consists of 25-45 seconds of a car's journey in a given environment. Each scence is composed of many samples.
* sample - A snapshot of a scence at a particular instance in time. Each sample is annoted with the objects present.
* sample_data - Contains the data collected from a particular sensor on the car.
* sample_annotation - An annotated instance of an object within our interest.
* instance - An enumeration of all object instance we observed.
* category - Taxonomy of object categories (e.g. vehicle, human).
* attribute - Property of an instance that can change while the category remains the same.
* visibility - (currently not used)
* sensor - A specific sensor type.
* calibrated sensor - Definition of a particular sensor as calibrated on a particular vehicle.
* ego_pose - Ego vehicle poses at a particular timestamp.
* log - Log information from which the data was extracted.
* map - Map data that is stored as binary semantic masks from a top-down view.
# What is LiDAR?
LiDAR (Light Detection and Ranging) is a method used to generate accurate 3D representations of the surroundings, and it uses laser light to achieve this. Basically, the 3D target is illuminated with a laser light (a focused, directed beam of light) and the reflected light is collected by sensors. The time required for the light to reflect back to the sensor is calculated.
# Visualizing the data
We will need the lyft_dataset_sdk library because it will help us visualize the image and LiDAR data easily. Only a simple pip install command is required. What it will do is it will filter the data and arange the whole data and it have some built-in functions which makes our work easier.

 ```!pip install lyft_dataset_sdk```

# Load the dataset using lyft_dataset_sdk
level5data = LyftDataset(data_path='../data/', json_path='../data/v1.01-train', verbose=True)
* replace data_path and json_path with your path where the data is stored
* we will get below output after running the above command correctly.
```
9 category,
18 attribute,
4 visibility,
18421 instance,
10 sensor,
148 calibrated_sensor,
177789 ego_pose,
180 log,
180 scene,
22680 sample,
189504 sample_data,
638179 sample_annotation,
1 map,
Done loading in 7.9 seconds.
======
Reverse indexing ...
Done reverse indexing in 2.9 seconds.
======  
```
The above output says we have these many number of particulars in that files.

# Understanding every file inside train_data folder
## scene.json
so we have 180 scenes and each scene is of 25-45 seconds. So  180*45(lets take 45)= 8100/60/60 is around 2.25 hours.
lets look at single scene and see what it contains

``` level5data.scene[0] ```

Oupt put is

```{'log_token': 'da4ed9e02f64c544f4f1f10c6738216dcb0e6b0d50952e158e5589854af9f100',
 'first_sample_token': '24b0962e44420e6322de3f25d9e4e5cc3c7a348ec00bfa69db21517e4ca92cc8',
 'name': 'host-a101-lidar0-1241893239199111666-1241893264098084346',
 'description': '',
 'last_sample_token': '2346756c83f6ae8c4d1adec62b4d0d31b62116d2e1819e96e9512667d15e7cec',
 'nbr_samples': 126,
 'token': 'da4ed9e02f64c544f4f1f10c6738216dcb0e6b0d50952e158e5589854af9f100'}
 ```
 Here scene is identified by ```token``` and this ```scene[0]``` has 126 samples which I will dicusses further which are represented by ```first_sample_token``` and ```last_sample_token```.
Remember through this enitre explanation of this dataset i will use only ```scene[0]``` which is identitfied by ```token``` value.
## sample.json
A sample is just screenshot of the surrounding at paritcular timestamp.It's simple. So don't think too much about it.
As we know we have 180 scenes and each scene has 126 samples, so 180*126 = 22680 samples are present in entire dataset.
level5data.sample will give all the samples in dataset but  as I said I will only ```scene[0]``` so this ```scene[0]``` has 126 samples
```
tes = []
for i in level5data.sample:
    if i['scene_token'] == 'da4ed9e02f64c544f4f1f10c6738216dcb0e6b0d50952e158e5589854af9f100':
        tes.append(i)
```
```len(tes)``` this will give 126 because one scene has 126 samples
```
{'next': 'c2ba18e4414ce9038ad52efab44e1a0a211ff1e6b297a632805000510756174d',
  'prev': '',
  'token': '24b0962e44420e6322de3f25d9e4e5cc3c7a348ec00bfa69db21517e4ca92cc8',
  'timestamp': 1557858039302414.8,
  'scene_token': 'da4ed9e02f64c544f4f1f10c6738216dcb0e6b0d50952e158e5589854af9f100',
  'data': {'CAM_BACK': '542a9e44f2e26221a6aa767c2a9b90a9f692c3aee2edb7145256b61e666633a4',
   'CAM_FRONT_ZOOMED': '9c9bc711d93d728666f5d7499703624249919dd1b290a477fcfa39f41b26259e',
   'LIDAR_FRONT_RIGHT': '8cfae06bc3d5d7f9be081f66157909ff18c9f332cc173d962460239990c7a4ff',
   'CAM_FRONT': 'fb40b3b5b9d289cd0e763bec34e327d3317a7b416f787feac0d387363b4d00f0',
   'CAM_FRONT_LEFT': 'f47a5d143bcebb24efc269b1a40ecb09440003df2c381a69e67cd2a726b27a0c',
   'CAM_FRONT_RIGHT': '5dc54375a9e14e8398a538ff97fbbee7543b6f5df082c60fc4477c919ba83a40',
   'CAM_BACK_RIGHT': 'ae8754c733560aa2506166cfaf559aeba670407631badadb065a9ffe7c337a7d',
   'CAM_BACK_LEFT': '01c0eecd4b56668e949143e02a117b5683025766d186920099d1e918c23c8b4b',
   'LIDAR_TOP': 'ec9950f7b5d4ae85ae48d07786e09cebbf4ee771d054353f1e24a95700b4c4af',
   'LIDAR_FRONT_LEFT': '5c3d79e1cf8c8182b2ceefa33af96cbebfc71f92e18bf64eb8d4e0bf162e01d4'},
  'anns': ['2a03c42173cde85f5829995c5851cc81158351e276db493b96946882059a5875',
   .........'c3c663ed5e7b6456ab27f09175743a551b0b31676dae71fbeef3420dfc6c7b09',
   ]}
   ```
   Each sample has its token , scene token which is token value of scene data, anns which we will discuss further
   
## sample_data.json
sample data is data of sample. Like each sample will get data from 3 lidars, and 7 cameras so in total we have 10 sensors.
so each sample will have 10 data. so 10 * 22680 samples = 226800 sample data but we have 189504 sample data because each sensor may not take the data.
```
tes2 = []
for i in level5data.sample_data:
    if i['sample_token'] == '24b0962e44420e6322de3f25d9e4e5cc3c7a348ec00bfa69db21517e4ca92cc8':
        tes2.append(i) 
```
```len(tes2),tes2[0:2]```  
* Note this sample_data is of scene[0] and sample is taken from this scene[0].
```
(10,
 [{'width': 1920,
   'height': 1080,
   'calibrated_sensor_token': '59155106c0ac5abe83cb6558ad8ce98400e3c3abf51234734bc89bc9d613470a',
   'token': '542a9e44f2e26221a6aa767c2a9b90a9f692c3aee2edb7145256b61e666633a4',
   'sample_token': '24b0962e44420e6322de3f25d9e4e5cc3c7a348ec00bfa69db21517e4ca92cc8',
   'is_key_frame': True,
   'prev': '',
   'fileformat': 'jpeg',
   'ego_pose_token': '0c257254dad346c9d90f7970ce2c0b8142f7c6e6a90716f4c0538cd2d2ef77d5',
   'timestamp': 1557858039200000.0,
   'next': '8d614daa8d1d48d3af4a0c817b676da1cb3e68f1432296eb52cfc428d0ff4d6d',
   'filename': 'images/host-a101_cam3_1241893239200000006.jpeg',
   'sensor_modality': 'camera',
   'channel': 'CAM_BACK'},
  {'width': 1920,
   'height': 1080,
   'calibrated_sensor_token': '4f30ede5a14a2644e870ae98a0f140c6c8e2d1507ecb82552ef66cd6fa8819f9',
   'token': '9c9bc711d93d728666f5d7499703624249919dd1b290a477fcfa39f41b26259e',
   'sample_token': '24b0962e44420e6322de3f25d9e4e5cc3c7a348ec00bfa69db21517e4ca92cc8',
   'is_key_frame': True,
   'prev': '',
   'fileformat': 'jpeg',
   'ego_pose_token': '0c257254dad346c9d90f7970ce2c0b8142f7c6e6a90716f4c0538cd2d2ef77d5',
   'timestamp': 1557858039250000.0,
   'next': 'b39f82985328728fdd22c94777ccdb44693009e90ce4d35504f9fe403763185b',
   'filename': 'images/host-a101_cam6_1241893239250000006.jpeg',
   'sensor_modality': 'camera',
   'channel': 'CAM_FRONT_ZOOMED'},])
```
* so this sample_data contains details about image which is taken from one of the 7 seven or 3 lidards

## sample_annotation.json
sample_annotation refers to any bounding box defining the position of an object seen in a sample(screenshot). All location data is given with respect to the global coordinate system. Let's look at the data
```
tes3 = []
for i in level5data.sample_annotation:
    if i['sample_token'] == '24b0962e44420e6322de3f25d9e4e5cc3c7a348ec00bfa69db21517e4ca92cc8':
        tes3.append(i) 
```
```len(tes3)``` output is 64

```
for i in level5data.sample:
    if i['token']=='24b0962e44420e6322de3f25d9e4e5cc3c7a348ec00bfa69db21517e4ca92cc8':
        print(len(i['anns']))
```
output is 64
so for every sample we have annotations this annotations detailed data will be save annotation.josn.
the above code have ```sample_token``` which is ```token``` value of sample.json in turn sample is takenfrom ```scene[0]```

 ```tes3[0]```
 ```
 {'token': '2a03c42173cde85f5829995c5851cc81158351e276db493b96946882059a5875',
  'num_lidar_pts': -1,
  'size': [1.997, 5.284, 1.725],
  'sample_token': '24b0962e44420e6322de3f25d9e4e5cc3c7a348ec00bfa69db21517e4ca92cc8',
  'rotation': [0.1539744509331139, 0, 0, 0.9880748293827983],
  'prev': '',
  'translation': [1048.155950230245, 1691.8102354006162, -23.304943447792454],
  'num_radar_pts': 0,
  'attribute_tokens': ['1ba8c9a8bda54423fa710b0af1441d849ecca8ed7b7f9393ba1794afe4aa6aa2',
   'daf16a3f6499553cc5e1df4a456de5ee46e2e6b06544686d918dfb1ddb088f6f'],
  'next': '9986dac1bcecb560153ab58ae7560028caeed3c1e067b37503cf50932e983afc',
  'instance_token': '695097711d9ec763b55f24d04ae9eb51289575645968f00723cee666a9f68c27',
  'visibility_token': '',
  'category_name': 'car'}
  ```
so sample annotatino has various field to which is used for bounding box.
so as usaly sample.annotations will have 'token' of smaple.json file labelled as usally as sample_json
sample_annotation is bounding box of objects in the sample(screenshot)
so each annotation will have again their own values
It has ```instance_token``` which we will discuss further.


## instance.json
Object instance are instances that need to be detected or tracked by an AV (e.g a particular vehicle, pedestrian).
our sample_annotation have ```instance_token``` which will match with ```token``` of instance.json
```
tes4 = []
for i in level5data.instance:
    if i['first_annotation_token'] == '2a03c42173cde85f5829995c5851cc81158351e276db493b96946882059a5875':
        tes4.append(i) 
```
```
tes4
[{'last_annotation_token': '36a6954dcf400dcca8a25353939f106c09d43fcd40a3ba18add54bd149e8afb5',
  'category_token': '8eccddb83fa7f8f992b2500f2ad658f65c9095588f3bc0ae338d97aff2dbcb9c',
  'token': '695097711d9ec763b55f24d04ae9eb51289575645968f00723cee666a9f68c27',
  'first_annotation_token': '2a03c42173cde85f5829995c5851cc81158351e276db493b96946882059a5875',
  'nbr_annotations': 92}]
```
so this instance has ```last_annotation_token``` ```last_annotation_token ``` these both will have same object. with bounding boxes in sample_annotatinos. this instance will have same object within a time.
```level5data.render_instance('695097711d9ec763b55f24d04ae9eb51289575645968f00723cee666a9f68c27')```
```level5data.render_annotation(my_instance['last_annotation_token'])```
```level5data.render_annotation(my_instance['first_annotation_token'])```
 the above commands will give same object like car or bus with bounding boxes.
 
## category.json
A category is the object assignment of an annotation. Let's look at the category table . The table contains the taxonomy of different object categories and also list the subcategories (delineated by a period).
 
 ```level5data.list_categories()```
```
Category stats
animal                      n=  186, width= 0.36±0.12, len= 0.73±0.19, height= 0.51±0.16, lw_aspect= 2.16±0.56
bicycle                     n=20928, width= 0.63±0.24, len= 1.76±0.29, height= 1.44±0.37, lw_aspect= 3.20±1.17
bus                         n= 8729, width= 2.96±0.24, len=12.34±3.41, height= 3.44±0.31, lw_aspect= 4.17±1.10
car                         n=534911, width= 1.93±0.16, len= 4.76±0.53, height= 1.72±0.24, lw_aspect= 2.47±0.22
emergency_vehicle           n=  132, width= 2.45±0.43, len= 6.52±1.44, height= 2.39±0.59, lw_aspect= 2.66±0.28
motorcycle                  n=  818, width= 0.96±0.20, len= 2.35±0.22, height= 1.59±0.16, lw_aspect= 2.53±0.50
other_vehicle               n=33376, width= 2.79±0.30, len= 8.20±1.71, height= 3.23±0.50, lw_aspect= 2.93±0.53
pedestrian                  n=24935, width= 0.77±0.14, len= 0.81±0.17, height= 1.78±0.16, lw_aspect= 1.06±0.20
truck                       n=14164, width= 2.84±0.32, len=10.24±4.09, height= 3.44±0.62, lw_aspect= 3.56±1.25
```

## attribute.json
An attribute is a property of an instance that may change throughout different parts of a scene while the category remains the same. Here we list the provided attributes and the number of annotations associated with a particular attribute.
```level5data.list_attributes()```
```
is_stationary: 321981
object_action_abnormal_or_traffic_violation: 2
object_action_driving_straight_forward: 244805
object_action_gliding_on_wheels: 165
object_action_lane_change_left: 1463
object_action_lane_change_right: 1370
object_action_left_turn: 5074
object_action_loss_of_control: 1
object_action_other_motion: 582
object_action_parked: 257939
object_action_reversing: 278
object_action_right_turn: 6694
object_action_running: 621
object_action_sitting: 586
object_action_standing: 5332
object_action_stopped: 94970
object_action_u_turn: 407
object_action_walking: 17890
```
## sensor.json
The Level 5 dataset consists of data collected from our full sensor suite which consists of:
1 x LIDAR, (up to three in final dataset)
7 x cameras
```level5data.sensor[0]```
```
{'modality': 'camera',
  'channel': 'CAM_FRONT_LEFT',
  'token': 'f7dad6bb70cb8e6245f96e5537e382848335872e6e259218b0a80cc071d162c4'}
```
## calibrated_sensor.json
calibrated_sensor consists of the definition of a particular sensor (lidar/camera) as calibrated on a particular vehicle. 
```level5data.calibrated_sensor[0]```
```
{'sensor_token': 'c84592e22beb2c0f14d5159245ce8d6678431b879e940eed580651c09cc7d2f1',
 'rotation': [-0.6784803059109364,
  0.6875255645268346,
  0.1910946403595628,
  -0.1745162199880262],
 'camera_intrinsic': [[882.42699274, 0, 602.047851885],
  [0.0, 882.42699274, 527.99972239],
  [0.0, 0.0, 1.0]],
 'translation': [1.0399186259366102, 0.30857658859026604, 1.65459751959659],
 'token': '80349b63ead8bfe5f4ce2cbe27fed9e4b5d699a5fc422232349263d21cd5eb70'}
 ```
## ego_pose.json
ego_pose contains information about the location (encoded in translation) and the orientation (encoded in rotation) of the ego vehicle body frame, with respect to the global coordinate system.
```level5data.ego_pose[0]```
```
{'rotation': [0.8045437502844562,
  0.02410792365998265,
  0.016871015279640935,
  0.5931639998672046],
 'translation': [622.726103039258, 3460.3733051620957, -6.813459358283256],
 'token': '2de7c52546f3f9bf6734084da8a2c1edaa0ffbea7f6e86f76a6ea66593d9a26b',
 'timestamp': 1547165667901671.8}
 ```
## log.json
The log table contains log information from which the data was extracted. A log record corresponds to one journey of our ego vehicle along a predefined route. Let's check the number of logs and the metadata of a log.
'log_token' from scene.json and 'token' form log.josn are soame
so we have 180 scenes and we have 180 log 
```print("Number of `logs` in our loaded database: {}".format(len(level5data.log)))``` output is 180
```
tes5 = []
for i in level5data.log:
    if i['token'] == 'da4ed9e02f64c544f4f1f10c6738216dcb0e6b0d50952e158e5589854af9f100':#this id is taken from scene[0].json
        tes5.append(i) 
 ```
 ```
[{'date_captured': '2019-05-14',
  'location': 'Palo Alto',
  'token': 'da4ed9e02f64c544f4f1f10c6738216dcb0e6b0d50952e158e5589854af9f100',
  'vehicle': 'a101',
  'logfile': '',
  'map_token': '53992ee3023e5494b90c316c183be829'}]
  ```
## map.json
Map information is currently stored in a 2D rasterized image. Let's check the number of maps and metadata of a map.
```print("There are {} maps masks in the loaded dataset".format(len(level5data.map)))```

output is 
```There are 1 maps masks in the loaded dataset```
