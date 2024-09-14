# Downloading patches from Google Earth Engine

Google Earth Engine (GEE) is a geospatial processing service. GEE allows users to run algorithms on georeferenced imagery and vectors stored on Google's infrastructure. The GEE API provides a library of functions which may be applied to data for display and analysis.With Earth Engine, you can perform geospatial processing at scale, powered by Google Cloud Platform. The purpose of Earth Engine is to:

1. Provide an interactive platform for geospatial algorithm development at scale
2. Enable high-impact, data-driven science
3. Make substantive progress on global challenges that involve large geospatial datasets

### 01. Libraries to be used:

```python
import ee
import geemap as emap
```

Selecting a specific basemap as follow:

```python
Map = emap.Map()
Map.add_basemap("SATELLITE")
```

### 02. Authentication and Initialization

Prior to using the Earth Engine Python client library, you need to authenticate and use the resultant credentials to initialize the Python client. To initialize, you will need to provide a project that you own, or have permissions to use. This project will be used for running all Earth Engine operations:

```python
ee.Authenticate()
ee.Initialize(project='ee-geoyons')
```

See the [authentication guide](https://developers.google.com/earth-engine/guides/auth) for troubleshooting and to learn more about authentication modes and Cloud projects.

### 03. Connecting to GEE

```python
tile = ee.List([30])

port_tiles = ee.FeatureCollection("projects/ee-geoyons/assets/tiles_portugal_50km")
label = ee.Image("projects/ee-geoyons/assets/raster_ardida_2017")
features = port_tiles.filter(ee.Filter.inList('id', tile));

# Define a collection
colS2 = 'COPERNICUS/S2_SR_HARMONIZED'

# NBR function
def addNBR (image):
  nbr = image.normalizedDifference(['B8', 'B12']).rename("NBR");
  return image.addBands(nbr);

# masking clouds
def maskSQA(image):
  qa = image.select('QA60')
  opaque_cloud = (1 << 10);
  cirrus_cloud = (1 << 11);
  mask = qa.bitwiseAnd(opaque_cloud).eq(0).And(qa.bitwiseAnd(cirrus_cloud).eq(0));
  return image.updateMask(mask);

cloud_cover = 10;

bnames = ['B2', 'B3', 'B4', 'B5','B6','B7','B8','B8A','B11','B12']

# Dates AFTER the fire event
start_date_after = '2017-09-30';
end_date_after = '2017-12-30';

img2017 = ee.ImageCollection(colS2)\
                .filterBounds(features)\
                .filterDate(start_date_after, end_date_after)\
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))\
                .map(maskSQA)\
                .select(bnames)\
                .reduce(ee.Reducer.median())

nbrColl = ee.ImageCollection(colS2)\
                .filterBounds(features)\
                .filterDate(start_date_after, end_date_after)\
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))\
                .map(maskSQA)\
                .select(bnames)\
                .map(addNBR)\
                .reduce(ee.Reducer.min());

nbr = nbrColl.select('NBR_min');

# Dates BEFORE the fire event
start_date_before = '2020-09-30'; # '2017-04-01';
end_date_before = '2020-12-30'; # '2017-06-15'

nbrColl2 = ee.ImageCollection(colS2)\
                .filterBounds(features)\
                .filterDate(start_date_before, end_date_before)\
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90))\
                .map(maskSQA)\
                .select(bnames)\
                .map(addNBR)\
                .reduce(ee.Reducer.mean());

nbr_before = nbrColl2.select('NBR_mean');

nbr_diff = nbr.subtract(nbr_before);
```

```python
vizParams = {
  "bands": ['B12_median', 'B8_median', 'B4_median'],
  "min": 0,
  "max": 3500,
  "gamma": [0.95, 1.1, 1]
  };

# Definimos los parametros de visualizacion
params_nbr = {
  "bands": ['NBR_min'],
  "min": -0.5,
  "max": 0.6,
  "palette": ['FFFFFF','CC9966','CC9900','996600', '33CC00', '009900','006600']};

# Definimos los parametros de visualizacion
params_nbr_before = {
  "bands": ['NBR_mean'],
  "min": -0.5,
  "max": 0.6,
  "palette": ['FFFFFF','CC9966','CC9900','996600', '33CC00', '009900','006600']};

Map.centerObject(features, 9)
Map.addLayer(features, {},'Portugal tile', True);
Map.addLayer(img2017.clip(features), vizParams, 'S2 - post-fire event 2017');
Map.addLayer(nbr_before.clip(features), params_nbr_before, 'NBR before', True);
Map.addLayer(nbr.clip(features), params_nbr, 'NBR after', True);
Map.addLayer(nbr_diff.clip(features), {min: -0.8, max: 0.2}, 'NBR Difference', True);
Map
```

```python
export_options = {
  'patchDimensions': [512, 512], #512*512
  'maxFileSize': 104857600,
  'compressed': True
}

ee.batch.Export.image({
  "image": d3.clip(features).double(),
  "description": 'PatchesExport',
  "fileNamePrefix": 'Port_tile30',
  "scale": 10, # resolution
  "folder": 'California_patches_512-512_radar',
  "fileFormat": 'TFRecord',
  "region": features,
  "formatOptions": export_options,
  "maxPixels": 1e+13,
})
```
