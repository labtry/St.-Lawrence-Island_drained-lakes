var image = ee.Image("JRC/GSW1_4/GlobalSurfaceWater"),
var image2 = ee.Image("users/image/Water_loss"),
var aoi = ee.FeatureCollection("users/GRL_2024/Round1/StudyArea"),

var water = image.select("max_extent").clip(aoi).selfMask().rename("water");
//Map.addLayer(water, {palette: 'blue'}, 'water',false);

var objectId = water.connectedComponents({connectedness: ee.Kernel.plus(1),maxSize: 1024});
Map.addLayer(objectId.randomVisualizer(), null, 'Objects',false);
var objectSize = objectId.select('labels').connectedPixelCount({maxSize: 1024, eightConnected: false});
var pixelArea = ee.Image.pixelArea();
var objectArea = objectSize.multiply(pixelArea);
var areaMask = objectArea.gte(10000);
objectId = objectId.updateMask(areaMask);
Map.addLayer(objectId.select("water"), {palette: 'blue'}, 'Large water',false);

var waterloss_1 = image.select("transition").clip(aoi).eq(3).selfMask().rename("waterloss");
var waterloss_2 = image2.clip(aoi).eq(1).selfMask().rename("waterloss");
var waterloss = ee.ImageCollection([waterloss_1,waterloss_2]).mosaic();


var newobject = objectId.addBands(waterloss);
print(newobject);
var patchTemp = newobject.reduceConnectedComponents({
  reducer: ee.Reducer.count(),//ee.Reducer.sum(),
  maxSize: 512,
  labelBand: 'labels'
});

print(patchTemp);
var prop = patchTemp.select("waterloss").divide(patchTemp.select("water"));

var prop_lt50p = prop.mask(prop.gt(0.5));
var patch_lt10 = patchTemp.select("waterloss").mask(patchTemp.select("waterloss").gt(111));

var losspatch = ee.ImageCollection([prop_lt20p,patch_lt10]).mosaic();

losspatch = prop.mask(losspatch);

Map.addLayer(losspatch,{palette: ['yellow']},'losspatch');

Map.addLayer(waterloss, {palette:'FF0000'}, 'waterloss');

print(losspatch);

// WKT projection description for https://epsg.io/102017
var proj = ee.Projection(
    'PROJCS["North_Pole_Lambert_Azimuthal_Equal_Area",'+
    '    GEOGCS["GCS_WGS_1984",'+
    '        DATUM["WGS_1984",'+
    '            SPHEROID["WGS_1984",6378137,298.257223563]],'+
    '        PRIMEM["Greenwich",0],'+
    '        UNIT["Degree",0.017453292519943295]],'+
    '    PROJECTION["Lambert_Azimuthal_Equal_Area"],'+
    '    PARAMETER["False_Easting",0],'+
    '    PARAMETER["False_Northing",0],'+
    '    PARAMETER["Central_Meridian",0],'+
    '    PARAMETER["Latitude_Of_Origin",90],'+
    '    UNIT["Meter",1],'+
    '    AUTHORITY["EPSG","102017"]]');
	
Export.image.toDrive({
      image: losspatch,
      description: "losspatch",
      fileNamePrefix: "losspatch",
      scale: 30,
      //region: aoi,
      skipEmptyTiles: true,
      crs: proj,
      maxPixels: 1e13
    });
var lossobject = losspatch.gt(0);
lossobject = lossobject.updateMask(lossobject);
Export.image.toDrive({
      image: lossobject,
      description: "lossobject",
      fileNamePrefix: "lossobject",
      scale: 30,
      //region: aoi,
      skipEmptyTiles: true,
      crs: proj,
      maxPixels: 1e13
    });

var losspixel = waterloss.updateMask(lossobject);

Export.image.toDrive({
      image: losspixel,
      description: "losspixel",
      fileNamePrefix: "losspixel",
      scale: 30,
      //region: aoi,
      skipEmptyTiles: true,
      crs: proj,
      maxPixels: 1e13
    });
