var aoi = ee.FeatureCollection("users/GRL_2024/Round2/drained_lakes"),

Map.centerObject(aoi, 9);
Map.addLayer(aoi, {palette: 'gray'}, 'aoi');

var dataset = ee.ImageCollection('NASA/ORNL/DAYMET_V4')
                  .filter(ee.Filter.date('2000-12-01', '2021-12-31'))
                  .select('tmax','prcp','swe');

var years = ee.List.sequence(2001, 2021);

var computeYearlyMean = function(year) {
  var yearStart = ee.Date.fromYMD(year, 1, 1);
  var yearEnd = ee.Date.fromYMD(year, 12, 31);
  var yearlyMean = dataset.filterDate(yearStart, yearEnd).mean();
  yearlyMean = yearlyMean.rename("tmax_ann","precip_ann","swe_ann");
  return yearlyMean.set('year', year);
};

var DAYMET_YearlyMean = ee.ImageCollection.fromImages(years.map(computeYearlyMean));
print(DAYMET_YearlyMean);


var computeWinterMean = function(year) {
  var year_pre = ee.Number(year).subtract(1);
  var yearStart = ee.Date.fromYMD(year_pre, 12, 1);
  var yearEnd = ee.Date.fromYMD(year, 2, 28);
  var WinterMean = dataset.filterDate(yearStart, yearEnd).mean();
  WinterMean = WinterMean.rename("tmax_win","precip_win","swe_win");
  return WinterMean.set('year', year);
};

var DAYMET_winmerMean = ee.ImageCollection.fromImages(years.map(computeWinterMean));
//print(DAYMET_winmerMean);


var computeSpringMean = function(year) {
  var yearStart = ee.Date.fromYMD(year, 3, 1);
  var yearEnd = ee.Date.fromYMD(year, 5, 31);
  var SpringMean = dataset.filterDate(yearStart, yearEnd).mean();
  SpringMean = SpringMean.rename("tmax_spr","precip_spr","swe_spr");
  return SpringMean.set('year', year);
};

var DAYMET_sprmerMean = ee.ImageCollection.fromImages(years.map(computeSpringMean));
//print(DAYMET_sprmerMean);

var computeSummerMean = function(year) {
  var yearStart = ee.Date.fromYMD(year, 6, 1);
  var yearEnd = ee.Date.fromYMD(year, 8, 31);
  var SummerMean = dataset.filterDate(yearStart, yearEnd).mean();
  SummerMean = SummerMean.rename("tmax_sum","precip_sum","swe_sum");
  return SummerMean.set('year', year);
};

var DAYMET_SummerMean = ee.ImageCollection.fromImages(years.map(computeSummerMean));
//print(DAYMET_SummerMean);


var computeAutumnMean = function(year) {
  var yearStart = ee.Date.fromYMD(year, 9, 1);
  var yearEnd = ee.Date.fromYMD(year, 11, 30);
  var AutumnMean = dataset.filterDate(yearStart, yearEnd).mean();
  AutumnMean = AutumnMean.rename("tmax_atm","precip_atm","swe_atm");
  return AutumnMean.set('year', year);
};

var DAYMET_atmmerMean = ee.ImageCollection.fromImages(years.map(computeAutumnMean));
//print(DAYMET_atmmerMean);


var mergedCollection = DAYMET_YearlyMean.map(function(image1) {
  var image2 = DAYMET_sprmerMean.filter(ee.Filter.eq('year', image1.get('year'))).first();
  var mergedImage = image1.addBands(image2);
  return mergedImage;
});

mergedCollection = mergedCollection.map(function(image1) {
  var image2 = DAYMET_SummerMean.filter(ee.Filter.eq('year', image1.get('year'))).first();
  var mergedImage = image1.addBands(image2);
  return mergedImage;
});

mergedCollection = mergedCollection.map(function(image1) {
  var image2 = DAYMET_atmmerMean.filter(ee.Filter.eq('year', image1.get('year'))).first();
  var mergedImage = image1.addBands(image2);
  return mergedImage;
});

mergedCollection = mergedCollection.map(function(image1) {
  var image2 = DAYMET_winmerMean.filter(ee.Filter.eq('year', image1.get('year'))).first();
  var mergedImage = image1.addBands(image2);
  return mergedImage;
});

print(mergedCollection);

var newCol = ee.ImageCollection(mergedCollection.map(function(img) {
  var year = img.get('year');
  var yr = ee.Image.constant(ee.Number(year)).toShort();
  return ee.Image.cat(yr, img).set('year', year);
}));

var bandNames = ["tmax_ann", "precip_ann", "swe_ann", "tmax_spr", "precip_spr", "swe_spr", "tmax_sum", "precip_sum", "swe_sum", "tmax_atm", "precip_atm", "swe_atm", "tmax_win", "precip_win", "swe_win"];

var slopes = ee.List(bandNames.map(function(band) {
  var sensSlope = newCol.select(["constant",band]).reduce(ee.Reducer.sensSlope()).select("slope");
  return sensSlope.rename(band+'_slope');
}));

var slope_img = ee.ImageCollection.fromImages(slopes).toBands();

print(slope_img);


var result=slope_img.reduceRegions({
    collection: aoi,
    reducer: ee.Reducer.mean(),
    scale: 1000,
    tileScale: 8
  });

print(result);

Export.table.toDrive({
  collection: result,
  description: 'Extracted_climate_Trends',
  fileFormat: 'CSV'
});

 