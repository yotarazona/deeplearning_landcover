<!-- #region -->
# **Projects**

## **01: Research**

[Investigating the Atmospheric Impact of Disturbances in Land Cover: A Study of Precipitation and Temperature Changes in post-fire landscapes](#01) (PhD project)

[Remotely sensed variables for land change analysis: opportunities for Land Use and Land Cover mapping](#02) (paper ongoing)

## **02: Open-source software development**

[scikit-eo: A Python package for Remote Sensing Data Analysis](#03)

[forestools: Tools for detecting deforestation and forest degradation](#04)

[ForesToolsRS (R Package): Remote Sensing Tools for Forest Monitoring](#05)

## <a name = "01"></a>**Investigating the Atmospheric Impact of Disturbances in Land Cover: A Study of Precipitation and Temperature Changes in post-fire landscapes** (PhD project)

**Supervisors**: [Vasco M. Mantas](http://www.linkedin.com/in/vascomantas) (UC/CITEUC) / [Courtney Schumacher](https://scholar.google.com/citations?user=_n7c6w0AAAAJ&hl=en) (Texas A&M University)

### Abstract
The PhD research will address for the first time the impacts of post-fire events and the dynamics of Land Cover and Land Use Changes (LCLUC) on regional temperature and precipitation patterns by combining several remotely sensed data such as Landsat/Sentinel-2 and Sentinel-1, GMP and SMAP, ground-truth, and artificial intelligence algorithms for a comprehensive analysis of the feedback and interaction processes between surface and atmosphere. The findings of these interactions, atmosphere and land cover, will enable a new understanding of post-fire landscapes and contribute to an informed post-disaster recovery process.

### State of the art
Human actions are driving significant changes in land cover, intensifying climate change challenges and posing critical threats to communities and ecosystems [1]. These changes arise from various processes and drivers, operating at different scales [1]. It is crucial to capture these diverse processes explicitly in space and time to support scientific and societal applications that go beyond mere geographic representation of disturbances [2]. Moreover, studies like [3] and [4] have revealed that regional-scale and large-scale tropical deforestation can result in substantial reductions in average precipitation and higher surface temperatures across all regions.
Wildland fires offer a unique chance to study the role of abrupt land cover change, gradual recovery, and the establishment of new equilibria. Satellite remote sensing provides valuable data on fire location, magnitude, and recovery ([5], [6]). Research has focused on direct and indirect impacts, including air and water quality deterioration and compound disasters linked to fire landscapes [7]. However, post-fire surface-atmosphere interactions are largely unexplored, separate from fire weather. Land cover affects precipitation patterns and the performance of climate models (e.g., [8]), yet our knowledge of post-fire feedbacks is limited. Surface roughness, albedo, emissivity, wind, and soil moisture changes have significant direct and indirect effects on precipitation [9]. These processes, operating at various scales, require disentanglement for a better understanding of land use and land cover's role in feedback mechanisms.
The knowledge gaps extend to the very understanding of product performance in such areas, where the disturbances create challenging conditions for commonly used algorithms [10]. This gap, undermines a clear understanding of how fire landscapes influence regional temperature and precipitation patterns and work to amplify or alleviate droughts/flooding and vegetation recovery/replacement. With climate change enabling more frequent and catastrophic fires, the role of post-fire landscape weather research may only become more relevant, and part of the planning towards more resilient ecosystems and communities. 
The 2017 fires in Portugal covered 442,418 hectares [11], resembling events in California, occurring in similar climate zones but in different pre-fire landscapes. Studying burn scars and their regeneration can provide insights into post-fire precipitation and temperature dynamics at local to regional scales. Leveraging the development of remote sensing missions such as Landsat, GPM, and SMAP and retrieval techniques, satellite-based remote sensing will provide the data needed to determine the contribution of different surface and atmospheric variables to the complex processes of precipitation generation and recycling [12].
In this project, we will design an innovative framework to characterize and quantify the impact of wildland fires, and the slow regeneration/transitions of post-fire landscapes over the local and regional precipitation and surface temperature patterns. To do so, a complex set of information from different missions will be acquired and processed using traditional and machine-/deep-learning for a comprehensive analysis of the feedback processes.

### Objectives

**General objective**: Characterize and quantify the impact of wild fires, and regeneration of post-fire landscapes on surface-atmosphere interactions, with a focus on precipitation and surface temperature.

- Specific objective 1: Design a framework and processing chain to accurately characterize burn scars for surface-atmosphere interactions (including ground validation of the used satellite products).
- Specific objective 2: Characterize surface temperature and soil moisture (local to regional) in post-fire landscapes.
- Specific objective 3: Characterize precipitation feedbacks on post-fire landscapes and its impact at a micro and mesoscale (1-1000 km) scale.

### References
- [1] Li, J.; Li, Z.L.; Wu, H.; You, N. Trend, seasonality, and abrupt change detection method for land surface temperature time-series analysis: Evaluation and improvement. Remote Sens. Environ. 2022, 280, 113222.
- [2] Mantas, V.; Fonseca, L.; Baltazar, E.; Canhoto, J.; Abrantes, I. Detection of Tree Decline (Pinus pinaster Aiton) in European Forests Using Sentinel-2 Data. Remote Sens. 2022, 14.
- [3] Wang, X., Cong, P., Jin, Y., Jia, X., Wang, J., Han, Y. Assessing the Effects of Land Cover Land Use Change on Precipitation Dynamics in Guangdong–Hong Kong–Macao Greater Bay Area from 2001 to 2019. Remote Sensing. 2021; 13(6), 1135. 
- [4] Pal., S., Ziaul, Sk. Detection of land use and land cover change and land surface temperature in English Bazar urban center. The Egyptian Journal of Remote Sensing and Space Science, 20 (1), 2017, 125-145.
- [5] Boschetti, L., Roy, D. P., Giglio, L., Huang, H., Zubkova, M., & Humber, M. L. (2019). Global validation of the collection 6 MODIS burned area product. Remote Sensing of Environment, 235(February), 111490.
- [6] Quintano, C., Fernandez-Manso, A., & Roberts, D. A. (2017). Burn severity mapping from Landsat MESMA fraction ima-ges and Land Surface Temperature. Remote Sensing of Environment, 190, 83–95.
- [7] Wagenbrenner, N. S., Chung, S. H., & Lamb, B. K. (2017). A large source of dust missing in particulate matter emission inventories? Wind erosion of post-fire landscapes. Elementa, 5.
- [8] Saavedra, M.; Junquas, C.; Espinoza, J.; Silva, Y. Impacts of topography and land use changes on the air surface temperature and precipitation over the central Peruvian Andes. Atmos. Res. 2020, 234, 104711.
- [9] Graf, M.; Arnault, J.; Fersch, B.; Kunstmann, H. Is the soil moisture precipitation feedback enhanced by heterogeneity and dry soils? A comparative study. Hydrol. Process. 2021, 35, 1–15.
- [10] Ermida, S. L., Soares, P., Mantas, V.M., Göttsche, F. M., & Trigo, I. F. (2020). Google earth engine open-source code for land surface temperature estimation from the landsat series. Remote Sensing, 12(9), 1–21. 
- [11] ICNF, 10 º Relatório provisório de incêndios florestais, 2017, Accessed June 2023 at https://www.icnf.pt/api/file/doc/2c45facee8d3e4f8.
- [12] Shen, Z., Yong., B. Downscaling the GPM-based satellite precipitation retrievals using gradient boosting decision tree approach over Mainland China. Journal of Hydrology, 602, 2021, 126803.

## <a name = "02"></a>**Remotely sensed variables for land change analysis: opportunities for Land Use and Land Cover mapping** (paper ongoing)

### Introduction

The Earth's surface is changing at an unprecedented rate due to climate change and various Land Cover and Land Use (LCLU) disturbances. Land change science, defined as the interdisciplinary field that seeks to understand the dynamics of LCLU as a coupled human-environment system to address theory, concepts, models, and applications relevant to environmental and societal problem has many components, in which one of the most fundamental and critical components is the observation, monitoring, and characterization of land change (Zhu et al., 2022). Therefore, assessing and monitoring vegetation dynamics is a key and essential requirement for research and better understanding of global land change.
LCLU can be used to analyze and monitor a variety of complex environmental processes and impacts such as characterization of Andean ecosystems (Tovar et al., 2013, Chimner et al. 2019), natural disaster management (Kucharczyk & Hugenholtz, 2021), crop classifications (Pott et al., 2021), forest disturbance (Reiche et al., 2015, Tarazona et al., 2018), forest degradation (Tarazona and Miyasiro-López, 2020), biodiversity conservation (Cavender-Bares et al. 2022), climate change impacts (Yang et al., 2013), among others.

When collecting remotely sensed data, it is most common to consider as many variables (called multivariate variables) as possible during an LCLU analysis. However, the greater the variability of the dataset, the more difficult it becomes to visualize the behavior of each of the variables, their contribution and their interaction with each other. Even more so today, with the availability of large satellite image databases, this selection should be a priority to reduce computational costs. Although, the constant development of cloud computing platforms provides the ability to not only process large volumes of satellite information through parallel computing, but also enables the use of large online data catalogs improving the capability to address these environmental issues. Even so, with this computational advance, it is always better to work with variables that have significant contributions than with passive variables without much importance.

Remote sensing plays an important role in this effort to map changes in the earth's surface through a combination of existing instruments and data (Muraoka and Koizumi, 2009; Palacios-Orueta et al., 2012). A fundamental advantage of remote sensing, thanks to the principles of repeatability, objectivity and coherence, is to provide synoptic measurements of the Earth's surface at different spectral, spatial and temporal resolutions and allows the study of land change. Satellite-based optical and radar remote sensing has been used to monitor land change combining only optical spectral bands on different dates (Serra et al., 2008; Amani et al., 2017), combining optical spectral bands with vegetation indices (Silva Junior, et al., 2020, Liu et al., 2020), with radar polarizations and topographic variables (Chimner et al. 2019, Tarazona et al., 2021) or applying dimensionality reduction with Principal Component Analysis (Tarazona et al., 2021) for enhanced accuracy, to name a few some examples of remotely sensed variables. Although in recent years, artificial intelligence, especially deep learning models have grown in popularity in satellite remote sensing, showing better performance over classical machine learning algorithms in land cover classification, we still lack two crucial aspects to further improve the ability to monitor land cover change. First, there is a lack of standardization of the variables that are used as input for any artificial intelligence model, i.e., although the exact combination of satellite variables and so on is not possible to know, but at least to have an idea which are the optimal ones available to increase the accuracy of land change mapping. Secondly, there is a need for a data catalog with these standardized variables hosted on platforms such as Google Earth Engine so that other users can use these datasets to characterize and analyze global land cover change.

Therefore, this research will address different objectives: i) to define which geospatial variables are the most suitable as input for artificial intelligence algorithms in order to improve the accuracy of global land characterization and change mapping, ii) discuss the contributions of the different variables in land change analysis, and iii) develop a first version of a catalog of datasets hosted on Google Earth Engine (GEE) for land cover classification tasks. To address these objectives, different datasets (variables) such as Sentinel-2 and Sentinel-1, vegetation indices and Digital Elevation Models (DEM) will be used. Additionally, Deep Learning classification method through semantic segmentation with Convolutional Neural Networks (CNN), Principal Component Analysis (PCA) for dimensionality reduction, ground-truth data for validation of classifications, and various metrics such as overall accuracy, estimating area using confusion matrix, omission and commission will be used. This metrics will allow measuring the error or accuracy of classification models with different predictor variables.


## <a name = "03"></a>**scikit-eo: A Python package for Remote Sensing Data Analysis**

## Highlights

<img src="https://raw.githubusercontent.com/yotarazona/scikit-eo/main/docs/images/scikit-eo_logo.jpg" align="right" width="220"/>

[scikit-eo](https://yotarazona.github.io/scikit-eo/) is an open-source package built entirely in Python, through Object-Oriented Programming and Structured Programming, that provides a useful variety of remote sensing tools, from basic and exploratory functions to more advanced methods to classify, calibrate, or fusing satellite imagery. Depending on users' needs, **scikit-eo** can provide the basic but essential land cover characterization mapping including the confusion matrix and the required metrics such as user's accuracy, producer's accuracy, omissions and commissions errors. The combination of these required metrics can be obtained in a form of a pandas ```DataFrame``` object. Furthermore, a class prediction map as a result of land cover mapping, i.e., a land cover map which represents the output of the classification algorithm or the output of the segmentation algorithm. These two outcomes must include uncertainties with a confidence levels (e.g., at $95$% or $90$%). There are other useful tools for remote sensing analysis that can be found in this package, for more information about the full list of the supported function as well as how to install and execute the package within a python setting, visit the [scikit-eo](https://yotarazona.github.io/scikit-eo/) website.

## Audience

**scikit-eo** is a versatile Python package designed to cover a wide range of users, including students, professionals of remote sensing, researchers of environmental analysis, and organizations looking for satellite image analysis. Its comprehensive features make it well-suited for various applications, such as university teaching, that include technical and practical sessions, and cutting-edge research using the most recent machine learning and deep learning techniques applied to the field of remote sensing. Whether the user are students seeking to get insights from a satellite image analysis or a experienced researcher looking for advanced tools, **scikit-eo** offers a valuable resource to support the most valuable methods for environmental studies.

## <a name = "04"></a>**forestools: Tools for detecting deforestation and forest degradation**

## Introduction

<img src="https://raw.githubusercontent.com/yotarazona/forestools/main/forestools/figures/img_readme.png" align="right" width="350"/>

**forestools** is a Python package mainly focused on mapping and monitoring deforestation, although it can be used for monitoring forest degradation or detecting early warnings. The detection algorithm embedded in this package is a non-seasonal detection approach - unlike seasonal algorithms - that does not model the seasonal component of a time series, is intuitive, has only one calibration parameter, and can be run with vegetation indices such as NDVI and EVI, photosynthetic vegetation from CLASlite software, with radar polarizations, and with NDFI fraction indices. In fact, this package includes an algorithm that is capable of obtaining NDFI indices, which until now was only possible to obtain from Google Earth Engine.

**forestools** is intended for students, professionals, researchers, and organizations dedicated to forest monitoring and assessment, and any public interested in mapping the changes experienced by the different forests on the planet due to anthropogenic disturbances but also to minor natural disturbances.

#### Citation

This repository is part of the paper *Mapping deforestation using fractions indices and the PVts-beta approach* submitted to **IEEE Geoscience and Remote Sensing Letters**.

Please, to cite the forestools package in publications, use this paper:

Tarazona, Y. (2021). Mapping deforestation using fractions indices and the PVts-beta approach, *IEEE Geoscience and Remote Sensing Letters*, DOI: [10.1109/LGRS.2021.3137277](https://ieeexplore.ieee.org/document/9656901).

## <a name = "01"></a>**ForesToolsRS (R Package): Remote Sensing Tools for Forest Monitoring**
<!-- #endregion -->
