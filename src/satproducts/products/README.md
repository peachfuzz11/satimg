# SAR Processing Levels Overview

This document provides an overview of the Synthetic Aperture Radar (SAR) data processing levels available from various satellite providers, including Capella Space, ICEYE, KOMPSAT-5, TerraSAR-X, and Umbra. It outlines the different levels of processing applied to SAR data, their use cases, and the file formats they are delivered in.

## Types of SAR Processing Levels

SAR data typically goes through several stages of processing, transforming raw radar signals into useful geospatial imagery. The processing levels fall into three main categories: **Complex**, **Detected**, and **Geocoded**.

### Complex SAR Data

Complex SAR imagery includes both amplitude and phase information, offering the most detailed data for advanced analysis such as interferometry, polarimetry, and change detection.

- **Single Look Complex (SLC):**  
  Provides complex SAR data in a raw format, ideal for tasks such as interferometry and polarimetry. Available in file formats like GeoTIFF, HDF5, and COSAR.
  
- **Sensor Independent Complex Data (SICD):**  
  Provides complex SAR data in a standardized format with associated metadata, supporting further processing. Delivered in the NITF file format.
  
- **Compensated Phase History Data (CPHD):**  
  Provides phase history information in a standardized format. Suitable for studying terrain elevation changes and target motion. Delivered in the CPHD file format.

### Detected SAR Data

Detected SAR imagery focuses on the amplitude information and removes background noise, providing a clearer picture of terrain, infrastructure, and other physical features.

- **Ground Range Detected (GRD):**  
  Offers detected SAR data in a raw format. It is suitable for tasks such as mapping and land cover classification. Delivered in GeoTIFF or HDF5 formats.
  
- **Sensor Independent Derived Data (SIDD):**  
  Provides detected SAR data with added metadata in a standardized format. This is suitable for cases where metadata and interoperability are important. Delivered in the NITF format.

### Geocoded SAR Data

Geocoded SAR imagery is mapped to a coordinate reference system (CRS), making it easier to integrate with other geographic datasets.

- **Geocoded Earth Corrected (GEC):**  
  Provides SAR data projected to a CRS but may contain distortions from the terrain. Useful for basic visualization and large-scale mapping. Available in GeoTIFF or HDF5 formats.
  
- **Geocoded Terrain Corrected (GTC):**  
  Provides geocoded SAR data corrected for terrain using a digital elevation model (DEM). Ideal for precise quantitative analysis, topographic mapping, and environmental monitoring. Available in GeoTIFF or HDF5 formats.

## Available SAR Data Collections

Below is a summary of the SAR processing levels available across several SAR data providers.

| **Provider**    | **Complex**         | **Detected**     | **Geocoded**     |
|-----------------|---------------------|------------------|------------------|
| **Capella Space**| SLC, SICD           | Not supported    | GEC, GTC         |
| **ICEYE**       | SLC                 | GRD              | Not supported    |
| **KOMPSAT-5**   | SLC                 | Not supported    | GEC, GTC         |
| **TerraSAR-X**  | SLC                 | GRD              | GEC, GTC         |
| **Umbra**       | SICD, CPHD          | SIDD             | GEC              |




