# RADARSAT Constellation Mission (RCM): An Introduction

## Overview

The **RADARSAT Constellation Mission (RCM)** is Canada's flagship Earth observation program, designed to provide continuous and reliable radar imaging capabilities for a wide variety of applications, including environmental monitoring, disaster response, maritime surveillance, and defense. Managed by the **Canadian Space Agency (CSA)**, RCM is the successor to the highly successful RADARSAT-1 and RADARSAT-2 missions, and brings enhanced imaging capabilities with its three-satellite constellation.

RCM uses **Synthetic Aperture Radar (SAR)** technology, which operates in the C-band frequency range (5.4 GHz). This technology allows RCM to capture images regardless of weather conditions or daylight, making it an essential tool for all-weather, day-and-night observation. 

Launched in **June 2019**, the constellation provides global coverage and revisits the same area several times per day, making it an invaluable resource for both real-time monitoring and long-term studies.

## Key Features

1. **Three-Satellite Constellation**: 
    - RCM consists of three identical satellites that work together to provide comprehensive radar coverage of Earth’s surface. The constellation ensures shorter revisit times and enhanced coverage, particularly for Canadian regions, the Arctic, and maritime approaches.

2. **C-band SAR**: 
    - The SAR payload on each satellite operates in the C-band, offering reliable imaging capabilities in a wide range of operational modes, with resolutions ranging from 3 meters to 100 meters.

3. **All-Weather, Day-and-Night Observation**: 
    - One of the key advantages of SAR technology is its ability to operate under any weather conditions, including clouds, fog, and darkness. This makes RCM an essential tool for continuous Earth observation, especially in regions with frequent cloud cover.

4. **Global Monitoring**:
    - RCM is capable of providing global radar coverage, with a particular focus on monitoring Canadian territories and surrounding maritime zones. Its rapid revisit times enable near real-time monitoring of dynamic processes, such as ice movement, ship detection, and environmental changes.

5. **Enhanced Maritime Surveillance**:
    - RCM offers significant enhancements for maritime domain awareness. It is designed to detect and track vessels, monitor ice conditions, and support search and rescue operations. The mission includes an Automatic Identification System (AIS) that can track ships in real time.

6. **Environmental Monitoring and Climate Change**:
    - RCM plays a crucial role in environmental protection and climate change studies. It provides detailed data on deforestation, wetlands mapping, coastal erosion, and ice cover, supporting sustainable resource management and conservation efforts.

## Mission Objectives

The RCM mission is centered around three key objectives:

1. **Maritime Surveillance**:
    - RCM supports Canada’s maritime domain awareness by monitoring ship traffic, illegal fishing activities, oil spills, and ice hazards. This includes support for the **Canadian Coast Guard** and **Department of National Defence** in protecting Canadian waters.

2. **Disaster Management**:
    - The constellation helps in disaster response by providing critical data for flood mapping, forest fire monitoring, and earthquake damage assessment. This data allows for timely intervention and mitigation efforts.

3. **Ecosystem Monitoring**:
    - RCM supports environmental monitoring, with a focus on forestry, agriculture, wetlands, and the Arctic. The data is used to track changes in natural ecosystems, map vegetation growth, and monitor land-use changes over time.

## Satellite Overview and Technical Specifications

| **Feature**                   | **Details**                                          |
|-------------------------------|------------------------------------------------------|
| **Launch Date**                | June 12, 2019                                       |
| **Number of Satellites**       | 3 (identical configuration)                         |
| **Orbit Type**                 | Sun-synchronous, near-polar orbit                   |
| **Orbit Altitude**             | 600 km                                              |
| **Revisit Time**               | Daily over most of Canada, up to 4 times/day        |
| **Radar Frequency**            | C-band (5.405 GHz)                                  |
| **Resolution**                 | 3 m to 100 m                                        |
| **Swath Width**                | 20 km to 500 km                                     |
| **Polarization**               | Single, Dual, and Quad Polarization                 |
| **Imaging Modes**              | Spotlight, Stripmap, ScanSAR, Maritime Surveillance |
| **Data Format**                | GeoTIFF, NITF                                       |




# RADARSAT Constellation Mission (RCM) Imaging Modes

The **RADARSAT Constellation Mission (RCM)** is designed to provide versatile and high-quality radar imaging capabilities for Earth observation. With three satellites operating in **C-band SAR**, RCM offers several imaging modes to meet a variety of scientific, environmental, and defense requirements. These modes differ in spatial resolution, swath width, polarization options, and incidence angle ranges. The choice of imaging mode depends on the specific application, from fine-detail urban mapping to wide-area maritime surveillance.

## Table of Contents
1. [Overview of RCM SAR Modes](#overview-of-rcm-sar-modes)
2. [Spotlight Mode](#spotlight-mode)
3. [Medium Resolution (Stripmap) Mode](#medium-resolution-stripmap-mode)
4. [High Resolution (Stripmap) Mode](#high-resolution-stripmap-mode)
5. [ScanSAR Mode](#scansar-mode)
6. [Extended High (EH) Mode](#extended-high-eh-mode)
7. [Maritime Surveillance Mode (MMS)](#maritime-surveillance-mode-mms)
8. [Polarization Options](#polarization-options)
9. [Data Format and Data Size](#data-format-and-data-size)
10. [Conclusion](#conclusion)

---

## 1. Overview of RCM SAR Modes

The **RCM SAR system** offers multiple imaging modes, each designed to capture data over different spatial resolutions and swath widths. The most commonly used modes include:

- **Spotlight Mode**: High-resolution imaging for detailed observations.
- **Medium Resolution Mode (Stripmap)**: Balance between resolution and swath width.
- **High Resolution Mode (Stripmap)**: Improved resolution for more detailed imagery.
- **ScanSAR Mode**: Wide-area coverage at the expense of resolution.
- **Extended High Mode**: Wide swath with improved resolution.
- **Maritime Surveillance Mode (MMS)**: Optimized for ocean surveillance, detecting ships and ice.

---

## 2. Spotlight Mode

The **Spotlight Mode** is the highest-resolution imaging mode offered by RCM. In this mode, the radar beam is steered electronically to dwell longer over a specific area, resulting in fine spatial detail.

| Parameter                 | Value                                    |
|---------------------------|------------------------------------------|
| **Resolution**             | 0.5 m to 1 m                            |
| **Swath Width**            | 10 km                                   |
| **Incidence Angle Range**  | 20° to 49°                              |
| **Polarization**           | Single or Dual Polarization             |
| **Data Size (per scene)**  | 1 GB to 1.5 GB                          |

- **Application**: Urban infrastructure monitoring, environmental studies, detailed mapping of small areas.

---

## 3. Medium Resolution (Stripmap) Mode

In **Medium Resolution Mode**, also known as Stripmap Mode, the radar beam is continuously swept across a wide swath. This mode provides a good balance between resolution and swath width.

| Parameter                 | Value                                    |
|---------------------------|------------------------------------------|
| **Resolution**             | 5 m to 8 m                              |
| **Swath Width**            | 30 km to 100 km                         |
| **Incidence Angle Range**  | 20° to 46°                              |
| **Polarization**           | Single or Dual Polarization             |
| **Data Size (per scene)**  | 500 MB to 800 MB                        |

- **Application**: Environmental monitoring, resource management, forestry.

---

## 4. High Resolution (Stripmap) Mode

**High Resolution Stripmap Mode** improves spatial resolution while maintaining reasonable swath width, ideal for tasks requiring moderate detail.

| Parameter                 | Value                                    |
|---------------------------|------------------------------------------|
| **Resolution**             | 3 m to 5 m                              |
| **Swath Width**            | 20 km to 50 km                          |
| **Incidence Angle Range**  | 25° to 45°                              |
| **Polarization**           | Single or Dual Polarization             |
| **Data Size (per scene)**  | 600 MB to 1 GB                          |

- **Application**: Coastal monitoring, land use and land cover studies.

---

## 5. ScanSAR Mode

The **ScanSAR Mode** sacrifices spatial resolution in favor of covering large areas. It is well-suited for wide-area surveillance and rapid revisit times.

| Parameter                 | Value                                    |
|---------------------------|------------------------------------------|
| **Resolution**             | 20 m to 50 m                            |
| **Swath Width**            | 100 km to 500 km                        |
| **Incidence Angle Range**  | 20° to 45°                              |
| **Polarization**           | Single or Dual Polarization             |
| **Data Size (per scene)**  | 200 MB to 500 MB                        |

- **Application**: Disaster monitoring, ocean and ice surveillance, wide-area mapping.

---

## 6. Extended High (EH) Mode

The **Extended High (EH) Mode** offers extended swath width while maintaining improved spatial resolution. It is a versatile mode for large-area mapping with reasonable detail.

| Parameter                 | Value                                    |
|---------------------------|------------------------------------------|
| **Resolution**             | 8 m to 16 m                             |
| **Swath Width**            | 100 km to 150 km                        |
| **Incidence Angle Range**  | 25° to 47°                              |
| **Polarization**           | Single or Dual Polarization             |
| **Data Size (per scene)**  | 300 MB to 600 MB                        |

- **Application**: Large-area environmental monitoring, urban sprawl detection, agriculture.

---

## 7. Maritime Surveillance Mode (MMS)

The **Maritime Surveillance Mode (MMS)** is specifically tailored for detecting ships, oil spills, and ice in ocean environments. It combines moderate resolution with a wide swath for efficient maritime observation.

| Parameter                 | Value                                    |
|---------------------------|------------------------------------------|
| **Resolution**             | 10 m to 50 m                            |
| **Swath Width**            | 100 km to 300 km                        |
| **Incidence Angle Range**  | 20° to 45°                              |
| **Polarization**           | Single or Dual Polarization             |
| **Data Size (per scene)**  | 200 MB to 500 MB                        |

- **Application**: Ship detection, illegal fishing monitoring, ice mapping, oil spill detection.

---

## 8. Polarization Options

RCM supports various polarization configurations:

- **Single Polarization** (HH or VV): One polarization is transmitted and received.
- **Dual Polarization** (HH+HV or VV+VH): Combines two polarizations for better discrimination of features on the ground.
- **Quad Polarization** (HH+HV+VV+VH): Captures full-polarization data, useful for vegetation analysis and surface characterization.

---

## 9. Data Format and Data Size

The RCM SAR data is typically delivered in **GeoTIFF** format or **RADARSAT proprietary format**. The size of the data files depends on the imaging mode and spatial resolution. A high-resolution scene may range from **500 MB to 1.5 GB**, while wide-area ScanSAR scenes are typically smaller in size.
