# Umbra SAR Data

## Introduction

Umbra is a cutting-edge commercial space technology company specializing in synthetic aperture radar (SAR) data, providing some of the highest resolution imagery available on the market today. Umbra’s satellites are capable of capturing high-quality imagery regardless of weather conditions or time of day, making them particularly valuable for applications in maritime surveillance, disaster response, defense, and environmental monitoring.

Umbra's SAR technology operates in the X-band frequency, enabling it to deliver images with resolutions as fine as 25 cm. This data is available to the public through Umbra's open data platform, which offers users free access to high-resolution imagery under a permissive license. This democratization of high-quality SAR data makes it accessible to various industries and research sectors that rely on detailed remote sensing.

Their satellite constellation is designed for global coverage, capturing imagery of land, sea, and infrastructure across the world. The imagery is processed into several standardized products, offering flexibility to users based on their specific application needs. The data is formatted in GeoTIFF for easy integration with geospatial software, and comprehensive metadata is provided to describe acquisition parameters, orbit information, and image processing details.

For more information on Umbra’s SAR products and how to access them, please visit their [Open Data platform](https://umbra.space/open-data/).


# Umbra SAR Data

## Technical Overview of Umbra Satellites and Sensors

Umbra operates a growing constellation of synthetic aperture radar (SAR) satellites that provide some of the highest resolution satellite imagery available. These satellites are designed for all-weather, day-and-night imaging, and are used in various applications such as defense, environmental monitoring, disaster response, and maritime surveillance.

### Satellite Constellation

Umbra's satellites are designed for global coverage, offering high revisit times and the ability to capture images under almost any condition. The constellation is expanding, with new satellites regularly added to increase imaging capacity and reduce latency for data delivery.

### Umbra Satellite Technical Characteristics

| **Specification**                | **Details**                                                    |
|-----------------------------------|----------------------------------------------------------------|
| **Satellite Name**                | Umbra SAR                                                      |
| **Number of Satellites**          | 6 (as of 2024), with plans for further expansion               |
| **Orbit Type**                    | Sun-synchronous low Earth orbit (SSO)                          |
| **Orbit Altitude**                | 500 km                                                         |
| **Revisit Time**                  | Multiple passes per day, varying by latitude and region        |
| **Orbit Inclination**             | 97.4°                                                          |
| **Orbital Period**                | ~95 minutes                                                    |
| **Radar Band**                    | X-band (9.6 GHz)                                               |
| **Ground Sample Distance (GSD)**  | As fine as 25 cm, up to 1.2 m depending on acquisition mode    |
| **Polarization**                  | Single polarization (VV)                                       |
| **Imaging Modes**                 | Stripmap, Spotlight                                            |
| **Swath Width**                   | 15 km (Spotlight), up to 50 km (Stripmap)                      |
| **SAR Product Formats**           | GeoTIFF, NITF                                                  |
| **Data Formats**                  | GeoTIFF (standard), includes comprehensive metadata XML files  |
| **Look Direction**                | Right-looking SAR                                              |

### Key Technical Details

1. **Radar Technology**: Umbra’s satellites are equipped with synthetic aperture radar (SAR) technology, operating in the X-band (9.6 GHz). This provides the ability to capture high-resolution images through clouds, rain, and darkness, making the satellites ideal for reliable, all-weather imaging.
  
2. **Resolution**: Umbra’s SAR technology delivers resolutions as fine as 25 cm, which is among the highest commercially available. The imaging modes determine the resolution: finer details are captured in spotlight mode, while larger areas are covered in stripmap mode.

3. **Imaging Modes**:
   - **Spotlight Mode**: Focuses on smaller target areas for high-resolution imaging. Provides the highest detail (GSD of 25 cm) with a swath width of around 15 km.
   - **Stripmap Mode**: Covers larger areas with moderate resolution (GSD of around 1.2 m) and a swath width of up to 50 km.

4. **Orbital Characteristics**:
   - **Sun-Synchronous Orbit (SSO)**: Umbra satellites operate in a sun-synchronous orbit, meaning they pass over the same region at the same local solar time each day. This orbit ensures consistent lighting conditions for optical and SAR imaging.
   - **Altitude**: The satellites operate at an altitude of approximately 500 km, balancing coverage and resolution while allowing for frequent revisits over key areas.
   - **Revisit Frequency**: Due to their orbit and number of satellites, Umbra can provide multiple revisits per day, particularly at higher latitudes. This is especially useful for applications requiring frequent monitoring, such as maritime surveillance and disaster response.

### Sensor Specifications

The SAR sensor onboard Umbra satellites is specifically designed to deliver high-quality imagery for both military and civilian applications. Here are the technical specifications of the SAR payload:

| **Parameter**                     | **Details**                                                    |
|-----------------------------------|----------------------------------------------------------------|
| **Radar Band**                    | X-band (9.6 GHz)                                               |
| **Polarization**                  | Single (VV)                                                    |
| **Range Sampling Frequency**      | 187.5 MHz                                                      |
| **Chirp Bandwidth**               | 75 MHz                                                         |
| **PRF (Pulse Repetition Frequency)** | Up to 4.7 kHz (varies by mode)                               |
| **Antenna Look Direction**        | Right-looking                                                  |
| **Satellite Heading Angle**       | ~192° (varies slightly by orbit position)                      |
| **Incidence Angle**               | 25° to 45° (adjustable based on acquisition mode)              |

### Orbit Design

Umbra satellites use a near-polar, sun-synchronous orbit (SSO) with an inclination of 97.4°, ensuring that the satellites pass over any location on Earth at roughly the same local solar time on each orbit. This design offers excellent global coverage and makes it easier to collect consistent imagery over time.

- **Orbit Altitude**: The 500 km altitude is a carefully chosen balance between resolution and swath width, allowing the satellites to provide high-resolution images while covering broad areas.
- **Sun-Synchronous Design**: The SSO orbit ensures that the satellites can capture images of the same location at consistent lighting conditions (for optical sensors), but it also maximizes revisit opportunities for radar sensors.

### Satellite and Sensor Summary

Umbra’s SAR satellites provide high-resolution imagery with versatile data products suited for a wide range of applications, including maritime monitoring, disaster response, environmental studies, and more. The combination of fine resolution, all-weather capability, and frequent revisit times makes Umbra’s satellite constellation a critical asset for both commercial and governmental organizations worldwide.

For more technical specifications and details on accessing Umbra's SAR data, visit their [product guide](https://help.umbra.space/product-guide).




# Umbra SAR Data: Modes and Technical Details

## SAR Imaging Modes

Umbra satellites utilize synthetic aperture radar (SAR) technology operating in the X-band frequency range (9.6 GHz). The SAR sensor provides multiple imaging modes to capture detailed imagery under varying conditions, balancing coverage and resolution based on the application. Umbra SAR data is used in a wide range of industries such as defense, disaster response, maritime monitoring, and environmental monitoring.

### Imaging Modes Overview

Umbra SAR sensors support two primary imaging modes: **Spotlight Mode** and **Stripmap Mode**. Each mode is optimized for different operational needs, providing a trade-off between resolution, swath width, and data size. Both modes operate in single-polarization (VV) and use right-looking radar configurations.

| **Mode**        | **Resolution**         | **Swath Width**       | **Image Size**          | **Primary Use Case**             |
|-----------------|------------------------|-----------------------|-------------------------|----------------------------------|
| **Spotlight**    | 25 cm (GSD)            | ~15 km                | Large (up to 1.5 GB)    | High-detail target imaging       |
| **Stripmap**     | 1.2 m (GSD)            | Up to 50 km           | Moderate (500 MB–1 GB)  | Wide-area surveillance           |

### 1. **Spotlight Mode**

#### Description:
Spotlight mode is designed for high-resolution imaging over a small area. In this mode, the SAR sensor "dwells" on a specific target, continuously illuminating the area as the satellite moves along its orbital path. This enables the collection of fine details, making it suitable for applications requiring precise analysis, such as infrastructure monitoring, change detection, or target identification.

#### Specifications:

| **Parameter**                     | **Spotlight Mode Details**                                   |
|-----------------------------------|--------------------------------------------------------------|
| **Ground Sample Distance (GSD)**  | 25 cm                                                        |
| **Swath Width**                   | ~15 km                                                       |
| **Resolution**                    | 25 cm pixel size                                             |
| **Incidence Angle**               | 20°–45° (adjustable)                                         |
| **Polarization**                  | Single polarization (VV)                                     |
| **Data Format**                   | GeoTIFF, NITF                                                |
| **Data Size**                     | Up to 1.5 GB per scene                                       |
| **Image Dimensions**              | Variable, typically up to 15,000 x 15,000 pixels             |
| **Radar Frequency**               | 9.6 GHz (X-band)                                             |
| **Azimuth Pixel Spacing**         | 25 cm                                                        |
| **Range Pixel Spacing**           | 25 cm                                                        |
| **PRF (Pulse Repetition Frequency)** | Up to 4.7 kHz                                              |

#### Mathematical Details:
The resolution in Spotlight mode is mathematically tied to the synthetic aperture length and the frequency of the X-band radar. The theoretical resolution is governed by the following formula:

\[ \text{Resolution} = \frac{\lambda}{2 \cdot \sin(\theta)} \]

Where:
- \( \lambda \) is the radar wavelength (~0.03125 m for 9.6 GHz),
- \( \theta \) is the incidence angle (ranging from 20° to 45°).

### 2. **Stripmap Mode**

#### Description:
Stripmap mode is designed for wide-area imaging. In this mode, the satellite collects radar data in continuous strips as it moves along its orbit. This mode is useful for applications such as environmental monitoring, maritime surveillance, and large-scale mapping, where coverage is prioritized over resolution.

#### Specifications:

| **Parameter**                     | **Stripmap Mode Details**                                   |
|-----------------------------------|-------------------------------------------------------------|
| **Ground Sample Distance (GSD)**  | 1.2 m                                                       |
| **Swath Width**                   | Up to 50 km                                                 |
| **Resolution**                    | 1.2 m pixel size                                            |
| **Incidence Angle**               | 25°–45° (adjustable)                                        |
| **Polarization**                  | Single polarization (VV)                                    |
| **Data Format**                   | GeoTIFF, NITF                                               |
| **Data Size**                     | 500 MB to 1 GB                                              |
| **Image Dimensions**              | Typically 50,000 x 10,000 pixels                            |
| **Radar Frequency**               | 9.6 GHz (X-band)                                            |
| **Azimuth Pixel Spacing**         | 1.2 m                                                       |
| **Range Pixel Spacing**           | 1.2 m                                                       |
| **PRF (Pulse Repetition Frequency)** | Up to 3 kHz                                                |

#### Mathematical Details:
In Stripmap mode, the resolution is lower than in Spotlight mode because the satellite is not focusing on a specific target but scanning a broader swath. The azimuth and range resolutions are determined by the following:

\[ \text{Azimuth Resolution} = \frac{\lambda}{2 \cdot \sin(\theta)} \]

Where:
- \( \lambda \) is the radar wavelength (~0.03125 m for 9.6 GHz),
- \( \theta \) is the incidence angle.

For Stripmap, the incidence angle can vary between 25° and 45°, depending on the target area.

## Umbra SAR Data Characteristics

### Incidence Angle and Look Direction

Umbra’s SAR sensor uses an adjustable incidence angle between 20° and 45°, depending on the mode and application. The SAR is right-looking, meaning it captures data to the right of the satellite’s flight path. The incidence angle is crucial in determining the resolution and image geometry.

- **Incidence Angle (Spotlight Mode)**: 20° to 45°
- **Incidence Angle (Stripmap Mode)**: 25° to 45°

### Polarization

Umbra SAR data is acquired in single polarization (VV). While some SAR systems support dual or quad polarization, Umbra's VV polarization is well-suited for general-purpose imaging, particularly for surfaces like water bodies, urban environments, and vegetation.

### Data Format

Umbra provides data in standard formats to ensure compatibility with common geographic information system (GIS) and remote sensing platforms.

- **GeoTIFF**: This is the default format for Umbra data, providing georeferenced imagery that can be easily integrated with mapping software.
- **NITF**: The National Imagery Transmission Format is also available for specialized users, particularly in defense and intelligence sectors. This format includes additional metadata and security features.

### Data Size

The size of Umbra SAR data varies depending on the mode, resolution, and area imaged. The data size for each acquisition is proportional to the pixel resolution and the swath width.

- **Spotlight Mode**: Images can reach up to 1.5 GB due to the high resolution (25 cm) and large image dimensions.
- **Stripmap Mode**: Data sizes range from 500 MB to 1 GB, depending on the swath width and image dimensions.

### Metadata

Each data product is accompanied by an XML metadata file, which contains detailed information about the acquisition parameters, satellite information, processing details, and geometric data (including incidence angles, look direction, orbit parameters, etc.).

The metadata typically includes:
- **Acquisition Parameters**: Time, date, sensor mode, polarization, and resolution.
- **Orbit Data**: Satellite position, velocity, and orbit type (Sun-synchronous, low Earth orbit).
- **Geolocation Data**: Information about the coordinates of the image and ground control points.

## Mathematical Models for Resolution and Pixel Spacing

The resolution and pixel spacing for both Spotlight and Stripmap modes can be computed based on the radar frequency, incidence angle, and the PRF of the system. For instance, the resolution in the azimuth and range directions is given by:

- **Range Resolution**:

\[ R_{res} = \frac{c}{2 \cdot B} \]

Where:
- \( R_{res} \) is the range resolution,
- \( c \) is the speed of light (~3 \times 10^8 \, m/s),
- \( B \) is the chirp bandwidth (75 MHz for Umbra).

- **Azimuth Resolution**:

\[ A_{res} = \frac{\lambda}{2 \cdot \sin(\theta)} \]

Where:
- \( A_{res} \) is the azimuth resolution,
- \( \lambda \) is the radar wavelength (9.6 GHz, or ~0.03125 m),
- \( \theta \) is the incidence angle.

These formulas are used to design the SAR system and ensure that the resulting images meet the required specifications for each mode.

## Conclusion

Umbra SAR data provides a versatile, high-resolution imaging solution for various applications, from wide-area monitoring to detailed target analysis. With flexible modes, high revisit rates, and all-weather capability, Umbra SAR data is an essential resource for industries requiring reliable, timely, and precise satellite imagery.
