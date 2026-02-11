# Identifying Exoplanetary Transits

## Overview
This project focuses on identifying exoplanets using the transit method.
Exoplanetary transits are detected by analyzing stellar light curves and
observing periodic dips in brightness caused by a planet passing in front
of its host star.

## Run in Google Colab
You can run this project directly in your browser using Google Colab:

🔗 **Open in Colab**  
https://colab.research.google.com/drive/1KiHOwaAvt4cMpb0AvA8P-Js3s1QePVBn#scrollTo=ifPiQcGzbmJg

## Scientific Background
When an exoplanet transits its star, it blocks a small fraction of the
starlight, producing a characteristic dip in the observed flux.
By studying these repeated patterns in time-series data, it is possible
to infer the presence of exoplanets.

## Dataset
- Time-series stellar flux measurements
- Each data point represents recorded flux at a fixed time interval
- Noise and variability are handled through preprocessing techniques

## Methodology
- Light curve preprocessing and normalization
- Noise reduction and smoothing
- Transit detection using statistical and machine learning techniques
- Visualization of detected transit events

## Results
The approach identifies potential exoplanetary transits by detecting
consistent and periodic flux dips in stellar light curves.

## Tools & Technologies
- Python
- NumPy, Pandas
- Matplotlib
- Scikit-learn
- Google Colab

## How to Run Locally
1. Clone the repository
2. Install required dependencies
3. Run the notebook

## Future Work
- Improve noise filtering
- Apply deep learning models (CNN/LSTM)
- Validate detections using confirmed exoplanet catalogs


