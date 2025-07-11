# Wine Quality Analysis

This repository contains an analysis of the physicochemical properties and sensory quality scores of Portuguese "Vinho Verde" wine, combining both red and white variants. The goal is to predict wine quality based on its physicochemical properties using machine learning models.<br>

## Dataset Description

The dataset includes 12 input variables:
- Fixed acidity (g/dm³)
- Volatile acidity (g/dm³)
- Citric acid (g/dm³)
- Residual sugar (g/dm³)
- Chlorides (g/dm³)
- Free sulfur dioxide (mg/dm³)
- Total sulfur dioxide (mg/dm³)
- Density (g/cm³)
- pH
- Sulphates (g/dm³)
- Alcohol (% by volume)
- Type (red or white)

The output variable is **quality**, scored between 0 and 10 based on sensory data. (both inclusive)

## Contents

- `Wine_Quality.ipynb`: Jupyter Notebook containing the analysis and machine learning models.
- `data/wine-quality.csv`: The dataset used for the analysis.
- `models/model.json`: Saved model architecture in JSON format.
- `models/model.joblib`: Saved trained model.
- `images/model_arch.JPG`: The archtecture of the deep learning model.

## Contributing

Feel free to contribute to this repository by opening issues or pull requests.

## License

This project is licensed under the [Apache License - Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
