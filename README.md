# Skewness & Kurtosis: From Moments to Machine Learning

*The mathematics, intuition, interpretation, and practical use of higher-order moments.*

![Cover](skewness_kurtosis_cover.png)

[![Read on Towards Data Science](https://img.shields.io/badge/Read%20on-Towards%20Data%20Science-1f4e79?style=for-the-badge)](https://towardsdatascience.com/skewness-and-kurtosis-from-moments-to-machine-learning/)

📖 **Published article:** [Skewness and Kurtosis: From Moments to Machine Learning | Towards Data Science](https://towardsdatascience.com/skewness-and-kurtosis-from-moments-to-machine-learning/)

📄 **Full article (markdown, with all figures):** [`Skewness_and_kurtosis.md`](Skewness_and_kurtosis.md)

*Originally published on Towards Data Science.*

---

## About

This repository contains the full source of my article on **skewness and kurtosis**, including the code, figures, and supporting visuals.

The article builds the topic from the ground up. It starts with populations, samples, and degrees of freedom, moves through moments and the formulas for skewness and kurtosis, and ends with data transformations and a practical EDA workflow.

## What You'll Learn

1. Why we need samples: population vs. sample
2. Why we divide by n − 1: degrees of freedom explained
3. Moments: the building blocks of shape
4. Skewness: measuring asymmetry (Fisher–Pearson, bias correction, interpretation)
5. Kurtosis: it's about tails, not peaks (excess kurtosis, leptokurtic, mesokurtic, platykurtic)
6. Calculating skewness and kurtosis in Python
7. Taming skewness with transformations: Log, Square Root, Cube Root, Box-Cox, Yeo-Johnson
8. Standardization vs. transformation
9. Kurtosis in machine learning
10. A practical EDA workflow

## Repository Structure

```text
.
├── README.md
├── Skewness_and_kurtosis_final.md   # Full article in markdown
├── skewness_kurtosis_cover.png      # Cover image
├── supporting_visuals/              # Diagrams and flowcharts
│   ├── 02_moments_diagram.png
│   ├── 03_population_vs_sample.png
│   ├── 04_variance_skewness_kurtosis_flow.png
│   ├── 05_kurtosis_interpretation.png
│   ├── 06_excess_kurtosis_interpretation.png
│   ├── 07_skewness_vs_kurtosis_table.png
│   └── 08_distribution_analysis_flowchart.png
└── *.png                            # Plots generated with Python
```

## Tech Stack

- Python
- NumPy, Pandas
- SciPy (`skew`, `kurtosis`, `boxcox`, `yeojohnson`)
- Matplotlib, Seaborn

## Quick Example

```python
import numpy as np
from scipy.stats import skew, kurtosis

data = np.random.exponential(scale=2, size=1000)

print("Skewness:", skew(data))
print("Excess kurtosis:", kurtosis(data))  # Fisher's definition (normal = 0)
```

## Image & Figure Credits

All figures in this article were created by the author using Python or Lucidchart. The featured image was AI-generated.

## Author

**Mansi Bramta**

| Platform | Link |
|---|---|
| 💼 LinkedIn | [mansi-bramta](https://www.linkedin.com/in/mansi-bramta-65358741b) |
| ✍️ Medium | [@mansibramta12](https://medium.com/@mansibramta12/skewness-kurtosis-from-moments-to-machine-learning-7964a9170043) |
| 📊 Kaggle | [Skewness & Kurtosis](https://www.kaggle.com/code/mansibramta/skewness-kurtosis) |
| 📖 Towards Data Science | [Skewness and Kurtosis article](https://towardsdatascience.com/skewness-and-kurtosis-from-moments-to-machine-learning/) |

---

⭐ If you found this helpful, consider starring the repo and sharing the article.
