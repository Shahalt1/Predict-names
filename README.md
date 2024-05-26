# Predict-names
# Bigram Language Model (BigramLM)

This repository contains a Python implementation of a Bigram Language Model using PyTorch. The BigramLM class reads a text file, constructs a bigram probability matrix, and generates names based on this matrix. Additionally, it provides visualization of the bigram matrix and computes the log-likelihood loss of the text.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Methods](#methods)
- [Example Usage](#example-usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

A Bigram Language Model predicts the next character in a sequence based on the previous character. This project demonstrates a simple implementation of such a model using PyTorch. It includes functionalities to read text data, create a bigram probability matrix, visualize this matrix, generate names, and compute log-likelihood loss.

## Installation

To use this project, clone the repository and ensure you have the necessary dependencies installed.

### Clone the Repository

```bash
git clone https://github.com/yourusername/BigramLM.git
cd BigramLM
```

### Install Dependencies

This project requires `torch` and `matplotlib`. You can install them using pip:

```bash
pip install torch matplotlib
```

## Usage

The main class provided in this repository is `BigramLM`. Here is a detailed guide on how to use it.

### Initializing the BigramLM Class

You can initialize the `BigramLM` class by providing the path to a text file. The class will automatically read the text and create the bigram matrix.

```python
from bigram_lm import BigramLM

# Initialize with the path to your text file
lm = BigramLM('/content/name.txt')
```

### Plotting the Bigram Matrix

After initializing and creating the bigram matrix, you can plot it to visualize the bigram probabilities.

```python
lm.plot()
```

### Generating Names

You can generate names based on the bigram probabilities using the `create_names` method.

```python
names = lm.create_names()
print(names)
```

### Calculating Log-Likelihood Loss

To calculate the log-likelihood loss of the text based on the bigram model, use the `loss` method.

```python
loss = lm.loss()
print(loss)
```

## Methods

### `__init__(self, text_path=None)`

- Initializes the BigramLM class.
- If `text_path` is provided, it automatically uploads the text and creates the bigram matrix.

### `upload_text(self, path)`

- Reads the text file from the provided path.
- Converts the text to lowercase and splits it into words.
- Filters out unwanted characters.

### `create_graph(self)`

- Creates the bigram probability matrix from the uploaded text.

### `plot(self)`

- Plots the bigram probability matrix using `matplotlib`.

### `create_names(self)`

- Generates 20 names based on the bigram probabilities.

### `loss(self)`

- Calculates the log-likelihood loss of the text using the bigram model.

## Example Usage

Here is a complete example of how to use the `BigramLM` class:

```python
import torch
import matplotlib.pyplot as plt
from bigram_lm import BigramLM

# Initialize with the path to your text file
lm = BigramLM('/content/name.txt')

# Plot the bigram matrix
lm.plot()

# Generate names
names = lm.create_names()
print(names)

# Calculate log-likelihood loss
loss = lm.loss()
print(loss)
```

## Contributing

Contributions are welcome! If you have any improvements or suggestions, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

In the `Usage` section, replace `'from bigram_lm import BigramLM'` with the correct import statement based on your project's structure if necessary.

Feel free to adjust the repository URL, licensing information, and any other details to better fit your specific project.
