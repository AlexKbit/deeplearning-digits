[![Build Status](https://travis-ci.org/AlexKbit/deeplearning-digits.svg?branch=master)](https://travis-ci.org/AlexKbit/deeplearning-digits)

# Project Overview #

Sample of using deep-learning for digits recognition.
This solution based on [deeplearning4j](https://deeplearning4j.org/index.html).

## Application run
  - ApplicationLauncher - spring boot launcher with UI
  - DataSetPrepare - Java runner for prepare training data (/dataset/digits.txt)
  - TrainNeuralNet - Java runner for neural net training (/resources/nnModel)

## User interface

<img alt="UI digit 1" src="https://ndownloader.figshare.com/files/12476879/preview/12476879/preview.jpg">
<img alt="UI digit 3" src="https://ndownloader.figshare.com/files/12476873/preview/12476873/preview.jpg">
<img alt="UI digit 8" src="https://ndownloader.figshare.com/files/12476876/preview/12476876/preview.jpg">

Open application by root path.
  - Draw you digit in window
  - Submit your image
  - Get result of digit on image
