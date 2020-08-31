# You Only Look Once v4 (YOLOv4)

The training pipeline is detailed in `train.ipynb`. To create the dataset, run `labeltoYOLO.ipynb` after modifying the file paths to `train.csv` downloaded from [Global Wheat](https://www.kaggle.com/c/global-wheat-detection/data). 

There are two possible inference files. Firstly, `WBFinference.ipynb` uses Weighted Boxes Fusion (WBF) to ensemble over test time augmentation (TTA). This is a baseline inference.

`Pseudoinference.ipynb` uses self-train pseudolabelling to finetune the model on the test data. During pseudolabelling and inference, TTAx4 is used, as well as out of fold inference. It is intended to be a high-performance inference notebook for a single model. This notebook scored #39, see [Notebook](https://www.kaggle.com/stanleyjzheng/apache2-yolov4-pseudolabelling-oof?scriptVersionId=40172709).

Using the baseline inference combined with the model created with `train.cfg`, a public score of 0.7115 and a private score of 0.6371 is attained. With bayesian optimization and self-training pseudolabels, a private score of 0.6498 and a public score of 0.7432 is attained. 
