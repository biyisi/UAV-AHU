# UAV-AHU PycharmProjects

## DataSets

uav-ahu dataset:
链接: https://pan.baidu.com/s/1R8Y8U74y7ze24ar9lMtNHA 提取码: 28ag

This data set uses the DJI Mavic series drones, which were taken on the Qingyuan campus of Anhui University, and 20 buildings were selected. The drone flies around the building at a height of 30-60 M and performs video recording. The collected video is intercepted at the specified frame rate, and then resized to obtain the data set used in this article. 


if not work please send email to xtcbiyisi@163.com

## Instruction manual
1. Create a new data folder to store datasets, and a new model folder to store network models
2. The default data set location is ./data/teacher/view/ and ./data/student/view, and the default model location is ./model/view
3. The model definition file is model.py. The model used in the experiment is set in the text. Please note that the models and model files loaded for training, retraining, distillation training, and testing are consistent. The specific definitions are indicated in the code in the form of comments. The specific function to load the existing model is in utils.py
4. Please ensure that the name of the .mat file loaded in evaluate_gpu_1_ctx.py is consistent with the name of the .mat file saved in test.py
5. Basic training, retraining, and distillation python instructions are provided at the end of the train.py file for reference 

The function code has been modularized and can be searched in the file as needed. If there is something unclear, you can send an e-mail consultation 


