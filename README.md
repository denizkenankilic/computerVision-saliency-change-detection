# computerVision-saliency-change-detection
Saliency Based Change Detection Algorithm with Performance Calculator

Reference for the algorithm:

Zheng, Y.; Jiao, L.; Liu, H.; Zhang, X.; Hou, B.; Wang, S. Unsupervised saliency-guided SAR image change detection. Pattern Recognition, 2017, 61, 309-326. https://doi.org/10.1016/j.patcog.2016.07.040

The "cd2014_thermal_read_arguments.bat" is used to trigger "changeDetector.py" code to calculate and write the change maps. 5th argument in each line in the batch file need to be edited according to file names in "CD_Codes/Zheng/cd2014_thermal". There are two type of algorithm for the saliency based method. The first one utilize the only saliency algorithm and the second one uses the saliency algorithm with PCA & K-means. The "calculate_performances.bat" is used to trigger "changeDetectionPerformanceTool.py" for calculating all performances with respect to ground truth images. The desired error metrics are specified in the "Inputs_of_Change_Detection_Performance_Tool.xls" files. The ones that are wanted to be calculated should be written as TRUE and the ones that are not wanted to be estimated as FALSE.
