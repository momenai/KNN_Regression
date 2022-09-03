 ## About the problem
 I have implemented  a nearest neighbor algorithm to estimate strength rate values of different concrete materials. I have also extended the implementation as weighted KNN as well. 

• Dataset consists of 1030 samples with continuous concrete strength rate values

## Attribute information for each sample in dataset:
1. Cement
2. Blast Furnace Slag
3. Fly Ash
4. Water
5. Superplasticizer
6. Coarse Aggregate
7. Fine Aggregate
8. Age
9. Concrete compressive strength (”csMPa”, ground-truth strength value of cement material)

## Learning about: 
1. Computing Mean Absolute Error (MAE) 
2. Shuffle all data using numpy permutationsand normalize Features
3. Cross validation for dataset
4. Implemented a nearest neighbor algorithm for 
5. Implemented weighted nearest neighbor algorithm 


## Summary of part 2

Calculating Mean absulote errors: 
Note: Those are simples taken while train but in avarage it is not too much different. 
| K|weighted_non_norm1 | non_weighted_norm1 | weighted_non_norm1 |weighted_norm1 |
| --- | --- | --- | --- | --- |
| 1 | 6.1 | 6.6 |6.2 | 6.6|
| 3 | 6.5 | 6.7 |5.4 | 5.8|
| 5 | 6.8 | 6.9 |5.4 | 5.9|
| 7 | 7.1 | 7.1 |5.5 | 5.9|
| 9 | 7.3 | 7.4 |5.6 | 6.0|



## Observations:
1. Selecting K as 5,7, is the most convientient for this tasks. 
2. weighted and and non-normalized data give in MAE when k = {3,4} 5.4. 
3. Normalization is not important here since distances are too clsoe. We have found that MAE is the best when data left as they are 
4. Without shuffling data,  Mean absolute error increased till 11. 
6. Learning happended, we we shuffle data and diverse the features. 
7. Cross validation increased time of training better to now use 5 K. 3 maybe good. because I does not increaze overall accuracy that much. 
9. **Weighted Non-normalized KNN** is the right implmentation for strengh estimation.
10. weighted_non_normalization give MAE = 5.656 in avarage over all k {1,3,5,7,9}
11. Normalization data does not help much when the distance are too close 

