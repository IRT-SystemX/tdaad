SMD (Server Machine Dataset) is a new 5-week-long dataset. We collected it from a large Internet company. This dataset contains 3 groups of entities. Each of them is named by machine-<group_index>-<index>.

SMD is made up by data from 28 different machines, and the 28 subsets should be trained and tested separately. For each of these subsets, we divide it into two parts of equal length for training and testing. We provide labels for whether a point is an anomaly and the dimensions contribute to every anomaly.

Thus SMD is made up by the following parts:

train: The former half part of the dataset.
test: The latter half part of the dataset.
test_label: The label of the test set. It denotes whether a point is an anomaly.
interpretation_label: The lists of dimensions contribute to each anomaly.
concatenate

# Test process
    
All source data are at : /home/shared/shared_results/Demonstrators_results/comp_511_7/SMD_dataset/   
- pip install -r requirements.txt
- Generate the preprocessing data by launching the python script:  create_dataset.py which create the two pickle files test_df.pkl and train_df.pkl    
- Execute the notebook test_SMD.ipynb
    
