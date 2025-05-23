source /home/dell/anaconda3/etc/profile.d/conda.sh
conda activate STDM

cd model_MDGCRN1

# baseline: MDGCRNAdj
python traintorch_MDGCRNAdjHiD.py --lamb 0.0 --lamb1 0.0 --lamb2 0.0 --schema 0 --gpu 0  # 2 is better than baseline

# MDGCRNAdjHiD + schema 2
python traintorch_MDGCRNAdjHiD.py --lamb 0.0 --lamb1 0.0 --lamb2 1.0 --schema 2 --gpu 1  # 2 is better than 3

# MDGCRNAdjHiD + schema 3
python traintorch_MDGCRNAdjHiD.py --lamb 0.0 --lamb1 0.0 --lamb2 1.0 --schema 3 --gpu 2

# MDGCRNAdjHiD + schema 2 + mask
python traintorch_MDGCRNAdjHiD.py --lamb 0.0 --lamb1 0.0 --lamb2 1.0 --schema 4 --gpu 3

# # MDGCRNAdjHiD + schema 3 + mask
python traintorch_MDGCRNAdjHiD.py --lamb 0.0 --lamb1 0.0 --lamb2 1.0 --schema 4 --gpu 3