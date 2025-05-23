source ~/anaconda3/etc/profile.d/conda.sh
conda activate traffic_flow

cd model_DGCRN1

# temperature
for t in 10.0 5.0 3.0 2.0 1.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python traintorch_MDGCRNAdj.py \
    --contra_loss=infonce \
    --temp=$t 
done

# # InfoNCE Loss
# for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 2.0 3.0; do
# for lambda in 0.01 0.05 0.1 0.5 1.0 2.0; do
# for lambda1 in 0.01 0.05 0.1 0.5 1.0 2.0; do
#     python traintorch_MDGCRNAdj.py \
#         --contra_loss=infonce \
#         --temp=$t \
#         --lamb=$lambda \
#         --lamb1=$lambda1 

# done
# done
# done

# # Triplet Loss
# for lambda in 0.01 0.05 0.1 0.5 1.0 2.0; do
# for lambda1 in 0.01 0.05 0.1 0.5 1.0 2.0; do
#     python traintorch_MDGCRNAdj.py \
#         --contra_loss=triplet \
#         --lamb=$lambda \
#         --lamb1=$lambda1 

# done
# done
  

