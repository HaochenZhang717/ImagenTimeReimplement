#DATANAMES=("stock" "energy" "fmri" "ETTh1" "ETTh2" "ETTm1" "ETTm2"  )
DATANAMES=("stock")


for DATA in "${DATANAMES[@]}"
do
  CUDA_VISIBLE_DEVICES=0 python run_unconditional.py \
  --config "./configs/unconditional/${DATA}.yaml"
done



