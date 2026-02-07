#DATANAMES=("stock" "energy" "fmri" "ETTh1" "ETTh2" "ETTm1" "ETTm2"  )
DATANAMES=("stock")

export TORCHINDUCTOR_CACHE_DIR=/work/vb21/haochen/torch_cache/inductor
export TORCH_COMPILE_CACHE_DIR=/work/vb21/haochen/torch_cache/compile
mkdir -p $TORCHINDUCTOR_CACHE_DIR
mkdir -p $TORCH_COMPILE_CACHE_DIR

for DATA in "${DATANAMES[@]}"
do
  CUDA_VISIBLE_DEVICES=0 python run_unconditional.py \
  --config "./configs/unconditional/${DATA}.yaml"
done



