dataset="cifar100"
expname="mbrelu"


python3 main.py --dataset ${dataset} --save "split/saved_models/${expname}/finetune/"


for i in {0..9}
do
loaddir="split/saved_models/${expname}/finetune/${dataset}_$i.pth.tar"
splitdir="split/saved_models/${expname}/${dataset}_$i.pth.tar"
bash compute_eigen.sh $i ${dataset} ${loaddir}
python3 index_eigen.py $i 14 ${dataset}
python3 main_split.py --dataset ${dataset} --exp-name ${expname} --split-index $i --rd $i --load ${loaddir}
python3 main_finetune.py --rd $i --dataset ${dataset} --load ${splitdir} --layer -1 --epoch 160 --lr 0.1 --warm 0
done
