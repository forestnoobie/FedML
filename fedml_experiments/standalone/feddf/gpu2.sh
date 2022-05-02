PART_NUM=8
for EPOCH in 5 10
do
python ./main_feddf.py --gpu 2 --client_num_in_total 20 --client_num_per_round ${PART_NUM} \
--batch_size 64 --dataset cifar10 --data_dir  /input/dataset/cifar10 \
--model resnet8_no_bn --partition_method hetero --partition_alpha 0.01 --comm_round 100 \
 --epochs ${EPOCH} --lr 0.1 --client_optimizer sgd --frequency_of_the_test 1 --ci 1 \
--valid_ratio 0.1 --unlabeled_data_dir /input/dataset/cifar100 \
 --unlabeled_dataset cifar100 --unlabeled_batch_size 128 --pname feddf

done
