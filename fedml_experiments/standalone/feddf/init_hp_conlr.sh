
for CON_LR in 0.005 0.01 0.05 0.1 0.5 1
do
    echo CON_LR
    python ./main_feddf.py --gpu 0 --client_num_in_total 20 --client_num_per_round 8 \
    --batch_size 64 --dataset cifar10 --data_dir /home/osilab7/hdd/cifar \
    --model cnn --partition_method hetero --partition_alpha 0.01 --comm_round 100 \
     --epochs 40 --lr 0.1 --client_optimizer sgd --frequency_of_the_test 1 --ci 1 \
    --valid_ratio 0.1 --unlabeled_data_dir /home/osilab7/hdd/cifar --unlabeled_dataset cifar100 --unlabeled_batch_size 128 \
     --pname con-init-hp --condense --image_per_class 10 --outer_loops 1000 --image_lr 1 --server_steps 0 \
    --train_condense_server --con_rand --condense_server_steps 2000 --condense_patience_steps 100 --condense_train_type soft \
--condense_init  --init_outer_loops 1000 --condense_batch_size 6 --condense_lr ${CON_LR} --condense_optimizer adam
done