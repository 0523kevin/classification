dataname=$1
n_class=18
opt_list='Adam'
lr_list='1e-3 5e-4 1e-4'
aug_list='gray_crop' # random_all random_all1' #gaussian 
bs_list='32 64'
model_list='efficientnet_b0'
p_list='1 0.5'


for model in $model_list
do
    for bs in $bs_list
    do
        for opt in $opt_list
        do
            for lr in $lr_list
            do
                for aug in $aug_list
                do
                    for p in $p_list
                    do
                        # use scheduler
                        echo "model:$model, bs: $bs, opt: $opt, lr: $lr, aug: $aug, p: $p use_sched: True"
                        exp_name="${model}_CE_${lr}_${bs}_${aug}_scheduler_p=$p"
                        
                        if [ -d "$exp_name" ]
                        then
                            echo "$exp_name is exist"
                        else
                            python train.py \
                                --model $model \
                                --exp_name $exp_name \
                                --n_class $n_class \
                                --optimizer $opt \
                                --aug $aug \
                                --p $p \
                                --batch_size $bs \
                                --lr $lr \
                                --scheduler \
                                --epochs 50
                        fi
                    done
                    # # not use scheduler
                    # echo "loss: $loss, model:$model, bs: $bs, opt: $opt, lr: $lr, aug: $aug, use_sched: False"
                    # exp_name="$model""_$loss""_$lr""_$bs""_$aug"

                    # if [ -d "$exp_name" ]
                    # then
                    #     echo "$exp_name is exist"
                    # else
                    #     python train.py \
                    #         --model $model \
                    #         --exp_name $exp_name \
                    #         --n_class $n_class \
                    #         --optimizer $opt \
                    #         --aug $aug \
                    #         --batch_size $bs \
                    #         --lr $lr \
                    #         --epochs 50
                    # fi
                done
            done
        done
    done
done
