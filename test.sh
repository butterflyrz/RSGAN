python Main.py --model GMF --regs [1e-6,1e-6] --loss_func logloss --task tunningBPR --dataset ml-1m --batch_size 256 --embed_size 16  --lr 0.05 --epochs 2
python Main.py --model MLP --regs [1e-6,1e-6] --loss_func logloss --task tunningBPR --dataset ml-1m --batch_size 256 --embed_size 16  --lr 0.05 --epochs 2
python Main.py --model MLP --regs [1e-6,1e-6] --eval global --loss_func logloss --task tunningBPR --dataset ml-1m --batch_size 256 --embed_size 16  --lr 0.05 --epochs 2
python Main.py --model FISM --regs [1e-6,1e-6] --loss_func logloss --task tunningBPR --dataset ml-1m --batch_size 256 --embed_size 16  --lr 0.05 --epochs 2
python Main.py --model FISM --regs [1e-6,1e-6] --eval global --loss_func logloss --task tunningBPR --dataset ml-1m --batch_size 256 --embed_size 16  --lr 0.05 --epochs 2
