# MovieLens
CUDA_VISIBLE_DEVICES=0 python main.py -d movielens --warmup 150 -mc

# KuaiRec
CUDA_VISIBLE_DEVICES=0 python main.py -d kuairec -mc -luser 1

# Yelp
CUDA_VISIBLE_DEVICES=0 python main.py -d yelp -mc --warmup 300

# Books
CUDA_VISIBLE_DEVICES=0 python main.py -d books -mc -tau 32