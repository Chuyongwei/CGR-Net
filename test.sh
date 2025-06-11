cd core/
python main.py --run_mode=test      \
              --model_path=../log_icml4/train/ \
              --res_path=../log_icml4/yfcc_know \
              --data_te=/home/web603/code/data_dump/yfcc100m/yfcc-sift-2000-testknown.hdf5 \
              --use_ransac=True \
              --log_base=../log_icml4/
python main.py --run_mode=test      \
              --model_path=../log_icml4/train/ \
              --res_path=../log_icml4/yfcc_know \
              --data_te=/home/web603/code/data_dump/yfcc100m/yfcc-sift-2000-testknown.hdf5 \
              --use_ransac=False \
              --log_base=../log_icml4/
python main.py --run_mode=test      \
              --model_path=../log_icml4/train/ \
              --res_path=../log_icml4/yfcc_unknow \
              --data_te=/home/web603/code/data_dump/yfcc100m/yfcc-sift-2000-test.hdf5 \
              --use_ransac=True \
              --log_base=../log_icml4/
python main.py --run_mode=test      \
              --model_path=../log_icml4/train/ \
              --res_path=../log_icml4/yfcc_unknow \
              --data_te=/home/web603/code/data_dump/yfcc100m/yfcc-sift-2000-test.hdf5 \
              --use_ransac=False \
              --log_base=../log_icml4/

