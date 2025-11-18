
cd core/

python main.py --run_mode=train      \
              --log_base=../log_icml4/     \
              --data_tr=/home/web603/code/data_dump/yfcc100m/yfcc-sift-2000-train.hdf5 \
              --data_va=/home/web603/code/data_dump/yfcc100m/yfcc-sift-2000-val.hdf5