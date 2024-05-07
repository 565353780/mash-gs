cd ..
git clone https://github.com/565353780/camera-manage.git
git clone https://github.com/565353780/colmap-manage.git
git clone https://github.com/565353780/udf-generate.git
git clone https://github.com/565353780/sibr-core.git

cd camera-manage
./setup.sh

cd ../colmap-manage
./setup.sh

cd ../udf-generate
./setup.sh

cd ../sibr-core
./setup.sh

sudo apt install libceres2 imagemagick -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm plyfile tensorboard

cd ../mash-gs/mash_gs/Lib/diff-gaussian-rasterization
pip install -e .

cd ../simple-knn
pip install -e .
