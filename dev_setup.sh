cd ..
git clone git@github.com:565353780/camera-manage.git
git clone git@github.com:565353780/colmap-manage.git
git clone git@github.com:565353780/udf-generate.git
git clone git@github.com:565353780/sibr-core.git

cd camera-manage
./dev_setup.sh

cd ../colmap-manage
./dev_setup.sh

cd ../udf-generate
./dev_setup.sh

cd ../sibr-core
./dev_setup.sh

sudo apt install libceres2 imagemagick -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm plyfile tensorboard

cd ../mash-gs/mash_gs/Lib/diff-gaussian-rasterization
pip install -e .

cd ../simple-knn
pip install -e .
