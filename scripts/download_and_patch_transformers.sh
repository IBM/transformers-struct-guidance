set -o errexit
cd src
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v3.3.1
# apply patch
git apply ../transformers-v3.3.1-x.patch
