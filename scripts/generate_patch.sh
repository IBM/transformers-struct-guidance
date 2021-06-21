set -o errexit
cd src/transformers/
git format-patch v3.3.1 --stdout > ../transformers-v3.3.1-x.patch
