## Plot networks
python src/network.py --config src/configs/balanced_random5.yaml
python src/network.py --config src/configs/balanced_random12.yaml
python src/network.py --config src/configs/balanced_random20.yaml

## Run experiments
python src/main.py --config src/configs/balanced_random5.yaml
python src/main.py --config src/configs/unbalanced_random5.yaml
python src/main.py --config src/configs/balanced_random12.yaml
python src/main.py --config src/configs/unbalanced_random12.yaml
python src/main.py --config src/configs/balanced_random20.yaml
python src/main.py --config src/configs/unbalanced_random20.yaml