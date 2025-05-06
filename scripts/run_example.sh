
python3 main.py \
    --config  "configs/boss.yaml" \
    --data data/example_mixed/Xy_train.csv \
    --output "output/example_output/boss/output.txt" \
    --knowledge data/example_mixed/knowledge.txt \
    --metadata data/example_mixed/metadata.json

    


python3 main.py \
    --config  "configs/direct_lingam.yaml" \
    --data data/example_continuous/Xy_train.csv \
    --output "output/example_output/direct_lingam/output.txt" \
    --knowledge data/example_continuous/knowledge.txt \
    --metadata data/example_continuous/metadata.json

    
python3 main.py \
    --config  "configs/fges.yaml" \
    --data data/example_mixed/Xy_train.csv \
    --output "output/example_output/fges/output.txt" \
    --knowledge data/example_mixed/knowledge.txt \
    --metadata data/example_mixed/metadata.json


python3 main.py \
    --config  "configs/grasp.yaml" \
    --data data/example_mixed/Xy_train.csv \
    --output "output/example_output/grasp/output.txt" \
    --knowledge data/example_mixed/knowledge.txt \
    --metadata data/example_mixed/metadata.json


python3 main.py \
    --config  "configs/pc.yaml" \
    --data data/example_mixed/Xy_train.csv \
    --output "output/example_output/pc/output.txt" \
    --knowledge data/example_mixed/knowledge.txt \
    --metadata data/example_mixed/metadata.json

python3 main.py \
    --config  "configs/dagma.yaml" \
    --data data/example_continuous/Xy_train.csv \
    --output "output/example_output/dagma/output.txt" \
    --knowledge data/example_continuous/knowledge.txt \
    --metadata data/example_continuous/metadata.json
