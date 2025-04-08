# MINEDOJO_HEADLESS=1 python -m app.main \
#   --fname experiments/world_model/100K-CombatSpider-JEPA.yaml \
#   --devices cuda:0

python -m app.main \
  --fname experiments/v-jepa/minedojo/vits16.yaml \
  --devices cuda:0