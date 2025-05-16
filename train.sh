# MINEDOJO_HEADLESS=1 python -m app.main \
#   --fname experiments/world_model/CollapseTest/100K-CombatSpider-BaseLine.yaml \
#   --devices cuda:0

MINEDOJO_HEADLESS=1 python -m app.main \
  --fname experiments/world_model/CollapseTest/100K-CombatSpider-InverseAndHybrid.yaml \
  --devices cuda:0

# MINEDOJO_HEADLESS=1 python -m app.main \
#   --fname experiments/world_model/CollapseTest/100K-CombatSpider-Inverse.yaml \
#   --devices cuda:0
