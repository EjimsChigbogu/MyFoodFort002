# egroceryapp/utils.py
def load_data_and_train():
    import os
    import pandas as pd
    from django.conf import settings

    rules_csv = os.path.join(
        settings.BASE_DIR,"egroceryapp", "static", "csv", "association_rules.csv"
    )
    rules = pd.read_csv(rules_csv)
    # maybe cache them or load into your model
    print("Rules loaded into memory")
    return rules
