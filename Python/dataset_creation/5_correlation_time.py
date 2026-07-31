# =======================================================================================================================================
# PLEASE READ
# PLEASE READ
# PLEASE READ

# The goal was to see the correlation between the mean delta time between the timestamps of the context and the delta time 
# between the last timestamp of the context and the timestamp of the next cell
#
# Takes some time (30-1h) to compute and the conclusion is that there are no correlation so we cannot use this as time feature
# to predict time. Also need a lot of memory.
#
# PLEASE READ
# PLEASE READ
# PLEASE READ
# =======================================================================================================================================
import json
import utils

with open("time.json", "r") as f:
    processed_train_data_time = json.load(f)

# correlation
# Calcul pour chaque gram

print("Extracting correlation")
correlation_data = utils.extract_correlation_pairs(processed_train_data_time)  # ton dict `time`
stats_by_gram    = {}
processed_train_data_time = None


print("Correlations")
for gram, data in correlation_data.items():
    if len(data["x"]) < 10:
        continue
    stats = utils.compute_correlation_stats(data["x"], data["y"])
    if stats is None:
        print(f"Gram {gram} : données insuffisantes ou constantes, ignoré.")
        continue
    if stats.get("error") == "constant_input":
        print(f"Gram {gram} : input constant (probablement gram=1 sans gaps de contexte), ignoré.")
        continue
    stats_by_gram[gram] = stats
    stats_by_gram[gram]["gram"] = gram

# Sauvegarde pour la viz
with open("correlation_stats.json", "w") as f:
    json.dump(stats_by_gram, f, indent=2)

print(json.dumps({g: {k:v for k,v in s.items() if k not in ("sample_x","sample_y")} 
                  for g,s in stats_by_gram.items()}, indent=2))


import webbrowser, os, pathlib

html = utils.generate_correlation_html(stats_by_gram)
out  = pathlib.Path("correlation_report.html")
out.write_text(html, encoding="utf-8")
webbrowser.open(out.resolve().as_uri())
print(f"Rapport ouvert : {out.resolve()}")