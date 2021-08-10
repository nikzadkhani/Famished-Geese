import linecache
import kaggle_environments
import os
import webbrowser
import tracemalloc
import numpy as np


def render(env: kaggle_environments.core.Environment) -> None:
    html_render = env.render(mode="html")
    html_render.replace('"', "&quot;")

    path = os.path.abspath("temp.html")
    with open(path, "w") as f:
        f.write(
            f'<iframe srcdoc="{html_render}" width="800" height="600"></iframe>')

    webbrowser.open("file://" + path)


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def print_weights(net) -> None:
  for i, weight in enumerate(net.parameters()):
    weight = weight.detach().cpu().numpy()
    average_weight = np.mean(weight)
    min_weight = np.min(weight)
    max_weight = np.max(weight)
    print("Layer {0}: shape of {1} and average weight of {2:.3f} min weight of  {3:.3f} and max weight of {4:.3f}"
          .format(i, weight.shape, average_weight, min_weight, max_weight))

  total_params = sum(p.numel() for p in net.parameters())
  print("Total Parameters: {}".format(total_params))
