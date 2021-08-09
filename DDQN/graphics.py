import os
import webbrowser
from kaggle_environments import make
import kaggle_environments

def render_env(env: kaggle_environments.core.Environment) -> None:
 render = env.render(mode="html")
 render = render.replace('"', "&quot;")

 path = os.path.abspath("temp.html")
 with open(path, "w") as f:
    f.write(f'<iframe srcdoc="{render}" width="800" height="600"></iframe>')

 webbrowser.open("file://" + path)