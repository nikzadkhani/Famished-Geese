import kaggle_environments
import os
import webbrowser


def render(env: kaggle_environments.core.Environment) -> None:
  html_render = env.render(mode="html")
  html_render.replace('"', "&quot;")

  path = os.path.abspath("temp.html")
  with open(path, "w") as f:
      f.write(f'<iframe srcdoc="{html_render}" width="800" height="600"></iframe>')

  webbrowser.open("file://" + path)
