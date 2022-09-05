import urllib.parse


def main(code):
    print(f"Python code:\n\n{code}\n")
    parsed_url = urllib.parse.quote(code)
    url_base = "https://jupyterlite.github.io/demo/repl/index.html"
    jupyterlite_options = "?kernel=python&toolbar=1&code="
    jupyterlite_url = url_base + jupyterlite_options + parsed_url

    print(f"# jupyterlite URL:\n{jupyterlite_url}")

    jupyterlite_iframe_rst = f"""\
<iframe
   src="{jupyterlite_url}"
   width="100%"
   height="500px"
></iframe>\
"""
    print(f"\n# RST for iframe for jupyterlite.rst:\n{jupyterlite_iframe_rst}")


if __name__ == "__main__":
    print("")

    code = """\
import micropip
await micropip.install(["pyhf==0.6.3", "jsonschema<=4.14.0"])
%matplotlib inline
import pyhf\
"""
    raise SystemExit(main(code))
