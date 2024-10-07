"""
Pyinvoke tasks.py file for automating releases and admin stuff.
"""

import glob
import json
import os
import re

import requests
from invoke import task
from monty.os import cd

import m3gnet

NEW_VER = m3gnet.__version__


@task
def make_doc(ctx):
    ctx.run("cp README.md docs_src/index.md")
    ctx.run("cp CHANGES.md docs_src/changes.md")
    with cd("docs_src"):
        ctx.run("rm *tests*.rst", warn=True)
        ctx.run("sphinx-apidoc --separate -d 6 -o . -f ../m3gnet")
        for f in glob.glob("*.rst"):
            if f.startswith("m3gnet") and f.endswith("rst"):
                newoutput = []
                suboutput = []
                subpackage = False
                with open(f) as fid:
                    for line in fid:
                        clean = line.strip()
                        if clean == "Subpackages":
                            subpackage = True
                        if not subpackage and not clean.endswith("tests"):
                            newoutput.append(line)
                        else:
                            if not clean.endswith("tests"):
                                suboutput.append(line)
                            if clean.startswith("m3gnet") and not clean.endswith("tests"):
                                newoutput.extend(suboutput)
                                subpackage = False
                                suboutput = []

                with open(f, "w") as fid:
                    fid.write("".join(newoutput))
    ctx.run("sphinx-build -b html docs_src docs")

    with cd("docs"):
        ctx.run("rm -r .doctrees")
        ctx.run("rm *tests*.html", warn=True)
        ctx.run("rm -r _sources")

        # This makes sure pymatgen.org works to redirect to the Gihub page
        # ctx.run("echo \"pymatgen.org\" > CNAME")
        # Avoid the use of jekyll so that _dir works as intended.
        ctx.run("touch .nojekyll")


@task
def publish(ctx):
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python setup.py sdist bdist_wheel")
    ctx.run("twine upload dist/*")


@task
def release_github(ctx):
    with open("CHANGES.md") as f:
        contents = f.read()
    toks = re.split(r"\#+", contents)
    desc = toks[1].strip()
    payload = {
        "tag_name": "v" + NEW_VER,
        "target_commitish": "main",
        "name": "v" + NEW_VER,
        "body": desc,
        "draft": False,
        "prerelease": False,
    }
    response = requests.post(
        "https://api.github.com/repos/materialsvirtuallab/m3gnet/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]},
    )
    print(response.text)


@task
def release(ctx, notest=False):
    ctx.run("rm -r dist build m3gnet.egg-info", warn=True)
    if not notest:
        ctx.run("pytest m3gnet")
    publish(ctx)
    release_github(ctx)
