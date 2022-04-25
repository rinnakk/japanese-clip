from setuptools import setup, find_packages
from codecs import open


def _requires_from_file(filename):
    return open(filename).read().splitlines()


exec(open('src/japanese_clip/version.py').read())
setup(
    name="japanese_clip",
    version=__version__,
    author="rinna Co., Ltd.",
    description="Japanese CLIP",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    url="",
    long_description_content_type="text/markdown",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=_requires_from_file('requirements.txt'),
    extras_require={'dev': ['pytest', 'python-dotenv']},
)
