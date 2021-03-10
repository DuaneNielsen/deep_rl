### Building the documentation

make sure sphinx is installed

```
pip install sphinx
```

build the docs

```
cd docsrc
sphinx-apidoc -o . ..
make html
cp -a _build/html/. ../docs
```

