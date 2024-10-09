conda env export | grep -v "^prefix: " > environment.yml
pip3 freeze > requirements.txt