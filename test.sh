#!/usr/bin/env bash
nosetests --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package="torch_embed_sim" tests
