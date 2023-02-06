#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from configuration import add_subparser as add_subparser_conf
from run import add_subparser as add_subparser_run
from evaluate import add_subparser as add_subparser_evaluate

parser = argparse.ArgumentParser(
    prog="python -m simson_MARL", description="Drag reduction in 3D channel"
)

subparsers = parser.add_subparsers()
add_subparser_run(subparsers)
add_subparser_evaluate(subparsers)
add_subparser_conf(subparsers)
args = parser.parse_args()
args.cmd(**vars(args))
