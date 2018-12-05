#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging

import iris
import numpy as np


logging.basicConfig(level=logging.INFO)


MESH_FILE = '/nobackup/rossby18/sm_wyser/barakuda/ORCA1.L75_barakuda/mesh_mask.nc4'


def isbig(array):
    return (array > 1.e20)


def check_cube(cube):
    data = cube.data
    if np.ma.is_masked(data):
        all_masked = data.mask.all()
    else:
        all_masked = False
    has_nans = np.isnan(data).any()
    has_bigs = isbig(data).any()
    return (all_masked, has_nans, has_bigs)


def check_file(filename):
    cl = iris.load(filename)
    warnings = []
    max_name_length = 20
    not_ok_issued = False
    msg_template = "%{}s: all_masked: %s, nans: %s, bigs: %s".format(
        max_name_length+2)
    logging.info("Checking %s.", filename)
    for c in cl:
        result = check_cube(c)
        if any(result):
            if not not_ok_issued:
                logging.warning("%s is NOT OK.", filename)
                not_ok_issued = True
            logging.warning(msg_template, c.var_name, *result)
        else:
            logging.info("%s is OK.", c.var_name)
    if not not_ok_issued:
        logging.info("%s is OK.", filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    return parser.parse_args()


def main():
    args = parse_args()
    for filename in args.files:
        check_file(filename)


if __name__ == '__main__':
    main()
