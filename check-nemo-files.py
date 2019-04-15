#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from collections import namedtuple
from enum import Enum
import logging
import os.path

import iris
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



logging.basicConfig(level=logging.INFO)


class WikiTableWriter(object):

    cell_style_good = '{background:#1b9e77}'

    cell_style_bad = '{background:#d95f02}'

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.target = open(self.path, 'w')
        self.write_header()
        return self

    def __exit__(self, *args):
        self.target.close()

    def write_header(self):
        self.target.write('|_. Filename/Varname '
                          '|_. is_empty '
                          '|_. all_masked '
                          '|_. unexpected_mask '
                          '|_. nans '
                          '|_. bigs |\n')

    def add_file(self, filename, is_ok):
        if is_ok:
            line = '|{style}.*@{filename}@*|\\5={style}. OK|\n'.format(
                filename=filename,
                style=self.cell_style_good,
            )
        else:
            line = '|{style}.*@{filename}@*|\\5={style}. *NOT OK*|\n'.format(
                filename=filename,
                style=self.cell_style_bad,
            )
        self.target.write(line)

    def add_variable(self, varname, result):
        def format_condition(condition):
            if condition:
                return '={}. *Yes*'.format(self.cell_style_bad)
            else:
                return '={}. No'.format(self.cell_style_good)
        self.target.write('|@{}@|{}|{}|{}|{}|{}|\n'.format(
            varname,
            *map(format_condition, result)
        ))


def read_mask(mesh_mask, kind):
    name = '{}mask'.format(kind)
    mask_cube = iris.load_cube(mesh_mask, name)
    assert(mask_cube.ndim == 4)
    assert(mask_cube.shape[0] == 1)
    mask = mask_cube[0].data.astype('bool')
    return ~mask


def build_umask(tmask):
    umask = np.empty_like(tmask)
    umask[:, :, -1] = tmask[:, :, -1]
    umask[:, :, :-1] = tmask[:, :, :-1] & tmask[:, :, 1:]
    return umask


def build_vmask(tmask):
    vmask = np.empty_like(tmask)
    vmask[:, -1, :] = tmask[:, -1, :]
    vmask[:, :-1, :] = tmask[:, :-1, :] & tmask[:, 1:, :]
    return vmask


def build_wmask(tmask):
    wmask = np.empty_like(tmask)
    wmask[0, :, :] = tmask[0, :, :]
    wmask[1:, :, :] = tmask[1:, :, :] & tmask[:-1, :, :]
    return wmask


def masks_load_tuv_build_w(mesh_mask):
    logging.info('reading prepared uv masks')
    masks = {}
    for kind in ['t', 'u', 'v']:
        masks[kind] = read_mask(mesh_mask, kind)
    masks['w'] = build_wmask(masks['t'])
    return masks


def masks_load_t_build_uvw(mesh_mask):
    logging.info('building my own uv masks')
    tmask = read_mask(mesh_mask, 't')
    masks = {
        't': tmask,
        'u': build_umask(tmask),
        'v': build_vmask(tmask),
        'w': build_wmask(tmask),
    }
    return masks


def prepare_masks(mesh_mask, build_u_v_masks):
    if build_u_v_masks:
        return masks_load_t_build_uvw(mesh_mask)
    else:
        return masks_load_tuv_build_w(mesh_mask)


def isbig(array):
    return (array > 1.e20)


Result = namedtuple('Result',
                    'is_empty '
                    'all_masked unexpected_mask '
                    'has_nans has_bigs')

Status = Enum('Status', 'true false inapplicable')


def check_cube(cube, mask, check_mask=True):
    is_empty = not all(cube.shape)
    if is_empty:
        return Result(Status.true,
                      Status.inapplicable, Status.inapplicable,
                      Status.inapplicable, Status.inapplicable)
    data = cube.data
    unexpected_mask = False
    if np.ma.is_masked(data):
        all_masked = data.mask.all()
        if check_mask:
            unexpected_mask = (data.mask != mask).any()
    else:
        all_masked = False
        if check_mask:
            unexpected_mask = mask is not None
    has_nans = np.isnan(data).any()
    has_bigs = isbig(data).any()
    return Result(is_empty, all_masked, unexpected_mask, has_nans, has_bigs)


def identify_mask(filename):
    components = filename.split('_')
    if components[-1] in ['2D', '3D']:
        dim = int(components[-1][0])
        if components[5] == 'grid':
            if components[6] == 'ptr':
                kind = components[7]
            else:
                kind = components[6]
            if kind in ['T', 'U', 'V', 'W']:
                return (kind.lower(), dim)
    elif (components[-1] == 'sum'
          and (components[-2] == 'vert'
               or (components[-3] == 'zoom'
                   and components[-2] in ['300', '700', '2000']))):
        return ('t', 2)
    return None, None


def check_file(full_filename, masks, table_writer):
    base_name = os.path.basename(full_filename)
    filename, extension = os.path.splitext(base_name)
    assert(extension == '.nc')
    mask_kind, mask_dim = identify_mask(filename)
    if mask_dim is None and mask_kind is None:
        mask = None
    elif mask_dim == 2:
        mask = masks[mask_kind][0]
    elif mask_dim == 3:
        mask = masks[mask_kind]
    else:
        raise RuntimeError('Found unknown dimensionality for mask: {}'
                           ''.format(mask_dim))
    check_mask = ('lim' not in filename) and ('ptr' not in filename)
    cl = iris.load(full_filename)
    warnings = []
    max_name_length = 20
    not_ok_issued = False
    msg_template = ("%{}s: is_empty: %5s, "
                    "all_masked: %5s, unexpected_mask: %5s, "
                    "nans: %5s, bigs: %5s").format(max_name_length+2)
    logging.info("%s: Checking", filename)
    for c in cl:
        result = check_cube(c, mask, check_mask)
        if any(result):
            if not not_ok_issued:
                logging.warning("%s: NOT OK.", filename)
                table_writer.add_file(filename, False)
                not_ok_issued = True
            logging.warning(msg_template, c.var_name, *result)
            table_writer.add_variable(c.var_name, result)
        else:
            logging.info("%s: OK.", c.var_name)
        if (result.unexpected_mask
            and not result.all_masked
            and not result.is_empty):
            fig = plt.figure()
            fig.suptitle("{}:{}".format(filename, c.var_name))
            if mask_dim is None:
                mask_dim = 2
            cube_ind = (0,)*(c.ndim-mask_dim) + (slice(None, None),)*mask_dim
            sc = c[cube_ind]
            if mask is None:
                diff = sc.data.mask.astype(int)
            else:
                mask_ind = (0,)*(mask.ndim-mask_dim) \
                           + (slice(None, None),)*mask_dim
                diff = sc.data.mask.astype(int) - mask[mask_ind].astype(int)
            plt.subplot(2, 2, 1)
            plot_ind = (0,)*(diff.ndim-2) + (slice(None, None),)*2
            plt.imshow(diff[plot_ind],
                       origin='lower', vmin=-1, vmax=1, rasterized=True)
            plt.colorbar()
            plt.subplot(2, 2, 3)
            plt.imshow(sc[plot_ind].data, origin='lower', rasterized=True)
            plt.colorbar()
            plt.subplot(2, 2, 2)
            plt.hist(diff.flatten(), bins=[-1.5, -0.5, 0.5, 1.5], log=True)
            plt.tight_layout(rect=[0, 0.03, 1., 0.95])
            fig.savefig("{}.{}.mask.pdf".format(filename, c.var_name))
    if not not_ok_issued:
        logging.info("%s: OK.", filename)
        table_writer.add_file(filename, True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mesh-mask', default='mesh_mask.nc',
                        help='location of mesh_mask file '
                        '(default: ./mesh_mask.nc)')
    parser.add_argument('-b', '--build-u-v-masks', action='store_true',
                        help='build masks for u and v fields from the t mask '
                        'instead of loading them from the mesh_mask file')
    parser.add_argument('files', nargs='+')
    return parser.parse_args()


def main():
    args = parse_args()
    masks = prepare_masks(args.mesh_mask, args.build_u_v_masks)
    prefix = os.path.basename(os.path.commonprefix(args.files))
    tablefilename = '{}wikitable.txt'.format(prefix)
    with WikiTableWriter(tablefilename) as table_writer:
        for filename in args.files:
            check_file(filename, masks, table_writer)


if __name__ == '__main__':
    main()
