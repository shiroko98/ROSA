"""Explicit ROSA injection locations, shared by training/checkpoints/inference."""
from __future__ import annotations

import math

SCALE_DEFAULTS = {'scale_mode': 'fixed', 'dynamic_max': .2, 'dynamic_margin': .5, 'dynamic_chunk': 128, 'dynamic_min_match': 6}


def scale_options(metadata):
    options = {name: metadata.get(name, default) for name, default in SCALE_DEFAULTS.items()}
    # Historical dynamic snapshots could omit the cap/minimum and used 0.5/4.
    # New runs serialize explicit settings; do not reinterpret old snapshots
    # when changing the CLI defaults to 0.2/6.
    if options['scale_mode'] == 'logit_gap' and 'dynamic_max' not in metadata:
        options['dynamic_max'] = .5
    if options['scale_mode'] == 'logit_gap' and 'dynamic_min_match' not in metadata:
        options['dynamic_min_match'] = 4
    return options


def scale_options_from_args(args):
    return {name: getattr(args, 'rosa_' + name, default) for name, default in SCALE_DEFAULTS.items()}


def validate_scale_options(position, output_norm, options):
    mode = options['scale_mode']
    if mode not in ('fixed', 'logit_gap'):
        raise ValueError(f'Unknown ROSA scale mode: {mode!r}')
    if mode == 'logit_gap' and (position != 'post_norm' or output_norm not in ('rms', 'rms_match')):
        raise ValueError('logit_gap scale requires post_norm and output_norm=rms/rms_match.')
    if not math.isfinite(options['dynamic_max']) or options['dynamic_max'] <= 0:
        raise ValueError('ROSA dynamic_max must be finite and positive.')
    if not math.isfinite(options['dynamic_margin']) or options['dynamic_margin'] < 0:
        raise ValueError('ROSA dynamic_margin must be finite and nonnegative.')
    if not isinstance(options['dynamic_chunk'], int) or options['dynamic_chunk'] <= 0:
        raise ValueError('ROSA dynamic_chunk must be a positive integer.')
    if not isinstance(options['dynamic_min_match'], int) or options['dynamic_min_match'] < 1:
        raise ValueError('ROSA dynamic_min_match must be a positive integer.')
    return options



def resolve_position(position='block', layers=None, *, n_layer=None):
    if position not in ('block', 'post_norm'):
        raise ValueError(f'Unknown ROSA position: {position!r}')
    if layers is None:
        layers = (0,) if position == 'block' else ()
    elif isinstance(layers, str):
        layers = () if not layers.strip() else tuple(int(i) for i in layers.split(','))
    else:
        layers = tuple(layers)
    if position == 'post_norm':
        if layers:
            raise ValueError('post_norm has no block layers; omit --rosa_layers (or explicitly pass an empty string).')
    elif not layers or min(layers) < 0 or len(set(layers)) != len(layers) or (n_layer is not None and max(layers) >= n_layer):
        raise ValueError('Invalid/duplicate ROSA injection layers.')
    return position, tuple(sorted(layers))


def validate_output_norm(position, output_norm='none'):
    if output_norm not in ('none', 'rms', 'rms_match'):
        raise ValueError(f'Unknown ROSA output normalization: {output_norm!r}')
    if output_norm != 'none' and position != 'post_norm':
        raise ValueError('ROSA output RMS normalization requires post_norm injection.')
    return output_norm


def normalize_metadata(metadata):
    # Historical checkpoints without position were ALWAYS block-before-injection.
    # Never reinterpret them as post_norm just because tensor shapes match.
    position, layers = resolve_position(metadata.get('position', 'block'), metadata.get('layers'))
    output_norm = validate_output_norm(position, metadata.get('output_norm', 'none'))
    options = validate_scale_options(position, output_norm, scale_options(metadata))
    return {**metadata, 'position': position, 'layers': ','.join(map(str, layers)), 'output_norm': output_norm, **options}
