from rosa_layer_sweep import build_arg_parser, run_sweep


def run_scan(args):
    return run_sweep(args)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
