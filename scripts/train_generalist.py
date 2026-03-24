from microsam_llrd.train import build_argparser, finetune_mito_nuc_em_generalist


def main():
    parser = build_argparser()
    args = parser.parse_args()
    finetune_mito_nuc_em_generalist(args)


if __name__ == "__main__":
    main()
