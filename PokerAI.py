import argparse
from submodules import rpt, rptg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rptg", "--rank-predictor-train-gen",
        action = "store_true",
        help = "Generate learning data for rank predictor")
    parser.add_argument("--rpt", "--rank-predictor-train",
        action = "store_true",
        help = "Train rank predictor")
    args = parser.parse_args()

    if args.rpt: rpt.main()
    if args.rptg: rptg.main()

if __name__ == "__main__": main()