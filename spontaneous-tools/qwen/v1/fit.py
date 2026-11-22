"""Direct invocation of the frozen fitting loop, with V1's data and output directory."""
import argparse
from common import *
import train

def main():
    p=argparse.ArgumentParser(); p.add_argument('arm',choices=ARMS);p.add_argument('seed',type=int)
    p.add_argument('--pilot',action='store_true');args=p.parse_args()
    train.HERE=STUDY
    train.CFG=json.loads((STUDY/'config.json').read_text())
    train.fit(args.arm,args.seed,args.pilot)
if __name__=='__main__':main()
