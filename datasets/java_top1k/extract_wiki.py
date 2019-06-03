import argparse
import re
import sys

from tqdm import tqdm
import wikiclean


TEXT_PATTERN = re.compile(r'<text xml:space="preserve">(.+)</text>', re.DOTALL)
REDIRECT_PATTERN = re.compile(r'#REDIRECT.*', re.MULTILINE)


def process(lines, clean=True):
    if len(lines) <= 3:
        return
    text = ''.join(lines)
    m = TEXT_PATTERN.search(text)
    if not m: 
        return
    text = m.group(1)
    if clean:
        try:
            text = wikiclean.clean(text).strip()
            text = REDIRECT_PATTERN.sub('', text)
        except:
            return
    if len(text) > 0: print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lines', type=int) # articles has 983313469
    parser.add_argument('--raw', action='store_true')
    args = parser.parse_args()
    lines = []
    active = False

    pbar = tqdm(sys.stdin, total=args.lines) if args.lines else sys.stdin
    for line in pbar:
        xml_line_flag = '<text xml:space="preserve">' in line
        if xml_line_flag and not active:
            lines.append(line)
            active = True
        elif (xml_line_flag and active) or '</text>' in line:
            active = False
            lines.append(line)
            process(lines, clean=not args.raw)
            lines = []
        elif active:
            lines.append(line)


if __name__ == '__main__':
    main()