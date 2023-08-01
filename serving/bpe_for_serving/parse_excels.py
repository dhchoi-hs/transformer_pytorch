#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=E0401, W0511


"""A module that parses the AI Hub Ko-En parallel corpus excel files.

It only works for AI Hub Ko-En parallel corpus excel files. If you need to
parse some other raw corpus files, you have to implement it from scratch.

Usage example:

    parse_excels.py -excel_dir EXCEL_DIR
                    -corpus_dir CORPUS_DIR
                    [-ko_corpus_name KO_CORPUS_NAME]
                    [-en_corpus_name EN_CORPUS_NAME]
"""

import argparse
import os
import re
import sys
import pandas as pd
from hs_aiteam_pkgs.util.logger import error, info
from hs_aiteam_pkgs.util.time_logger import TimeLogger


CHAR_FILTER = {
    # part 1: invisible/unprintable characters(mostly)
    '\x7f': '',     # (DEL -> '')
    '\x87': '',     # ('‡' -> '')
    '\xa0': ' ',    # (NBSP -> space)
    '\xad': '',     # (soft hyphen -> '')
    '\u115f': '',   # (hangul choseong filler -> '')
    '\u200b': '',   # (zero width space -> '')
    '\u200e': '',   # (left-to-right mark -> '')
    '\u202a': '',   # (left-to-right embedding -> '')
    '\u202c': '',   # (pop directional formatting -> '')
    '\u202d': '',   # (left-to-right override -> '')
    '\u2116': '',   # (№(numero) -> '')
    '\u3000': ' ',  # (ideographic space -> ' ')
    '\u302e': '',   # (HANGUL SINGLE DOT TONE MARK -> '')
    '\u3164': '',   # (hangul filler -> '')
    '\ufe0f': '',   # ('࿾' -> '')
    '\ufeff': '',   # (ZERO WIDTH NO-BREAK SPACE/BYTE ORDER MARK -> '')
    # part 2: non-ascii/non-alphanumeric characters(visible mostly)
    '˙': ' ',
    '˜': '~',
    '˝': '"',
    '̊': '˚',
    '·': '·',
    '‐': '-',
    '‑': '-',
    '–': '-',
    '—': '-',
    '―': '-',
    '‘': '\'',
    '’': '\'',
    '“': '"',
    '”': '"',
    '•': '·',
    '․': '·',
    '…': '..',
    '‧': '·',
    '‰': '%',
    '′': '\'',
    '″': '"',
    '‵': '\'',
    '⁄': '/',
    '₁': '1',
    '₂': '2',
    '₃': '3',
    '→': '->',
    '∕': '/',
    '∙': '·',
    '∪': 'U',
    '∼': '~',
    '≪': '<',
    '≫': '>',
    'ⓛ': '(l)',
    '┃': '|',
    '□': 'O',
    '○': 'o',
    '★': '*',
    '☆': '*',
    '♥': '♡',
    '➡': '->',
    '⟪': '<',
    '⟫': '>',
    '〃': '"',
    '〈': '<',
    '〉': '>',
    '《': '<',
    '》': '>',
    '「': '<',
    '」': '>',
    '『': '<',
    '』': '>',
    '【': '<',
    '】': '>',
    '〔': '[',
    '〕': ']',
    '〜': '~',
    '゜': '°',
    '・': '·',
    '㈔': '(사)',
    '㈜': '(주)',
    '㍱': 'hPa',
    '㎃': 'mA',
    '㎈': 'cal',
    '㎉': 'kcal',
    '㎎': 'mg',
    '㎏': 'kg',
    '㎐': 'Hz',
    '㎑': 'kHz',
    '㎒': 'MHZ',
    '㎓': 'GHz',
    '㎔': 'THz',
    '㎖': 'ml',
    '㎗': 'dl',
    '㎘': 'kl',
    '㎚': 'nm',
    '㎜': 'mm',
    '㎝': 'cm',
    '㎞': 'km',
    '㎟': 'sq mm',
    '㎠': 'sq cm',
    '㎡': 'sq m',
    '㎢': 'sq km',
    '㎥': 'cube m',
    '㎧': 'm/s',
    '㎨': 'm/sq s',
    '㎩': 'Pa',
    '㎫': 'MPa',
    '㎳': 'ms',
    '㎷': 'mV',
    '㎸': 'kV',
    '㎽': 'mW',
    '㎾': 'kW',
    '㎿': 'MW',
    '㏃': 'Bq',
    '㏄': 'cc',
    '㏈': 'dB',
    '㏊': 'hs',
    '㏏': 'kt',
    '％': '%',
    '＆': '&',
    '＋': '+',
    '，': ',',
    '－': '-',
    '．': '.',
    '３': '3',
    '：': ':',
    '＜': '<',
    '＞': '>',
    '？': '?',
    '～': '~',
    '｢': '<',
    '｣': '>',
    '･': '·',
    '￡': '£',
    '￦': '₩',
    # part 3: highly unlikely/useless by the corpus
    'ˈ': '',
    '¨': '',
    '«': '<',
    '»': '>',
    '¬': '-',
    'ː': ':',
    '¹': '1',
    '²': '2',
    '³': '3',
    '´': '\'',
    '¼': '1/4',
    '¾': '3/4',
    '÷': '/',
    'ʻ': '',
    'ㆍ': '·',
    '➋': '(2)',
    '➌': '(3)',
    '➍': '(4)',
    '➎': '(5)',
    '➏': '(6)',
    '①': '(1)',
    '②': '(2)',
    '➂': '(3)',
    '③': '(3)',
    '④': '(4)',
    '⑤': '(5)',
    '⑥': '(6)',
    '⑦': '(7)',
    '⑧': '(8)',
    '⑯': '(16)',
    '⃗': '',
    '⅓': '1/3',
    '⅔': '2/3',
    '℃': '°C',
    '℉': '°F',
    '㎍': 'µg',
    '㎛': 'µm',
    'ℓ': 'l'
}


def preprocess_text(text):
    """Conducts some preprocessing steps on the input text.

    Preprocessing steps are composed of simple cleansing chores like
    lowercasing or redundant space reduction and somewhat laborious
    tasks like character-variants normalization.
    * text preprocessing steps in order:
    1. strip
    2. CHAR_FILTER sub
    3. lowercasing
    4. space collapsing

    Args:
        text: The input text body to be preprocessed.

    Returns:
        A list of space-split word tokens that have been preprocessed.
    """
    rep = dict((re.escape(k), v) for k, v in CHAR_FILTER.items())
    pat1 = re.compile('|'.join(rep.keys()))
    pat2 = re.compile('[ ]{2,}')
    text = pat1.sub(lambda m: rep[re.escape(m.group(0))], text.strip())
    text = pat2.sub(' ', text.lower())
    return text.split()


def main(excel_dir, corpus_dir, ko_corpus_name, en_corpus_name):
    """Covers the whole process to parse AI Hub Ko-En raw corpus excel files.

    Needs some refactoring using the 'preprocess_text' function for later.

    Args:
        excel_dir: Input path that holds AI Hub Ko-En bilingual excel files.
        corpus_dir: Output path to save parsed Ko-En monolingual corpus files.
        ko_corpus_name: Korean corpus file name.
        en_corpus_name: English corpus file name.
    """
    # TODO refactoring using preprocess_text later
    rep = dict((re.escape(k), v) for k, v in CHAR_FILTER.items())
    pattern = re.compile('|'.join(rep.keys()))
    pattern2 = re.compile('[ ]{2,}')  # pattern for consecutive spaces
    exl2df = {}
    for file_name in sorted(os.listdir(excel_dir)):
        if not file_name.endswith('.xlsx'):
            continue
        exl2df[file_name] = pd.read_excel(excel_dir + '/' + file_name,
                                          usecols=['원문', '번역문'])
        info('%050s%s', file_name, exl2df[file_name].shape)
    ko_lines = []
    en_lines = []
    for file_name in sorted(exl2df.keys()):
        tmpdf = exl2df[file_name]
        ko_lines.extend(
            [pattern.sub(lambda m: rep[re.escape(m.group(0))], line.strip())
             for line in list(tmpdf[tmpdf.columns[0]])])
        en_lines.extend(
            [pattern.sub(lambda m: rep[re.escape(m.group(0))], line.strip())
             for line in list(tmpdf[tmpdf.columns[1]])])
    if len(ko_lines) != len(en_lines):
        error('(main) counts of lines in ko/en corpus are different!')
        sys.exit(1)
    ko_corpus_name = corpus_dir + '/' + ko_corpus_name
    with open(ko_corpus_name, 'wt', encoding='utf-8') as ko_outf:
        ko_outf.write(pattern2.sub(' ', ('\n'.join(ko_lines)).lower()))
    en_corpus_name = corpus_dir + '/' + en_corpus_name
    with open(en_corpus_name, 'wt', encoding='utf-8') as en_outf:
        en_outf.write(pattern2.sub(' ', ('\n'.join(en_lines)).lower()))
    del exl2df, ko_lines, en_lines


if __name__ == '__main__':
    info('[%s] module invoked', __file__)

    AP = argparse.ArgumentParser(description='args parser')
    AP.add_argument('-excel_dir', action='store', required=True,
                    help='input path to AI Hub Ko/En bilingual excel files')
    AP.add_argument('-corpus_dir', action='store', required=True,
                    help='output path 2 parsed Ko/En monolingual corpus files')
    AP.add_argument('-ko_corpus_name', action='store', default='ko.txt',
                    help='Korean corpus file name')
    AP.add_argument('-en_corpus_name', action='store', default='en.txt',
                    help='English corpus file name')
    ARGS = AP.parse_args()

    _timer = TimeLogger()
    main(ARGS.excel_dir, ARGS.corpus_dir, ARGS.ko_corpus_name,
         ARGS.en_corpus_name)
    info('main() takes time: %s', _timer.get_elapsed_time_str())
