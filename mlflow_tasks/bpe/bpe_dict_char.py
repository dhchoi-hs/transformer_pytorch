#! /usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=R0912, R0915

"""A module that implements BPE char-level tokenization.

The implementation is aligned with the original paper(Sennrich et al., 2015).
And the module accepts the input path for preprocessed corpus file(s), the
temp output path for split corpus file(s) according to the input split ratio,
and the target output path to hold the BPE char-level dictionary file.

Usage example:

  bpe_dict_char.py -cleaned_corpus_dir CLEANED_CORPUS_DIR
                   -split_corpus_dir SPLIT_CORPUS_DIR
                   -split_ratio SPLIT_RATIO
                   -tmp_dir TMP_DIR
                   -bpe_dict_dir BPE_DICT_DIR
"""

import argparse
import collections
import json
import re
import sys
from os import mkdir, listdir
from os.path import basename, isdir, isfile, join
from hs_aiteam_pkgs.util.logger import debug, info, warning, error
# from hs_aiteam_pkgs.util.signal_handler import SigTermException
from hs_aiteam_pkgs.util.time_logger import TimeLogger


# BPE dictionary
BPE_CHAR_DICT_NAME = 'BPE_char_dict.json'
# BPE temp file names
BI_CNT_FILE_NAME = 'BPE_char_bigram2cnt.txt'
BPE_COD_FILE_NAME = 'BPE_char_codes.txt'
BPE_SYM_FILE_NAME = 'BPE_char_symbols.txt'
BPE_VOC_FILE_NAME = 'BPE_char_vocab.json'
# BPE config params
BPE_MAX_ITERATION = 50000
LEADING_WORD_SYMBOL = '@@'
SYM_MIN_FREQ = 5


def count_bigrams(vocab):
    """Counts the bigram frequency.

    Receives a vocab dict of (token: freq) pairs and converts it into a dict
    of ((head, tail): freq) bigram-count pairs.

    Args:
        vocab: A dict of (token: freq) pairs. (ex) {"@@f our": 573, ..}

    Returns:
        A dict of (bigram: freq) pairs. (ex) {("@@o", "pp"): 2055, ..}
    """
    bigram2cnt = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            bigram2cnt[symbols[i],symbols[i+1]] += freq
    return bigram2cnt


def merge_vocab(best_bigram, v_in):
    """Compacts the given best bigram in the whole vocab.

    Updates the whole vocab concatenating all occurrences of the most frequent
    bigram pair.

    Args:
        best_bigram: The most frequent bigram pair in the vocab at the moment.
        v_in: The vocab in progress of best bigram pair compaction so far.

    Returns:
        The result vocab of best bigram compaction.
    """
    v_out = {}
    bigram = re.escape(' '.join(best_bigram))
    pat = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    compact_bigram = ''.join(best_bigram)
    for word in v_in:
        w_out = pat.sub(compact_bigram, word)
        v_out[w_out] = v_in[word]
    return v_out


def build_bpe_char_dict(symbols_path, codes_path):
    """Builds the BPE char-level dict using given resources.

    Combines the given unigram and bigram tokens to build a BPE char-level
    dict. Assumes both input resource paths to be valid.

    Args:
        symbols_path: The path for the unigram token list.
        codes_path: The path for the bigram BPE code token list.

    Returns:
        A dict mapping unigram & bigram BPE code tokens to their corresponding
        index.
    """
    bpe_dict = {'__UNK__': 0, '__PAD__': 1, '__BOS__': 2, '__EOS__': 3,
                '__MASK__': 4, LEADING_WORD_SYMBOL: 5}
    offset = len(bpe_dict)
    with open(symbols_path, 'rt', encoding='utf-8') as inf:
        symbols = sorted([line.strip() for line in inf.readlines()])
        for i, symbol in enumerate(symbols):
            bpe_dict[symbol] = offset + i
        offset += len(symbols)
    with open(codes_path, 'rt', encoding='utf-8') as inf:
        for i, line in enumerate(inf.readlines()):
            tk1, tk2, _ = line.split()
            bpe_dict[tk1 + tk2] = offset + i
    return bpe_dict


def main(
    cleaned_corpus_dir, split_corpus_dir, split_ratio, tmp_dir, bpe_dict_dir):
    """Covers the whole process to build a BPE char-level dict.

    Reads the preprocessed input corpus files, splits them by the given ratio,
    saves temporary files in a given path, and generates the final BPE dict
    and saves it in a given path.

    Args:
        cleaned_corpus_dir: Path to hold preprocessed input corpus files.
        split_corpus_dir: Path for split output corpus files.
        split_ratio: train:valid:test ratio. (ex) 0.8:0.1:0.1 or 8:1:1
        tmp_dir: Path to save intermediate files.
        bpe_dict_dir: Path for the output BPE dictionary file.
    """
    from corpora.split_corpus import split_corpus_file

    if not isdir(cleaned_corpus_dir):
        error('valid cleaned input corpora path should be given')
        sys.exit(1)
    if not isdir(split_corpus_dir):
        mkdir(split_corpus_dir)
    if not isdir(tmp_dir):
        mkdir(tmp_dir)
    if not isdir(bpe_dict_dir):
        mkdir(bpe_dict_dir)

    if isfile(join(tmp_dir, BPE_VOC_FILE_NAME)):
        with open(
            join(tmp_dir, BPE_VOC_FILE_NAME), 'rt', encoding='utf-8') as inf:
            vocab = collections.defaultdict(int, json.load(inf))
        info('(main progress) BPE vocab loaded')
    else:
        mono_corpora = []
        for fname in listdir(cleaned_corpus_dir):
            fname = join(cleaned_corpus_dir, fname)
            if not (isfile(fname) and fname.endswith('.txt')):
                continue
            split_corpora =\
                split_corpus_file(fname, split_corpus_dir, split_ratio)
            # read trainset only
            with open(split_corpora[0], 'rt', encoding='utf-8') as inf:
                space_split_lines =\
                    [line.strip().split() for line in inf.readlines()]
            mono_corpora.append(space_split_lines)
            info('(main progress) splitting %s done(# of lines: %d)',
                basename(fname), len(space_split_lines))
        vocab = collections.defaultdict(int) # {'@@ l o w':5,'@@ f o r':3,...}
        for mono_corpus in mono_corpora:
            for sentence_words in mono_corpus:
                for word in sentence_words:
                    vocab[LEADING_WORD_SYMBOL+' '+' '.join(list(word))] += 1
        info('(main progress) BPE vocab initialized')

    if not isfile(join(tmp_dir, BPE_SYM_FILE_NAME)):
        sym2cnt = collections.defaultdict(int)
        for word in vocab:
            symbols = list(word[2:].replace(' ', ''))
            for symbol in symbols:
                sym2cnt[symbol] += vocab[word]
        sym2cnt = {k: v for k, v in sym2cnt.items() if v >= SYM_MIN_FREQ}
        with open(
            join(tmp_dir, BPE_SYM_FILE_NAME), 'wt', encoding='utf-8') as outf:
            outf.write('\n'.join(sorted(sym2cnt.keys())))
        info('(main progress) BPE symbols saved')

    codes = {}
    if isfile(join(tmp_dir, BPE_COD_FILE_NAME)):
        with open(
            join(tmp_dir, BPE_COD_FILE_NAME), 'rt', encoding='utf-8') as inf:
            for line in inf.readlines():
                tk1, tk2, tk3 = line.strip().split()
                codes[tk1, tk2] = int(tk3)
        info('(main progress) BPE codes loaded')

    try:
        iter_idx = len(codes)
        bigram2cnt = count_bigrams(vocab)  # {('@@','l'):9, ('l','o'):7, ...}
        _timer = TimeLogger()
        while True:
            _timer.reset_start_time()
            if BPE_MAX_ITERATION <= iter_idx:
                info('BPE merge loop completed')
                break
            iter_idx += 1
            bigram_cnt = len(bigram2cnt)
            best_bigram = max(bigram2cnt, key=bigram2cnt.get)
            vocab = merge_vocab(best_bigram, vocab)
            codes[best_bigram] = iter_idx
            best_bigram_cnt = bigram2cnt[best_bigram]
            bigram2cnt = count_bigrams(vocab)
            debug('[BPE merge %d] best bigram: %s/%d, bigram cnt: %d, ' +\
                'elapsed: %ss', iter_idx, best_bigram, best_bigram_cnt,
                bigram_cnt, _timer.get_elapsed_time_seconds())
    except KeyboardInterrupt:
        info('BPE merge loop terminated by KeyboardInterrupt')
    # except SigTermException:
    #     info('BPE merge loop terminated by SigTermException')
    finally:
        with open(
            join(tmp_dir, BPE_VOC_FILE_NAME), 'wt', encoding='utf-8') as outf:
            json.dump(vocab, outf, ensure_ascii=False, indent=4)
        with open(
            join(tmp_dir, BPE_COD_FILE_NAME), 'wt', encoding='utf-8') as outf:
            outf.write('\n'.join([f'{k[0]}\t{k[1]}\t{codes[k]}'
                for k in sorted(codes, key=codes.get)]))
        with open(
            join(tmp_dir, BI_CNT_FILE_NAME), 'wt', encoding='utf-8') as outf:
            outf.write('\n'.join(
                [f'({k[0]}, {k[1]})\t{bigram2cnt[k]}'
                 for k in sorted(bigram2cnt, key=bigram2cnt.get,
                                 reverse=True)]))
        info('(main progress) BPE vocab/codes/bigram count saved')
        if len(codes) != iter_idx:
            # codes: not updated, vocab: not sure if updated or not
            warning('vocab and codes may not be in sync')
        with open(
            join(bpe_dict_dir, BPE_CHAR_DICT_NAME), 'wt', encoding='utf-8') as outf:
            json.dump(
                build_bpe_char_dict(
                    join(tmp_dir, BPE_SYM_FILE_NAME),
                    join(tmp_dir, BPE_COD_FILE_NAME)),
                outf, ensure_ascii=False, indent=4)
        info('(main progress) BPE dictionary built. SUCCESS!!!')


if __name__ == '__main__':
    info('[%s] module invoked', __file__)

    AP = argparse.ArgumentParser(description='args parser')
    AP.add_argument('-prj_root', action='store', required=True,
                    help='the project root path')
    AP.add_argument('-cleaned_corpus_dir', action='store', required=True,
                    help='path to hold preprocessed input corpus files')
    AP.add_argument('-split_corpus_dir', action='store', required=True,
                    help='path for split output corpus files')
    AP.add_argument('-split_ratio', action='store', required=True,
                    help='train:valid:test ratio - ex) 0.8:0.1:0.1 or 8:1:1')
    AP.add_argument('-tmp_dir', action='store', required=True,
                    help='path to save intermediate files')
    AP.add_argument('-bpe_dict_dir', action='store', required=True,
                    help='path for the output BPE dictionary file')
    ARGS = AP.parse_args()

    sys.path.append(join(ARGS.prj_root, 'dataset/Corpus'))

    _timer = TimeLogger()
    tr, va, te = [float(t) for t in ARGS.split_ratio.split(':')]
    tr, va, te = [t/(tr + va + te) for t in [tr, va, te]]
    main(ARGS.cleaned_corpus_dir, ARGS.split_corpus_dir, [tr, va, te],
         ARGS.tmp_dir, ARGS.bpe_dict_dir)
    info('main() takes time: %s', _timer.get_elapsed_time_str())
