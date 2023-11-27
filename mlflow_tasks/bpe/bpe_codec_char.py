"""This module encodes input corpus files with a given BPE dict.

The module receives a text body and cleanses it using parsers, and encodes
the cleansed text into a sequence of BPE token IDs. This module takes in only
input files with extension '.txt' and generates output files with the leading
'BPE_char_' prefix in the specified output path.

Usage example:

    bpe_codec_char.py -bpe_dict_path        bpe dictionary file path
                      -cleanser             cleanser name
                      -input_corpus_dir     input corpus files directory
                      -output_corpus_dir    output corpus files directory
                      -do_test              boolean flag
                      -test_line            input text line for test
"""

import argparse
import json
import re
import sys
from argparse import RawTextHelpFormatter
from bpe.bpe_dict_char import LEADING_WORD_SYMBOL
from os import listdir, mkdir, remove
from os.path import basename, isdir, isfile, join
from hs_aiteam_pkgs.util.logger import debug, error, info
from hs_aiteam_pkgs.util.time_logger import TimeLogger


cleansers = {
    'AIHub_KoEn': None,
    'Pile': None
}


def _get_bigrams(tokens):
    bigrams = set()
    for i in range(len(tokens) - 1):
        bigrams.add((tokens[i], tokens[i + 1]))
    return bigrams


def encode_bpe_char(bpe_dict, cleanser, text):
    """Encodes the given text body using a BPE dict and a text cleanser.

    Args:
        bpe_dict: Character-level BPE dictionary.
        cleanser: Text cleanser with some predefined preprocessing steps.
        text: Input text body to encode.

    Returns:
        A list of character-level BPE token IDs.
    """
    embed_ids = []
    words = cleanser(text)  # ['100ml의', '용액당', '24sq', 'cm의']
    for word in words:  # '100ml의'
        sub_words = []
        if '__MASK__' in word.upper():
            m = word.replace('__mask__', ' __mask__ ').split()
            for i, mm in enumerate(m):
                if mm == '__mask__':
                    sub_words.append(mm)
                else:
                    tokens = list(mm)
                    if i == 0:
                        tokens = [LEADING_WORD_SYMBOL] + tokens
                    sub_words.append(tokens)
        else:
            # ['@@', '1', '0', '0', 'm', 'l', '의']
            tokens = [LEADING_WORD_SYMBOL] + list(word)
            sub_words.append(tokens)

        for tokens in sub_words:
            if tokens == '__mask__':
                embed_ids.append(bpe_dict['__MASK__'])
                continue
            if len(tokens) == 1:
                embed_ids.append(bpe_dict.get(tokens[0], bpe_dict['__UNK__']))
                continue

            # {('1','0'),('m','l'),('0','0'),('@@','1'),('l','의'),('0','m')}
            bigrams = _get_bigrams(tokens)
            while True:
                best_bigram = min(
                    (bpe_dict.get(''.join(bigram), sys.maxsize), bigram)
                    for bigram in bigrams)
                if ''.join(best_bigram[1]) not in bpe_dict:
                    break
                i = 0
                first, second = best_bigram[1]  # first, second = ('@@', '1')
                next_tokens = []
                while i < len(tokens):
                    if first not in tokens[i:]:
                        next_tokens.extend(tokens[i:])
                        break
                    j = tokens.index(first, i)
                    next_tokens.extend(tokens[i:j])
                    if ((j + 1) < len(tokens)) and (tokens[j + 1] == second):
                        next_tokens.append(first + second)
                        i = j + 2
                    else:
                        next_tokens.append(first)
                        i = j + 1
                tokens = next_tokens
                if len(tokens) == 1:
                    break
                bigrams = _get_bigrams(tokens)
            embed_ids.extend([bpe_dict.get(token, 0) for token in tokens])
    return embed_ids


def test_bpe(bpe_dict, cleanser, test_line):
    """A kind of unit test for a single text chunk.

    You may test to see the result of the BPE encoding/decoding tasks
    on a simple text line.

    Args:
        bpe_dict: Character-level BPE dictionary.
        cleanser: Text cleanser with some predefined preprocessing steps.
        test_line: Text line to test for BPE encoding/decoding.
    """
    embed_ids = encode_bpe_char(bpe_dict, cleanser, test_line)
    info('(test_bpe) test line: %s', test_line)
    info('(test_bpe) BPE encoding result: %s', embed_ids)
    id2token = {v:k for k, v in bpe_dict.items()}
    info('(test_bpe) BPE decoding result: %s',
        [id2token[ID] for ID in embed_ids])


def main(bpe_dict, cleanser, input_corpus_dir, output_corpus_dir):
    """Covers the whole process to encode a group of corpora using BPE.

    Args:
        bpe_dict: Character-level BPE dictionary.
        cleanser: Text cleanser with some predefined preprocessing steps.
        input_corpus_dir: Directory path to input corpus files(.txt).
        output_corpus_dir: Directory path to output corpus files(BPE_char_*)
    """
    if not isdir(input_corpus_dir):
        error('input corpora path is invalid')
        sys.exit(1)
    if not output_corpus_dir:
        error('output_corpus_dir argument is invalid')
        sys.exit(1)
    if not isdir(output_corpus_dir):
        mkdir(output_corpus_dir)
    else:
        pat = re.compile('^BPE_char_')
        for fname in listdir(output_corpus_dir):
            if pat.match(fname) is not None:
                remove(join(output_corpus_dir, fname))

    for fname in listdir(input_corpus_dir):
        fname = join(input_corpus_dir, fname)
        if not (isfile(fname) and fname.endswith('.txt')):
            continue
        with open(fname, 'rt', encoding='utf-8') as inf:
            lines = list(inf.readlines())
            num_total_lines = len(lines)
        fname = basename(fname)
        with open(join(output_corpus_dir, 'BPE_char_' + fname), 'wt',
            encoding='utf-8') as outf:
            _timer = TimeLogger()
            for i, line in enumerate(lines):
                embed_ids = [
                    str(i) for i in encode_bpe_char(bpe_dict, cleanser, line)]
                outf.write(','.join(embed_ids) + '\n')
                if (i + 1) % 10000 == 0:
                    debug('(BPE encoding progress) %d/%d, elapsed: %ds',
                          i + 1, num_total_lines,
                          _timer.get_elapsed_time_seconds())
                    _timer.reset_start_time()
        info('(BPE encoding progress) %s corpus fully encoded', fname)


if __name__ == '__main__':
    info('[%s] module invoked', __file__)

    AP = argparse.ArgumentParser(description='args parser',
                                 formatter_class=RawTextHelpFormatter)
    AP.add_argument('-prj_root', action='store', required=True,
                    help='the project root path')
    AP.add_argument('-bpe_dict_path', action='store', required=True,
                    help='the BPE dictionary file path')
    AP.add_argument('-cleanser', action='store', required=True,
                    help='the cleanser to use for corpus cleansing:\n' +
                    '\n'.join('- ' + name for name in cleansers))
    AP.add_argument('-input_corpus_dir', action='store', default='',
                    help='directory path to input corpus files(.txt)')
    AP.add_argument('-output_corpus_dir', action='store', default='',
                    help='directory path to output corpus files(BPE_char_*)')
    AP.add_argument('-do_test', action='store', type=bool, default=False,
                    help='a simple test for BPE codec')
    AP.add_argument('-test_line', action='store',
                    default='100㎖의 용액당 24㎠의 차가운 아세트산을' +\
                            ' 첨가하는 것은 정확도를 높여줍니다.',
                    help='a test line for BPE codec test')
    ARGS = AP.parse_args()

    if not isfile(ARGS.bpe_dict_path):
        error('BPE dictionary file path is invalid')
        sys.exit(1)
    if ARGS.cleanser not in cleansers:
        error('invalid cleanser name passed!!!')
        sys.exit(1)

    sys.path.append(join(ARGS.prj_root, 'dataset/Corpus/parsers'))
    from AIHub_KoEn.parse_excels import preprocess_text as cleanse_text_1
    cleansers['AIHub_KoEn'] = cleanse_text_1
    # TODO add the text preprocessor function of Pile as cleanse_text_2 later
    # cleansers['Pile'] = cleanse_text_2

    with open(ARGS.bpe_dict_path, 'rt', encoding='utf-8') as _inf:
        _bpe_dict = json.load(_inf)
    _cleanser = cleansers[ARGS.cleanser]

    if ARGS.do_test:
        test_bpe(_bpe_dict, _cleanser, ARGS.test_line)
        sys.exit(0)

    _timer = TimeLogger()
    main(_bpe_dict, _cleanser, ARGS.input_corpus_dir, ARGS.output_corpus_dir)
    info('main() takes time: %s', _timer.get_elapsed_time_str())
