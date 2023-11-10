import re
import html


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
    'ℓ': 'l',
    ####
    '\x89ûª': '\'',
    '\x81': '',
    '\x89': '',
    '\x9d': '',
    'û': '',
    'ª': '',
    'ï': '',
    'ó': '',
    'ì': 'i',
}


def preprocess_text(text):
    """Conducts some preprocessing steps on the input text.

    Preprocessing steps are composed of simple cleansing chores like
    lowercasing or redundant space reduction and somewhat laborious
    tasks like character-variants normalization.
    * text preprocessing steps in order:
    1. turn html entities to characters
    2. lowercasing
    3. strip
    4. CHAR_FILTER sub
    5. remove url
    6. space collapsing

    Args:
        text: The input text body to be preprocessed.

    Returns:
        A list of space-split word tokens that have been preprocessed.
    """
    rep = dict((re.escape(k), v) for k, v in CHAR_FILTER.items())
    pat1 = re.compile('|'.join(rep.keys()))
    pat2 = re.compile(r'https?://[\w-]+\.[\w-]+\S*')
    pat3 = re.compile(r'\B@\w+')  # remove mention.
    pat4 = re.compile('[ ]{2,}')

    text = html.unescape(text)
    text = text.lower()
    text = pat1.sub(lambda m: rep[re.escape(m.group(0))], text.strip())
    text = pat2.sub('', text)
    text = pat3.sub('', text)
    text = pat4.sub(' ', text)

    return text.split()
