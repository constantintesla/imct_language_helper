"""(перенесено из lms_translater_src/dubservice/speech.py)
Text -> what the Russian TTS can actually pronounce: Latin words and acronyms, digits and symbols are written out in Cyrillic.
F5-TTS Russian reads Latin letters and digits badly, and the Whisper round-trip check expects Cyrillic, so both use this form."""
import json
import re

from num2words import num2words


VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
STRESS = re.compile(rf"\+(?=[{VOWELS}])")        # manual stress mark: "+" right before a vowel ("зам+ок")

LEXICON = {
    "python": "пайтон", "pycharm": "пайчарм", "pip": "пип", "pandas": "пандас", "numpy": "нампай", "jupyter": "джупитер",
    "matplotlib": "мэтплотлиб", "excel": "эксель", "windows": "виндоус", "linux": "линукс", "github": "гитхаб", "google": "гугл",
    "elif": "элиф", "else": "элс", "if": "иф", "for": "фор", "while": "уайл", "in": "ин", "and": "энд", "or": "ор", "not": "нот",
    "true": "тру", "false": "фолс", "none": "нан", "print": "принт", "input": "инпут", "int": "инт", "str": "стр",
    "float": "флоат", "list": "лист", "dict": "дикт", "set": "сет", "tuple": "тьюпл", "def": "деф", "return": "ритёрн",
    "import": "импорт", "from": "фром", "as": "эз", "chr": "чар", "ord": "орд", "len": "лен", "mod": "мод", "begin": "бегин",
    "end": "энд", "lambda": "лямбда", "kwargs": "кваргс", "args": "аргс", "break": "брейк", "continue": "континью",
    "pass": "пас", "try": "трай", "except": "эксепт", "class": "класс", "self": "сэлф", "range": "рейндж",
    "community": "комьюнити", "professional": "профешнл", "edition": "эдишн", "data": "дата", "analyst": "аналист",
    "titanic": "тайтаник", "covid": "ковид", "dataset": "датасет", "welcome": "уэлкам", "club": "клаб", "mike": "майк",
    "charm": "чарм", "tableau": "таблу", "power": "пауэр", "bi": "би-ай", "ai": "эй-ай", "it": "ай-ти", "ml": "эм-эл",
    "sql": "эс-кью-эл", "api": "эй-пи-ай", "csv": "си-эс-ви", "html": "эйч-ти-эм-эл", "url": "ю-ар-эл", "ide": "ай-ди-и",
    "to": "ту", "the": "зэ", "of": "оф", "is": "из", "on": "он", "you": "ю", "your": "ё", "hello": "хэллоу", "world": "ворлд",
    "gui": "джи-ю-ай", "pdf": "пи-ди-эф", "utf": "ю-ти-эф", "ascii": "аски", "ok": "окей", "a/b": "эй-би",
}
LETTERS = dict(zip("abcdefghijklmnopqrstuvwxyz", [
    "эй", "би", "си", "ди", "и", "эф", "джи", "эйч", "ай", "джей", "кей", "эл", "эм", "эн", "оу", "пи", "кью", "ар", "эс", "ти",
    "ю", "ви", "дабл-ю", "икс", "уай", "зет"]))
FOLD = {"питон": "пайтон", "пайчарм": "пайчарм", "пайчарме": "пайчарм"}      # what Whisper tends to write for the lexicon words

DIGRAPHS = [("sch", "ск"), ("tch", "ч"), ("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"), ("ck", "к"), ("qu", "кв"),
            ("kn", "н"), ("wh", "в"), ("oo", "у"), ("ee", "и"), ("ea", "и"), ("ou", "ау"), ("ow", "ау"), ("ai", "эй"),
            ("ay", "эй"), ("oa", "оу"), ("igh", "ай"), ("ng", "нг")]
SINGLE = dict(zip("abcdefghijklmnopqrstuvwxyz", ["а", "б", "к", "д", "е", "ф", "г", "х", "и", "дж", "к", "л", "м", "н", "о", "п", "к",
                                                   "р", "с", "т", "у", "в", "в", "кс", "и", "з"]))


def translit(word):
    """Rough phonetic Cyrillic spelling for an unknown Latin word."""
    w, out, i = word.lower(), "", 0
    while i < len(w):
        for dg, cy in DIGRAPHS:
            if w.startswith(dg, i):
                out += cy
                i += len(dg)
                break
        else:
            out += SINGLE.get(w[i], w[i])
            i += 1
    return out


def spell(word):
    return "-".join(LETTERS[c] for c in word.lower() if c in LETTERS)


def say_latin(tok):
    low = tok.lower()
    if low in LEXICON:
        return LEXICON[low]
    if len(tok) == 1 and low in LETTERS:
        return LETTERS[low]                                # a variable or a letter named in the lecture: T, W, a
    if tok.isupper() and 1 < len(tok) <= 5:
        return spell(tok)                                  # acronym: BI, SQL, HTTP
    return translit(tok)


def number(m):
    s = m.group(0)
    try:
        if "." in s and s.count(".") == 1 and len(s.split(".")[1]) <= 3 and not s.endswith("."):       # 782.67 -> decimal
            a, b = s.split(".")
            return f"{num2words(int(a), lang='ru')} запятая {num2words(int(b), lang='ru')}"
        return " точка ".join(num2words(int(p), lang="ru") for p in s.split("."))                       # 3.5.2, 2.7.12, 98
    except Exception:
        return s


def apply_dictionary(text, d):
    if not d:
        return text

    def rep(m):
        w = m.group(0)
        v = d.get(w.lower())
        if "+" in w or v is None:                         # a stress typed by hand wins over the dictionary
            return w
        if len(w) > 1 and w.isupper():                      # an abbreviation (IDE): the replacement is a phrase, not a sentence start
            return v
        return v[:1].upper() + v[1:] if w[:1].isupper() and v[:1].islower() else v
    return re.sub(r"[A-Za-zА-Яа-яЁё+]+", rep, text)


def to_spoken(text, dictionary=None):
    t = apply_dictionary(text, dictionary)
    t = re.sub(r"\b(?:[A-Za-z]-){2,}[A-Za-z]\b", lambda m: spell(m.group(0).replace("-", "")), t)       # C-H-A-R-M
    t = re.sub(r"\s*\*\*\s*", " в степени ", t)                                                           # 2 ** 3
    t = re.sub(r"(?<=[\w)])\s*//\s*(?=[\w(])", " двойной слэш ", t)                                        # 7 // 3
    t = re.sub(r"(?<=\d)\s*%\s*(?=[\d(])", " по модулю ", t)                                              # 7 % 3: the operator, not percent
    t = re.sub(r"(?<=\d)\s*/\s*(?=\d)", " делить на ", t)                                                 # 1/2, 98/7
    t = re.sub(r"(?<=\d)\s*[*×]\s*(?=\d)", " умножить на ", t)                                            # 3 * 2
    t = re.sub(r"(\d+)\s*%", r"\1 процентов", t)
    t = re.sub(r"\b[Aa]/[Bb]\b", "эй-би", t)
    t = re.sub(r"\b([A-Za-z]+)-(\d+)\b", lambda m: f"{say_latin(m.group(1))} {m.group(2)}", t)           # UTF-8, COVID-19
    t = re.sub(r"\d{1,3}(?:[   ]\d{3})+(?!\d)", lambda m: re.sub(r"\D", "", m.group(0)), t)    # 15 000 -> 15000
    t = re.sub(r"\d+(?:\.\d+)*", number, t)
    t = re.sub(r"[A-Za-z]+", lambda m: say_latin(m.group(0)), t)
    t = STRESS.sub("\0", t)                                     # keep manual stress marks, the rest of "+" is the maths sign
    t = t.replace("=", " равно ").replace("+", " плюс ").replace("\0", "+")
    return re.sub(r"\s+", " ", t).strip()
