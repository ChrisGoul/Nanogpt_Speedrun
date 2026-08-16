"""
Synthetic reasoning corpus generator — free, unlimited, clean, chain-of-thought.
Every example shows step-by-step working, so the model learns to REASON rather
than guess the answer. GSM8K-style format ("#### <answer>" at the end).

Problem types (mixable, difficulty-scalable):
  - column addition / subtraction with explicit carries/borrows
  - multiplication
  - multi-step word problems
  - one-step linear algebra (solve for x)
  - comparison / ordering
  - arithmetic sequences

Run `python reason_gen.py` to print samples. Later we scale to millions and encode.
"""
import random

R = random.Random(0)

def _num(maxd):
    d = R.randint(1, maxd)
    lo = 10 ** (d - 1) if d > 1 else 0
    return R.randint(lo, 10 ** d - 1)

def add_cot(maxd=3):
    a, b = _num(maxd), _num(maxd)
    sa, sb = str(a).zfill(max(len(str(a)), len(str(b)))), str(b).zfill(max(len(str(a)), len(str(b))))
    L = len(sa)
    lines, carry, res = [], 0, []
    for i in range(L - 1, -1, -1):
        s = int(sa[i]) + int(sb[i]) + carry
        cin = " + 1 carry" if carry else ""
        lines.append(f"ones place {sa[i]} + {sb[i]}{cin} = {s}, write {s % 10}" + (", carry 1" if s >= 10 else "")
                     if i == L - 1 else
                     f"next place {sa[i]} + {sb[i]}{cin} = {s}, write {s % 10}" + (", carry 1" if s >= 10 else ""))
        res.append(str(s % 10)); carry = s // 10
    if carry:
        res.append("1"); lines.append("final carry = 1")
    ans = int("".join(reversed(res)))
    return f"Question: What is {a} + {b}?\n" + "\n".join(lines) + f"\n#### {ans}", ans

def sub_cot(maxd=3):
    a, b = _num(maxd), _num(maxd)
    if b > a:
        a, b = b, a
    L = len(str(a))
    sa, sb = str(a).zfill(L), str(b).zfill(L)
    lines, res, borrow = [], [], 0
    for i in range(L - 1, -1, -1):
        t = int(sa[i]) - borrow
        d = int(sb[i])
        place = "ones place" if i == L - 1 else "next place"
        if t < d:
            lines.append(f"{place}: {t} + 10 - {d} = {t + 10 - d}, borrow 1")
            res.append(str(t + 10 - d)); borrow = 1
        else:
            lines.append(f"{place}: {t} - {d} = {t - d}")
            res.append(str(t - d)); borrow = 0
    return (f"Question: What is {a} - {b}?\n" + "\n".join(lines) + f"\n#### {a - b}"), a - b

def mul_cot(maxd=2):
    a, b = _num(maxd), _num(1)
    # distribute a across place values of... keep b single-digit for readability
    parts = []
    total = 0
    for i, ch in enumerate(reversed(str(a))):
        place = int(ch) * (10 ** i)
        parts.append(f"{place} x {b} = {place * b}")
        total += place * b
    return (f"Question: What is {a} x {b}?\n"
            f"Break {a} into place values and multiply each by {b}:\n"
            + "\n".join(parts) + f"\nAdd them: {a * b}.\n#### {a * b}"), a * b

def word_cot():
    name = R.choice(["Sarah", "Tom", "Maya", "Leo", "Ava", "Ben"])
    item = R.choice(["apples", "coins", "marbles", "stickers", "cookies"])
    start = R.randint(10, 40)
    give = R.randint(1, start // 2)
    buy = R.randint(1, 20)
    ans = start - give + buy
    return (f"Question: {name} has {start} {item}. {name} gives away {give} and then gets {buy} more. "
            f"How many {item} does {name} have now?\n"
            f"Start with {start}.\n"
            f"After giving away {give}: {start} - {give} = {start - give}.\n"
            f"After getting {buy} more: {start - give} + {buy} = {ans}.\n#### {ans}"), ans

def algebra_cot():
    x = R.randint(1, 30)
    k = R.randint(1, 20)
    if R.random() < 0.5:
        return (f"Question: Solve for x: x + {k} = {x + k}.\n"
                f"Subtract {k} from both sides: x = {x + k} - {k}.\nx = {x}.\n#### {x}"), x
    return (f"Question: Solve for x: x - {k} = {x - k}.\n"
            f"Add {k} to both sides: x = {x - k} + {k}.\nx = {x}.\n#### {x}"), x

def compare_cot():
    a, b = _num(2), _num(2)
    while a == b:
        b = _num(2)
    big = max(a, b)
    return (f"Question: Which is larger, {a} or {b}?\n"
            f"Compare the tens digit first, then the ones.\n{big} is larger.\n#### {big}"), big

def seq_cot():
    start = R.randint(1, 10)
    step = R.randint(2, 5)
    seq = [start + step * i for i in range(4)]
    ans = seq[-1] + step
    return (f"Question: What comes next: {', '.join(map(str, seq))}, ?\n"
            f"Each term increases by {step}.\n{seq[-1]} + {step} = {ans}.\n#### {ans}"), ans

NAMES = ["Tom", "Sam", "Bob", "Ava", "Leo", "Mia", "Ben", "Zoe"]

def transitive_cot():
    a, b, c = R.sample(NAMES, 3)
    comp, sup = R.choice([("taller", "tallest"), ("older", "oldest"), ("faster", "fastest"), ("heavier", "heaviest")])
    return (f"Question: {a} is {comp} than {b}. {b} is {comp} than {c}. Who is {sup}?\n"
            f"{a} is {comp} than {b}.\n{b} is {comp} than {c}.\n"
            f"So {a} is {comp} than everyone.\n#### {a}"), a

def syllogism_cot():
    a, b = R.choice([("cats", "animals"), ("roses", "plants"), ("dogs", "animals"), ("cars", "machines")])
    single = a[:-1]
    name = R.choice(NAMES)
    if R.random() < 0.5:
        return (f"Question: All {a} are {b}. {name} has a {single}. Is {name}'s {single} a kind of {b[:-1]}?\n"
                f"All {a} are {b}.\nA {single} is one of the {a}.\nSo yes.\n#### yes"), "yes"
    return (f"Question: All {a} are {b}. {name} has a rock. Is {name}'s rock a kind of {b[:-1]}?\n"
            f"All {a} are {b}, but a rock is not one of the {a}.\nSo no.\n#### no"), "no"

def boolean_cot():
    A, B = R.choice([True, False]), R.choice([True, False])
    op = R.choice(["and", "or"])
    val = (A and B) if op == "and" else (A or B)
    return (f"Question: A is {str(A).lower()}. B is {str(B).lower()}. Is (A {op} B) true?\n"
            f"A is {str(A).lower()} and B is {str(B).lower()}.\n"
            f"With {op}, the result is {str(val).lower()}.\n#### {'yes' if val else 'no'}"), "yes" if val else "no"

def rule_cot():
    p, q = R.choice([("it rains", "the ground is wet"), ("you study", "you pass"),
                     ("the sun is out", "it is bright"), ("the alarm rings", "you wake up")])
    if R.random() < 0.5:
        return (f"Question: If {p}, then {q}. We know that {p}. Is it true that {q}?\n"
                f"The rule says if {p}, then {q}.\nWe know {p}.\nSo {q}.\n#### yes"), "yes"
    return (f"Question: If {p}, then {q}. It is not the case that {p}. Can we be sure that {q}?\n"
            f"The rule only applies when {p}.\nSince {p} is not true, we cannot be sure.\n#### no"), "no"

def sort_cot():
    nums = R.sample(range(1, 60), 3)
    s = sorted(nums)
    return (f"Question: Put these in order from smallest to largest: {', '.join(map(str, nums))}.\n"
            f"The smallest is {s[0]}.\nNext is {s[1]}.\nLargest is {s[2]}.\n#### {', '.join(map(str, s))}"), ", ".join(map(str, s))

def setcount_cot():
    r, b = R.randint(2, 15), R.randint(2, 15)
    return (f"Question: There are {r} red balls and {b} blue balls. How many balls are not red?\n"
            f"The balls that are not red are the blue ones.\nThere are {b} blue balls.\n#### {b}"), b

GENERATORS = [add_cot, sub_cot, mul_cot, word_cot, algebra_cot, compare_cot, seq_cot,
              transitive_cot, syllogism_cot, boolean_cot, rule_cot, sort_cot, setcount_cot]

def sample():
    return R.choice(GENERATORS)()[0]

if __name__ == "__main__":
    for g in GENERATORS:
        print("=" * 60)
        print(g()[0])
        print()
