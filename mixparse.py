#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mixparse.py — 多语言混合文本解析器（仅依赖 Python 标准库）

功能：
  * 按“当前语言状态”扫描文本，只有遇到切换标记才切换语言；
  * 语言嵌套遵循栈纪律（后开先关：后进入的语言先退出）；
  * 当前语言的字符串 / 注释内部出现的切换标记被豁免（不触发切换）；
  * 错误报告：未闭合的语言状态（含起始行）、切换标记语法错误、
    同一位置同时命中多条规则（歧义）、嵌套顺序错误、未匹配的退出标记、
    未闭合的字符串 / 注释；
  * 输出结构树（每段的语言类型与内容）与错误报告。

用法：
  python3 mixparse.py 输入文件 [-c 配置.json] [--json] [--print-config]
  缺省使用内置 markup/script 配置；输入缺省读标准输入。
  退出码：0 无错误；1 有解析错误；2 配置/IO 错误。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------- 内置配置

DEFAULT_CONFIG = {
    "root": "markup",
    "languages": {
        "markup": {
            "enter": ["<@"],            # 从 script 嵌回 markup 的进入标记
            "exit": ["@>"],             # 退出该 markup 块、回到 script
            "strings": [["\"", "\""], ["'", "'"]],
            "line_comments": [],
            "block_comments": [["<!--", "-->"]],
            "bad_markers": [r"<\s+@", r"@\s+>"],
        },
        "script": {
            "enter": ["<script>"],
            "exit": ["</script>"],
            "strings": [["\"", "\""], ["'", "'"]],
            "line_comments": ["//"],
            "block_comments": [["/*", "*/"]],
            "bad_markers": [
                r"<script(?![>])",          # <script 后不是 >
                r"</script(?![>])",         # </script 后不是 >
                r"<\s*/\s*script\s*>",      # </ script> 之类
            ],
        },
    },
}


class ConfigError(Exception):
    pass


# ---------------------------------------------------------------- 数据结构

@dataclass
class Language:
    name: str
    enter: list                 # 进入该语言的切换标记（字面量）
    exit: list                  # 退出该语言的切换标记（字面量）
    strings: list               # [(open, close)] 字符串界符
    line_comments: list         # [prefix] 行注释前缀
    block_comments: list        # [(open, close)] 块注释界符
    bad_markers: list           # [编译后的正则] 形似但非法的切换标记
    escape: str = "\\"          # 字符串转义符（空串表示无转义）


@dataclass
class Node:
    language: str
    start_line: int
    end_line: int | None = None     # None 表示未闭合
    parts: list = field(default_factory=list)   # str（文本段） | Node（子语言块）


@dataclass
class ParseError:
    line: int
    kind: str       # unclosed / bad-marker / ambiguity / nesting / unmatched-exit / unterminated
    message: str

    def __str__(self):
        return f"line {self.line}: [{self.kind}] {self.message}"


# ---------------------------------------------------------------- 配置加载

def load_config(cfg: dict):
    """校验配置并返回 (root_name, {name: Language})。"""
    if not isinstance(cfg, dict):
        raise ConfigError("配置必须是 JSON 对象")
    langs_raw = cfg.get("languages")
    if not isinstance(langs_raw, dict) or not langs_raw:
        raise ConfigError("配置缺少非空的 'languages' 对象")
    root = cfg.get("root")
    if root not in langs_raw:
        raise ConfigError(f"root 语言 {root!r} 未在 languages 中定义")

    def str_list(name, key):
        v = langs_raw[name].get(key, [])
        if not isinstance(v, list) or any(not isinstance(x, str) or not x for x in v):
            raise ConfigError(f"语言 {name!r} 的 {key!r} 必须是非空字符串数组")
        return v

    def pair_list(name, key):
        v = langs_raw[name].get(key, [])
        ok = isinstance(v, list) and all(
            isinstance(p, list) and len(p) == 2
            and all(isinstance(x, str) and x for x in p) for p in v)
        if not ok:
            raise ConfigError(f"语言 {name!r} 的 {key!r} 必须是 [open, close] 二元组数组")
        return [tuple(p) for p in v]

    languages = {}
    for name in langs_raw:
        if not isinstance(langs_raw[name], dict):
            raise ConfigError(f"语言 {name!r} 的定义必须是对象")
        bad = []
        for pat in str_list(name, "bad_markers"):
            try:
                bad.append(re.compile(pat))
            except re.error as e:
                raise ConfigError(f"语言 {name!r} 的 bad_markers 正则 {pat!r} 非法: {e}")
        esc = langs_raw[name].get("escape", "\\")
        if esc is None:
            esc = ""
        if not isinstance(esc, str) or len(esc) > 1:
            raise ConfigError(f"语言 {name!r} 的 escape 必须是单个字符或 null")
        languages[name] = Language(
            name=name,
            enter=str_list(name, "enter"),
            exit=str_list(name, "exit"),
            strings=pair_list(name, "strings"),
            line_comments=str_list(name, "line_comments"),
            block_comments=pair_list(name, "block_comments"),
            bad_markers=bad,
            escape=esc,
        )
    return root, languages


# ---------------------------------------------------------------- 扫描器

class Scanner:
    """单遍扫描：状态 = 语言栈；每个位置按固定优先级尝试匹配。"""

    def __init__(self, text: str, root: str, languages: dict):
        self.text = text
        self.n = len(text)
        self.root_name = root
        self.languages = languages
        self.errors: list = []
        self.pos = 0
        self.line = 1

    def _advance_to(self, new_pos: int):
        if new_pos > self.pos:
            self.line += self.text.count("\n", self.pos, new_pos)
            self.pos = new_pos

    def _error(self, kind, message, line=None):
        self.errors.append(ParseError(self.line if line is None else line, kind, message))

    # -- 豁免区：字符串 / 注释，整体作为普通文本消费 --

    def _try_string(self, lang, buf) -> bool:
        for op, cl in lang.strings:
            if self.text.startswith(op, self.pos):
                start_line = self.line
                i = self.pos + len(op)
                esc = lang.escape
                closed = False
                while i < self.n:
                    if esc and self.text[i] == esc:
                        i += 2
                        continue
                    if self.text.startswith(cl, i):
                        i += len(cl)
                        closed = True
                        break
                    i += 1
                end = min(i, self.n)
                buf.append(self.text[self.pos:end])
                self._advance_to(end)
                if not closed:
                    self._error("unterminated",
                                f"语言 {lang.name!r} 的字符串（始于第 {start_line} 行）未闭合",
                                line=start_line)
                return True
        return False

    def _try_line_comment(self, lang, buf) -> bool:
        for prefix in lang.line_comments:
            if self.text.startswith(prefix, self.pos):
                j = self.text.find("\n", self.pos)
                end = self.n if j == -1 else j
                buf.append(self.text[self.pos:end])
                self._advance_to(end)
                return True
        return False

    def _try_block_comment(self, lang, buf) -> bool:
        for op, cl in lang.block_comments:
            if self.text.startswith(op, self.pos):
                start_line = self.line
                j = self.text.find(cl, self.pos + len(op))
                if j == -1:
                    buf.append(self.text[self.pos:])
                    self._advance_to(self.n)
                    self._error("unterminated",
                                f"语言 {lang.name!r} 的块注释（始于第 {start_line} 行）未闭合",
                                line=start_line)
                else:
                    end = j + len(cl)
                    buf.append(self.text[self.pos:end])
                    self._advance_to(end)
                return True
        return False

    # -- 切换标记 --

    def _marker_matches(self):
        """收集当前位置命中的全部切换规则（所有语言的 enter / exit）。"""
        seen, matches = set(), []
        for lname, lang in self.languages.items():
            for kind, markers in (("enter", lang.enter), ("exit", lang.exit)):
                for m in markers:
                    key = (kind, lname, m)
                    if key not in seen and self.text.startswith(m, self.pos):
                        seen.add(key)
                        matches.append(key)
        return matches

    def _resolve(self, matches):
        """歧义消解：最长匹配优先；等长时 exit 优先于 enter。返回选中项。"""
        if len(matches) == 1:
            return matches[0]
        desc = ", ".join(f"{k}({l} {m!r})" for k, l, m in matches)
        matches = sorted(matches, key=lambda t: (-len(t[2]), 0 if t[0] == "exit" else 1))
        chosen = matches[0]
        self._error("ambiguity",
                    f"同一位置命中多条规则（歧义）：{desc}；"
                    f"按“最长匹配、退出优先”消解为 {chosen[0]}({chosen[1]})")
        return chosen

    def _handle_exit(self, lname, marker, stack, buf, flush):
        top_name, top_node = stack[-1]
        if lname == top_name and len(stack) > 1:
            flush()
            top_node.end_line = self.line
            stack.pop()
            self._advance_to(self.pos + len(marker))
        elif lname == top_name:  # 栈底是 root，无可退出的外层
            self._error("unmatched-exit",
                        f"退出标记 {marker!r}（语言 {lname!r}）没有匹配的进入标记，已按普通文本处理")
            buf.append(marker)
            self._advance_to(self.pos + len(marker))
        elif any(name == lname for name, _ in stack):
            # 退出标记属于某个外层语言：违反“后开先关”
            self._error("nesting",
                        f"退出标记 {marker!r} 属于外层语言 {lname!r}，但内层语言 "
                        f"{top_name!r}（第 {top_node.start_line} 行进入）尚未关闭；"
                        f"嵌套须后开先关，已按普通文本处理")
            buf.append(marker)
            self._advance_to(self.pos + len(marker))
        else:
            self._error("unmatched-exit",
                        f"退出标记 {marker!r}（语言 {lname!r}）没有匹配的进入标记，已按普通文本处理")
            buf.append(marker)
            self._advance_to(self.pos + len(marker))

    def _try_bad_marker(self) -> bool:
        # 进入标记全局生效，故“形似但非法”的标记模式也对所有语言检查
        for lang in self.languages.values():
            for pat in lang.bad_markers:
                m = pat.match(self.text, self.pos)
                if m:
                    self._error("bad-marker",
                                f"疑似语法错误的切换标记 {m.group(0)!r}"
                                f"（语言 {lang.name!r} 的标记），已跳过")
                    self._advance_to(m.end())
                    return True
        return False

    # -- 主循环 --

    def scan(self) -> Node:
        root_node = Node(self.root_name, 1)
        stack = [(self.root_name, root_node)]   # [(语言名, 对应树节点)]
        buf = []

        def flush():
            if buf:
                stack[-1][1].parts.append("".join(buf))
                buf.clear()

        while self.pos < self.n:
            lang = self.languages[stack[-1][0]]

            # 1) 豁免区优先：字符串、行注释、块注释整体消费
            if self._try_string(lang, buf):
                continue
            if self._try_line_comment(lang, buf):
                continue
            if self._try_block_comment(lang, buf):
                continue

            # 2) 切换标记（含歧义检测与消解）
            matches = self._marker_matches()
            if matches:
                kind, lname, marker = self._resolve(matches)
                if kind == "enter":
                    flush()
                    child = Node(lname, self.line)
                    stack[-1][1].parts.append(child)
                    stack.append((lname, child))
                    self._advance_to(self.pos + len(marker))
                else:
                    self._handle_exit(lname, marker, stack, buf, flush)
                continue

            # 3) 形似但非法的切换标记
            if self._try_bad_marker():
                continue

            # 4) 普通字符，归入当前语言的文本段
            buf.append(self.text[self.pos])
            self._advance_to(self.pos + 1)

        flush()
        # EOF：凡仍压在栈上的语言状态即“未闭合”，报告其进入行
        while len(stack) > 1:
            lname, node = stack.pop()
            self.errors.append(ParseError(
                node.start_line, "unclosed",
                f"语言 {lname!r} 自第 {node.start_line} 行进入后未闭合（缺少退出标记）"))
        root_node.end_line = self.line
        return root_node


# ---------------------------------------------------------------- 输出

def node_to_dict(node: Node) -> dict:
    return {
        "language": node.language,
        "start_line": node.start_line,
        "end_line": node.end_line,
        "parts": [p if isinstance(p, str) else node_to_dict(p) for p in node.parts],
    }


def pretty(node: Node, depth=0, out=None, max_text=72):
    if out is None:
        out = []
    pad = "  " * depth
    end = node.end_line if node.end_line is not None else "?"
    out.append(f"{pad}<{node.language}>  (lines {node.start_line}-{end})")
    for p in node.parts:
        if isinstance(p, Node):
            pretty(p, depth + 1, out, max_text)
        else:
            s = p.replace("\n", "\\n")
            if len(s) > max_text:
                s = s[: max_text - 3] + "..."
            out.append(f"{pad}  text: {s!r}")
    return out


# ---------------------------------------------------------------- 入口

def main(argv=None):
    ap = argparse.ArgumentParser(description="多语言混合文本解析器（结构树 + 错误报告）")
    ap.add_argument("input", nargs="?", help="输入文件（缺省读标准输入）")
    ap.add_argument("-c", "--config", help="语言切换标记定义（JSON）；缺省用内置 markup/script 配置")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结构树与错误")
    ap.add_argument("--print-config", action="store_true", help="打印内置配置后退出")
    args = ap.parse_args(argv)

    if args.print_config:
        json.dump(DEFAULT_CONFIG, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.config:
        try:
            with open(args.config, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"读取配置失败: {e}", file=sys.stderr)
            return 2
    else:
        cfg = DEFAULT_CONFIG

    try:
        root, languages = load_config(cfg)
    except ConfigError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        return 2

    try:
        text = open(args.input, encoding="utf-8").read() if args.input else sys.stdin.read()
    except OSError as e:
        print(f"读取输入失败: {e}", file=sys.stderr)
        return 2

    scanner = Scanner(text, root, languages)
    tree = scanner.scan()

    if args.json:
        json.dump({
            "tree": node_to_dict(tree),
            "errors": [e.__dict__ for e in scanner.errors],
        }, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print("== 结构树 ==")
        for line in pretty(tree):
            print(line)
        print()
        print("== 错误报告 ==")
        if scanner.errors:
            for e in scanner.errors:
                print(f"  {e}")
        else:
            print("  （无）")
    return 1 if scanner.errors else 0


if __name__ == "__main__":
    sys.exit(main())
