#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mixed_parser.py — 多语言混合文本解析器（纯 Python 标准库，单文件）

核心行为
--------
* 按"当前语言状态"扫描文本，只有遇到切换标记才改变语言状态；
* 嵌套遵循"后开先关"（栈式管理），乱序闭合会被报告并自动恢复；
* 字符串 / 注释中的切换标记被豁免（不算切换）；
* 语言状态未闭合（进入后没有退出标记）时报告其起始行；
* 切换标记"形似但语法错误"（前缀命中、完整模式不命中）会被报告；
* 同一位置同时命中多条规则（歧义）会被报告，并按确定规则裁决；
* 输出结构树（各段落的语言类型与内容）与错误报告。

配置格式（JSON，结构见 DEFAULT_CONFIG）
---------------------------------------
{
  "root": "markup",                     // 初始（最外层）语言
  "languages": {
    "<语言名>": {
      "enter":    [{"pattern": "<正则>", "prefix": "<前缀>", "label": "<说明>"}],
      "exit":     [...],                // 同上；root 语言的 exit 用于发现"多余闭合"
      "strings":  [["起始", "结束", "转义符(可选)"]],
      "comments": [["起始", "结束"]]
    }
  }
}

enter 标记定义在"目标语言"上：在任何语言中遇到 X 的 enter 标记即进入 X
（包括 X 自身，允许自嵌套）；exit 标记只对栈中对应语言生效。

用法
----
  python3 mixed_parser.py 文件 [-c 配置.json] [--json]
  python3 mixed_parser.py --text '源码字符串'
  python3 mixed_parser.py --demo          # 运行内置示例
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------- 数据模型

@dataclass
class Marker:
    """一条切换标记：enter=进入 language，exit=退出 language。"""
    language: str
    kind: str               # 'enter' | 'exit'
    regex: re.Pattern
    prefix: str | None      # 用于探测"形似标记但语法错误"
    label: str


@dataclass
class Node:
    """结构树节点：一段同一语言的文本，parts 中文本与子语言段按序交替。"""
    language: str
    start_line: int
    parts: list = field(default_factory=list)   # 元素为 str 或 Node
    end_line: int | None = None


@dataclass
class ParseError:
    line: int
    kind: str
    message: str

    def __str__(self) -> str:
        return f"第 {self.line} 行 [{self.kind}] {self.message}"


class Language:
    """一门语言的词法外壳：切换标记 + 需要豁免的字符串/注释。"""

    def __init__(self, name: str, spec: dict):
        self.name = name
        self.enter_markers = [self._marker(m, "enter") for m in spec.get("enter", [])]
        self.exit_markers = [self._marker(m, "exit") for m in spec.get("exit", [])]
        self.strings = [self._pair(p, "字符串") for p in spec.get("strings", [])]
        self.comments = [self._pair(p, "注释") for p in spec.get("comments", [])]

    def _marker(self, spec: dict, kind: str) -> Marker:
        try:
            regex = re.compile(spec["pattern"])
        except re.error as exc:
            raise ValueError(f"语言 {self.name!r} 的 {kind} 标记正则非法: {exc}") from exc
        return Marker(self.name, kind, regex, spec.get("prefix"),
                      spec.get("label") or spec["pattern"])

    @staticmethod
    def _pair(pair: list, kind: str):
        if len(pair) == 2:
            start, end, esc = pair[0], pair[1], None
        elif len(pair) == 3:
            start, end, esc = pair
        else:
            raise ValueError(f"{kind}定义需为 [起始, 结束] 或 [起始, 结束, 转义符]，得到: {pair!r}")
        return (start, end, esc, kind)


# ---------------------------------------------------------------- 解析器

class Parser:
    def __init__(self, config: dict):
        langs = config.get("languages")
        if not isinstance(langs, dict) or not langs:
            raise ValueError("配置缺少非空的 languages 表")
        self.root_lang = config.get("root")
        if self.root_lang not in langs:
            raise ValueError(f"root 语言 {self.root_lang!r} 未在 languages 中定义")
        self.langs = {name: Language(name, spec) for name, spec in langs.items()}

    # ---- 主循环 -------------------------------------------------------

    def parse(self, text: str):
        self.text = text
        self._line_starts = [0] + [m.end() for m in re.finditer("\n", text)]
        self.errors: list[ParseError] = []

        root = Node(self.root_lang, 1)
        stack = [root]                    # 语言状态栈：栈底是 root 语言
        pos, n = 0, len(text)
        buf = []                          # 当前语言段累积的纯文本

        def flush():
            if buf:
                stack[-1].parts.append("".join(buf))
                buf.clear()

        while pos < n:
            cur = self.langs[stack[-1].language]

            # 1) 豁免：字符串 / 注释整体跳过，其中的切换标记不生效
            skip_end = self._match_skippable(cur, pos)
            if skip_end is not None:
                buf.append(text[pos:skip_end])
                pos = skip_end
                continue

            # 2) 切换标记：收集该位置所有命中（退出栈中语言 / 进入任意语言）
            hits = self._match_markers(stack, pos)
            if hits:
                chosen = self._resolve(hits, stack, pos)
                flush()
                pos = self._apply(chosen, stack, pos)
                continue

            # 3) 语法错误：前缀命中但完整模式不命中（形似切换标记）
            broken = self._match_broken_prefix(stack, pos)
            if broken is not None:
                self.errors.append(ParseError(
                    self.line_of(pos), "标记语法错误",
                    f"疑似切换标记 {broken.prefix!r}，但不符合 {broken.label!r} 的语法，已按普通文本处理"))
                buf.append(text[pos:pos + len(broken.prefix)])
                pos += len(broken.prefix)
                continue

            # 4) 普通字符
            buf.append(text[pos])
            pos += 1

        flush()

        # 文件结束时仍压在栈上的语言：未闭合，报告各自的起始行
        for node in reversed(stack[1:]):
            self.errors.append(ParseError(
                node.start_line, "语言未闭合",
                f"语言 {node.language!r} 自第 {node.start_line} 行进入后，到文件末尾未遇到退出标记"))
        return root, self.errors

    # ---- 各阶段 -------------------------------------------------------

    def line_of(self, pos: int) -> int:
        return bisect.bisect_right(self._line_starts, pos)

    def _match_skippable(self, lang: Language, pos: int):
        """若 pos 处是字符串/注释起点，返回其结束位置；未闭合则报错并跳到文件尾。"""
        text = self.text
        for start, end, esc, kind in lang.strings + lang.comments:
            if not text.startswith(start, pos):
                continue
            i = pos + len(start)
            while i < len(text):
                if esc and text.startswith(esc, i):
                    i += len(esc) + 1           # 跳过转义符及其后一个字符
                    continue
                if text.startswith(end, i):
                    return i + len(end)
                i += 1
            self.errors.append(ParseError(
                self.line_of(pos), f"{kind}未闭合",
                f"{kind} {start!r} 从第 {self.line_of(pos)} 行开始，到文件末尾未闭合"))
            return len(text)
        return None

    def _match_markers(self, stack: list, pos: int):
        """收集 pos 处命中的全部切换标记。

        返回 [(匹配文本, Marker, 栈下标或 None)]：栈下标表示退出栈中哪层语言，
        None 表示进入新语言。检查所有命中是为了发现歧义与乱序闭合。
        """
        hits = []
        seen_exit = set()
        # 自栈顶向下：同一退出标记只命中"最内层"的对应语言（内层优先闭合）
        for idx in range(len(stack) - 1, -1, -1):
            for mk in self.langs[stack[idx].language].exit_markers:
                if id(mk) in seen_exit:
                    continue
                m = mk.regex.match(self.text, pos)
                if m and m.end() > pos:
                    seen_exit.add(id(mk))
                    hits.append((m.group(0), mk, idx))
        for lang in self.langs.values():
            for mk in lang.enter_markers:
                m = mk.regex.match(self.text, pos)
                if m and m.end() > pos:
                    hits.append((m.group(0), mk, None))
        return hits

    def _resolve(self, hits: list, stack: list, pos: int):
        """歧义裁决：最长匹配优先，其次"退出栈顶"优先于"进入"优先于"退出深层"。"""
        if len(hits) > 1:
            desc = "、".join(self._describe(h, stack) for h in hits)
            self.errors.append(ParseError(
                self.line_of(pos), "歧义",
                f"同一位置同时匹配多条规则：{desc}；"
                f"已按“最长匹配，其次栈顶退出 > 进入 > 深层退出”裁决"))
        top = len(stack) - 1

        def rank(hit):
            text, _mk, idx = hit
            prio = 0 if idx == top else (1 if idx is None else 2)
            return (-len(text), prio)

        return min(hits, key=rank)

    def _describe(self, hit, stack) -> str:
        text, mk, idx = hit
        if idx is None:
            return f"{text!r}→进入 {mk.language!r}"
        return f"{text!r}→退出 {mk.language!r}（第 {stack[idx].start_line} 行开启）"

    def _apply(self, hit, stack: list, pos: int) -> int:
        text, mk, idx = hit
        line = self.line_of(pos)

        if idx is None:                               # 进入：压栈
            child = Node(mk.language, line)
            stack[-1].parts.append(child)
            stack.append(child)
            return pos + len(text)

        if idx == 0:                                  # root 的退出标记：无对应开启
            self.errors.append(ParseError(
                line, "多余闭合标记",
                f"退出标记 {text!r}（语言 {mk.language!r}）没有对应的开启标记，已忽略"))
            return pos + len(text)

        top = len(stack) - 1
        if idx != top:                                # 违反"后开先关"：报错并自动闭合内层
            inner = "、".join(f"{nd.language!r}(第{nd.start_line}行)" for nd in stack[idx + 1:])
            self.errors.append(ParseError(
                line, "闭合顺序错误",
                f"退出标记 {text!r} 要闭合第 {stack[idx].start_line} 行开启的 {mk.language!r}，"
                f"但内层语言 {inner} 尚未关闭；已自动闭合内层"))
            for nd in stack[idx + 1:]:
                self.errors.append(ParseError(
                    nd.start_line, "语言未闭合",
                    f"语言 {nd.language!r} 自第 {nd.start_line} 行进入后未正常退出（被外层闭合截断）"))

        stack[idx].end_line = line
        del stack[idx:]                               # 弹出被闭合的语言（含乱序时的内层）
        return pos + len(text)

    def _match_broken_prefix(self, stack: list, pos: int):
        """完整标记均未命中时，若某标记的前缀命中，则视为语法错误的切换标记。"""
        markers = []
        for node in stack:
            markers.extend(self.langs[node.language].exit_markers)
        for lang in self.langs.values():
            markers.extend(lang.enter_markers)
        for mk in markers:
            if mk.prefix and self.text.startswith(mk.prefix, pos):
                return mk
        return None


# ---------------------------------------------------------------- 输出

def render_tree(node: Node, indent: int = 0) -> list[str]:
    pad = "  " * indent
    end = f"-{node.end_line}" if node.end_line else ""
    lines = [f"{pad}[{node.language}] 行 {node.start_line}{end}"]
    for part in node.parts:
        if isinstance(part, Node):
            lines.extend(render_tree(part, indent + 1))
        else:
            text = " ".join(part.split())
            if len(text) > 60:
                text = text[:57] + "..."
            if text:
                lines.append(f"{pad}  文本: {text!r}")
    return lines


def node_to_dict(node: Node) -> dict:
    return {
        "language": node.language,
        "start_line": node.start_line,
        "end_line": node.end_line,
        "parts": [node_to_dict(p) if isinstance(p, Node) else p for p in node.parts],
    }


# ---------------------------------------------------------------- 内置示例

DEFAULT_CONFIG = {
    "root": "markup",
    "languages": {
        "markup": {
            "enter": [
                {"pattern": r"\{%\s*markup\s*%\}", "prefix": "{%",
                 "label": "{% markup %}（脚本中嵌回标记）"},
            ],
            "exit": [
                {"pattern": r"\{%\s*endmarkup\s*%\}", "prefix": "{%",
                 "label": "{% endmarkup %}"},
            ],
            "strings": [["\"", "\"", "\\"], ["'", "'", "\\"]],
            "comments": [["<!--", "-->"]],
        },
        "script": {
            "enter": [{"pattern": r"<%", "label": "<%（进入脚本）"}],
            "exit": [{"pattern": r"%>", "label": "%>（退出脚本）"}],
            "strings": [["\"", "\"", "\\"], ["'", "'", "\\"]],
            "comments": [["//", "\n"], ["/*", "*/"]],
        },
    },
}

DEMO_TEXT = """<!DOCTYPE html>
<html>
<body>
<%
  var s = "字符串里的 %> 不算切换";
  // 行注释里的 %> 也不算
  /* {% markup %} 在块注释里同样豁免 */
%>
{% markup %}
<p>脚本中嵌回的标记</p>
<% var nested = 1; %>
{% endmarkup %}
</body>
</html>
"""


# ---------------------------------------------------------------- 入口

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="多语言混合文本解析器（纯标准库单文件）")
    ap.add_argument("file", nargs="?", help="待解析的源文件（缺省读标准输入）")
    ap.add_argument("-c", "--config", help="语言切换标记定义（JSON），缺省用内置示例配置")
    ap.add_argument("--text", help="直接给定待解析文本")
    ap.add_argument("--demo", action="store_true", help="运行内置示例")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结构树与错误")
    args = ap.parse_args(argv)

    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            config = json.load(fh)
    else:
        config = DEFAULT_CONFIG

    if args.demo:
        text = DEMO_TEXT
    elif args.text is not None:
        text = args.text
    elif args.file:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()

    try:
        parser = Parser(config)
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2

    root, errors = parser.parse(text)
    errors.sort(key=lambda e: e.line)

    if args.json:
        print(json.dumps({
            "tree": node_to_dict(root),
            "errors": [{"line": e.line, "kind": e.kind, "message": e.message} for e in errors],
        }, ensure_ascii=False, indent=2))
    else:
        print("== 结构树 ==")
        print("\n".join(render_tree(root)))
        print("\n== 错误报告 ==")
        if errors:
            for err in errors:
                print(f"  {err}")
        else:
            print("  （无错误）")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
