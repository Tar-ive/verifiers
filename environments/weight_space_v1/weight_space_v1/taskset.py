import json
import random
import re

import verifiers.v1 as vf

from weight_space_v1.nets import (FUNCTIONS, INPUTS, Net, accuracy, apply_edits, generate, net_from_answer,
                                  render_prompt, target_table)

MODES = ("read", "fix", "write")

SYSTEM_PROMPT = """\
You read and write neural network weights directly, as text. You cannot run code or the network: reason about what
each unit computes from its weights. Think as needed, then finish with exactly one block:
<answer>
{ ...JSON in the format the task asks for... }
</answer>"""


def parse_answer(text: str) -> dict | None:
    # The last non-empty <answer> block, found from the end so that an opener mentioned
    # in the reasoning ("Now the <answer> block:") cannot swallow the real block.
    text = text or ""
    start = len(text)
    while (start := text.rfind("<answer>", 0, start)) >= 0:
        close = text.find("</answer>", start + len("<answer>"))
        if close < 0:
            continue
        body = text[start + len("<answer>"):close].strip()
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body)
        if not body:
            continue
        try:
            parsed = json.loads(body)
        except (ValueError, RecursionError):  # JSONDecodeError, the int digit limit, deep nesting
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def unit_prf(predicted, gold: list[int]) -> tuple[float, float, float]:
    try:
        pred = {int(u) for u in predicted} if isinstance(predicted, list) else set()
    except (TypeError, ValueError):
        pred = set()
    gold = set(gold)
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    hits = len(pred & gold)
    p = hits / len(pred) if pred else 0.0
    r = hits / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def lift(acc: float, base: float) -> float:
    """Accuracy rescaled so `base` scores 0 and a perfect network scores 1."""
    return max(0.0, (acc - base) / (1 - base)) if base < 1 else float(acc == 1)


def baseline(mode: str, function: str, net: Net | None) -> float:
    """Accuracy of doing nothing: the best constant output, or in fix mode the corrupted net as
    shown if that is better. (With the corrupted net alone, an edit that silences the network
    was paid the lift wherever the corruption had dropped the net below a constant output.)"""
    ones = sum(target_table(function))
    const = max(ones, len(INPUTS) - ones) / len(INPUTS)
    if mode == "fix" and net is not None:
        return max(float(accuracy(net, function)), const)
    return const


class WeightSpaceData(vf.TaskData):
    seed: int
    mode: str
    function: str
    net: dict | None
    """The network shown to the model (read and fix tasks)."""
    expected: dict


class WeightSpaceTask(vf.Task[WeightSpaceData]):
    def _answer(self, trace: vf.Trace) -> dict | None:
        return parse_answer(trace.last_reply)

    def _result_net(self, answer: dict | None) -> Net | None:
        """The network the answer produces (fix: the shown net with the edits applied; write: the written net)."""
        if answer is None:
            return None
        if self.data.mode == "fix":
            return apply_edits(Net.from_json(self.data.net), answer.get("edits"))
        if self.data.mode == "write":
            return net_from_answer(answer)
        return None

    def _accuracy(self, trace: vf.Trace) -> float:
        net = self._result_net(self._answer(trace))
        return float(accuracy(net, self.data.function)) if net else 0.0

    def _baseline(self) -> float:
        return baseline(self.data.mode, self.data.function, Net.from_json(self.data.net) if self.data.net else None)

    @vf.reward(weight=1.0)
    async def weight_score(self, trace: vf.Trace) -> float:
        """read: 0.6 x right function + 0.4 x functional-unit F1. fix / write: accuracy lift over doing nothing."""
        answer = self._answer(trace)
        if self.data.mode == "read":
            if answer is None:
                return 0.0
            right = answer.get("function") == self.data.function
            return 0.6 * right + 0.4 * unit_prf(answer.get("functional_units"), self.data.expected["functional_units"])[2]
        return lift(self._accuracy(trace), self._baseline())

    @vf.metric
    async def answer_parsed(self, trace: vf.Trace) -> float:
        return float(self._answer(trace) is not None)

    @vf.metric
    async def solved(self, trace: vf.Trace) -> float:
        """read: right function and exact unit set. fix / write: right on all 32 inputs."""
        answer = self._answer(trace)
        if answer is None:
            return 0.0
        if self.data.mode == "read":
            return float(answer.get("function") == self.data.function
                         and unit_prf(answer.get("functional_units"), self.data.expected["functional_units"])[2] == 1)
        return float(self._accuracy(trace) == 1.0)

    @vf.metric
    async def function_correct(self, trace: vf.Trace) -> float:
        answer = self._answer(trace)
        return float(self.data.mode == "read" and answer is not None and answer.get("function") == self.data.function)

    @vf.metric
    async def unit_f1(self, trace: vf.Trace) -> float:
        answer = self._answer(trace)
        if self.data.mode != "read" or answer is None:
            return 0.0
        return unit_prf(answer.get("functional_units"), self.data.expected["functional_units"])[2]

    @vf.metric
    async def accuracy_32(self, trace: vf.Trace) -> float:
        """Fraction of the 32 inputs the resulting network gets right (fix / write)."""
        return self._accuracy(trace) if self.data.mode != "read" else 0.0

    @vf.metric
    async def found_corruption(self, trace: vf.Trace) -> float:
        """fix: whether an edit touched the corrupted parameter."""
        answer = self._answer(trace)
        if self.data.mode != "fix" or answer is None or not isinstance(answer.get("edits"), list):
            return 0.0
        loc = self.data.expected["corruption"]
        for e in answer["edits"]:
            if isinstance(e, dict) and e.get("param") == loc["param"] and str(e.get("row")) == str(loc.get("row")) \
                    and str(e.get("col")) == str(loc.get("col")):
                return 1.0
        return 0.0


class WeightSpaceConfig(vf.TasksetConfig):
    num_tasks: int = 48
    """How many tasks to build; modes cycle read, fix, write."""
    hidden: int = 16
    """Hidden units in the networks shown for read and fix tasks."""
    seed: int = 20261103
    """Base seed; task i uses seed + i."""


class WeightSpaceTaskset(vf.Taskset[WeightSpaceTask, WeightSpaceConfig]):
    def load(self) -> list[WeightSpaceTask]:
        names = list(FUNCTIONS)
        orders = {m: random.Random(self.config.seed + k).sample(names, len(names)) for k, m in enumerate(MODES)}
        tasks = []
        for i in range(self.config.num_tasks):
            seed, mode = self.config.seed + i, MODES[i % 3]
            function = orders[mode][(i // 3) % len(names)]
            spec = generate(seed, mode, function, self.config.hidden)
            data = WeightSpaceData(
                id=f"wsp-{mode}-{seed}",
                name=f"{mode}-{function}-{seed}",
                system_prompt=SYSTEM_PROMPT,
                prompt=render_prompt(spec),
                seed=seed,
                mode=mode,
                function=function,
                net=spec.net.to_json() if spec.net else None,
                expected=spec.expected,
            )
            tasks.append(WeightSpaceTask(data, self.config.task))
        return tasks
