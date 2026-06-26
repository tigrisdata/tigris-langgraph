"""The agent under test, plus the realistic prod state we'll evaluate against.

The agent is a one-node support assistant. The *only* thing that varies between
the baseline and the candidate is the system prompt — everything else (the
graph, the model, the accumulated thread history) is held fixed, so any
difference in the verdict is attributable to the change you're testing.

We never persist the system prompt into the thread; it's prepended at call time.
That's what lets us replay the exact same real conversation under two different
prompts and compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain.chat_models import init_chat_model
from langgraph.graph import START, MessagesState, StateGraph

MODEL = "claude-haiku-4-5-20251001"

# --- The change under test ----------------------------------------------------
# Baseline: the generic prompt most agents ship with. It answers fine, but does
# nothing to make the model lean on what it already knows about the customer.
BASELINE_PROMPT = (
    "You are a helpful customer support assistant. Answer the user's question."
)

# Candidate: explicitly tells the model to use the history it already has. This
# is the kind of one-line prompt tweak that looks identical on toy fixtures and
# only proves itself against real, memory-rich threads.
CANDIDATE_PROMPT = (
    "You are a helpful customer support assistant. Before answering, recall "
    "everything you already know about THIS customer from the conversation so "
    "far — their plan, preferences, constraints, and past issues — and tailor "
    "your answer to it specifically. Never ask them to repeat something they've "
    "already told you. Keep it concise."
)


def build_agent(system_prompt: str, temperature: float = 0.2) -> StateGraph:
    """Return an *uncompiled* graph. Callers compile it against a checkpointer.

    Low temperature: we want to measure the prompt's effect, not sampling noise.
    """
    model = init_chat_model(MODEL, temperature=temperature)

    def call_model(state: MessagesState) -> dict:
        messages = [{"role": "system", "content": system_prompt}, *state["messages"]]
        return {"messages": model.invoke(messages)}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    return builder


# --- The "weeks of accumulated state" we're evaluating against ----------------
@dataclass
class Thread:
    """One real customer conversation: the memory we seed, the held-out probe we
    evaluate on, and the keywords that signal the agent actually used the memory.
    """

    thread_id: str
    # Prior turns that establish customer-specific facts (the "memory").
    history: list[str]
    # The held-out question we score both variants on.
    probe: str
    # Lowercased substrings; presence in a reply means it used the known context.
    recall_markers: list[str] = field(default_factory=list)


# Each thread carries a fact in its history that a good answer to the probe
# *must* use. Fixtures wouldn't have this memory, so the candidate's win would be
# invisible. That's the whole point.
THREADS: list[Thread] = [
    Thread(
        thread_id="cust-ana",
        history=[
            "Hi, I'm Ana. Quick note for my account: I'm vegetarian and I have a "
            "severe tree-nut allergy, so please keep that in mind for any meal-kit "
            "recommendations.",
        ],
        probe="Can you suggest a dinner from this week's meal-kit menu for me?",
        recall_markers=["vegetarian", "nut"],
    ),
    Thread(
        thread_id="cust-ben",
        history=[
            "Hey, it's Ben. I'm on the Pro plan and I'm based in Tokyo (JST). "
            "Wanted that on record for support timing.",
        ],
        probe="If I open a ticket tonight, when can I expect a first response?",
        recall_markers=["pro"],
    ),
    Thread(
        thread_id="cust-cleo",
        history=[
            "Hi! I'm Cleo and I'll be honest — I'm a total beginner with this kind "
            "of software, so simple, jargon-free explanations help me a lot.",
        ],
        probe="How do I set up automated backups?",
        recall_markers=["step"],
    ),
    Thread(
        thread_id="cust-dan",
        history=[
            "This is Dan. Last month I was double-charged and your team fixed it "
            "with a one-time credit — appreciated. Just flagging the history.",
        ],
        probe="I'm seeing another charge I don't recognize. What now?",
        recall_markers=["credit"],
    ),
]
