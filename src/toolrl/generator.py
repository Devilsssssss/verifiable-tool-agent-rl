from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

from .schemas import AgentTask

FAMILIES = (
    "cancel_processing",
    "cancel_shipped",
    "exchange_cheaper",
    "return_eligible",
    "return_expired",
)

SPLIT_PREFIX = {"train": "Cedar", "validation": "Harbor", "test": "Summit"}
SPLIT_ID_OFFSET = {"train": 0, "validation": 1_000_000, "test": 2_000_000}

PROMPT_TEMPLATES = {
    "train": {
        "cancel_processing": "I am {email}. Cancel my order containing {old_name} and refund it.",
        "cancel_shipped": "I am {email}. Please cancel the order with {old_name} and refund me.",
        "exchange_cheaper": (
            "I am {email}. Replace the {old_name} in my order with {new_name}; refund any difference."
        ),
        "return_eligible": "I am {email}. Return the delivered {old_name} and refund it.",
        "return_expired": "I am {email}. Return the delivered {old_name} and refund it.",
    },
    "validation": {
        "cancel_processing": "For account {email}, cancel and refund the purchase containing {old_name}.",
        "cancel_shipped": "For account {email}, stop the {old_name} shipment and refund the payment.",
        "exchange_cheaper": (
            "For account {email}, swap {old_name} for {new_name} and send back the price difference."
        ),
        "return_eligible": "For account {email}, arrange a return and refund for {old_name}.",
        "return_expired": "For account {email}, arrange a return and refund for {old_name}.",
    },
    "test": {
        "cancel_processing": "Using {email}, I need the {old_name} purchase cancelled with a full refund.",
        "cancel_shipped": "Using {email}, cancel my shipped {old_name} order and reimburse me.",
        "exchange_cheaper": (
            "Using {email}, exchange the ordered {old_name} for {new_name} and settle the difference."
        ),
        "return_eligible": "Using {email}, I want to send back {old_name} for a refund.",
        "return_expired": "Using {email}, I want to send back {old_name} for a refund.",
    },
}


def generate_tasks(
    split_counts: dict[str, int] | None = None,
    *,
    seed: int = 42,
) -> list[AgentTask]:
    counts = split_counts or {"train": 300, "validation": 60, "test": 120}
    unknown = set(counts) - set(SPLIT_PREFIX)
    if unknown:
        raise ValueError(f"unsupported splits: {sorted(unknown)}")
    tasks = []
    for split_index, (split, count) in enumerate(counts.items()):
        rng = random.Random(seed + split_index * 10_000)
        for index in range(count):
            family = FAMILIES[index % len(FAMILIES)]
            tasks.append(_make_task(split, index, family, rng))
    return tasks


def _make_task(split: str, index: int, family: str, rng: random.Random) -> AgentTask:
    prefix = SPLIT_PREFIX[split]
    id_offset = SPLIT_ID_OFFSET[split]
    user_id = id_offset + 10_000 + index
    target_order_id = id_offset + 20_000 + index * 10
    distractor_order_id = target_order_id + 1
    old_product_id = id_offset + 30_000 + index * 10
    new_product_id = old_product_id + 1
    distractor_product_id = old_product_id + 2
    quantity = 1 + index % 2
    old_price = rng.choice((10_900, 12_900, 14_900, 16_900))
    new_price = old_price - rng.choice((1_000, 2_000, 3_000))
    payment_method = rng.choice(("alipay", "wechat_pay", "visa_ending_4242"))
    email = f"{split}.{index:04d}@example.test"
    old_name = f"{prefix} Headphones {index:04d}"
    new_name = f"{prefix} Earbuds {index:04d}"
    distractor_name = f"{prefix} Charger {index:04d}"
    status, delivered_days = _target_status(family)
    target_total = old_price * quantity

    initial_state: dict[str, list[dict[str, Any]]] = {
        "users": [{"id": user_id, "name": f"{prefix} User {index:04d}", "email": email}],
        "products": [
            {
                "id": old_product_id,
                "name": old_name,
                "price_cents": old_price,
                "stock": rng.randint(2, 8),
            },
            {
                "id": new_product_id,
                "name": new_name,
                "price_cents": new_price,
                "stock": quantity + rng.randint(2, 8),
            },
            {
                "id": distractor_product_id,
                "name": distractor_name,
                "price_cents": 2_900,
                "stock": rng.randint(3, 12),
            },
        ],
        "orders": [
            {
                "id": target_order_id,
                "user_id": user_id,
                "status": status,
                "payment_method": payment_method,
                "total_cents": target_total,
                "days_since_delivery": delivered_days,
                "pending_refund_cents": 0,
            },
            {
                "id": distractor_order_id,
                "user_id": user_id,
                "status": "delivered",
                "payment_method": "alipay",
                "total_cents": 2_900,
                "days_since_delivery": 8,
                "pending_refund_cents": 0,
            },
        ],
        "order_items": [
            {
                "order_id": target_order_id,
                "product_id": old_product_id,
                "quantity": quantity,
                "unit_price_cents": old_price,
            },
            {
                "order_id": distractor_order_id,
                "product_id": distractor_product_id,
                "quantity": 1,
                "unit_price_cents": 2_900,
            },
        ],
        "refunds": [],
        "returns": [],
        "escalations": [],
    }
    prompt = PROMPT_TEMPLATES[split][family].format(
        email=email, old_name=old_name, new_name=new_name
    )
    goal, plan = _goal_and_plan(
        family=family,
        email=email,
        user_id=user_id,
        order_id=target_order_id,
        old_product_id=old_product_id,
        new_product_id=new_product_id,
        new_product_name=new_name,
        quantity=quantity,
        old_price=old_price,
        new_price=new_price,
        payment_method=payment_method,
    )
    return AgentTask(
        task_id=f"{split}-{family}-{index:04d}",
        family=family,
        split=split,
        prompt=prompt,
        initial_state=initial_state,
        goal=goal,
        expert_plan=tuple(plan),
        max_steps=max(8, len(plan) + 2),
        optimal_steps=len(plan),
        template_id=f"{split}-{family}-v1",
    )


def _target_status(family: str) -> tuple[str, int | None]:
    if family in {"cancel_processing", "exchange_cheaper"}:
        return "processing", None
    if family == "cancel_shipped":
        return "shipped", None
    if family == "return_eligible":
        return "delivered", 12
    if family == "return_expired":
        return "delivered", 45
    raise ValueError(f"unsupported family: {family}")


def _goal_and_plan(
    *,
    family: str,
    email: str,
    user_id: int,
    order_id: int,
    old_product_id: int,
    new_product_id: int,
    new_product_name: str,
    quantity: int,
    old_price: int,
    new_price: int,
    payment_method: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inspect = [
        _step("lookup_user", email=email),
        _step("list_orders", user_id=user_id),
        _step("get_order", order_id=order_id),
    ]
    if family == "cancel_processing":
        refund = old_price * quantity
        return (
            {
                "kind": "cancel_refund",
                "order_id": order_id,
                "refund_cents": refund,
                "payment_method": payment_method,
            },
            inspect
            + [
                _step("cancel_order", order_id=order_id),
                _step(
                    "issue_refund",
                    order_id=order_id,
                    amount_cents=refund,
                    payment_method=payment_method,
                ),
            ],
        )
    if family == "exchange_cheaper":
        refund = (old_price - new_price) * quantity
        return (
            {
                "kind": "exchange_refund",
                "order_id": order_id,
                "new_product_id": new_product_id,
                "quantity": quantity,
                "expected_total_cents": new_price * quantity,
                "refund_cents": refund,
                "payment_method": payment_method,
            },
            inspect
            + [
                _step("search_products", query=new_product_name),
                _step("check_inventory", product_id=new_product_id, quantity=quantity),
                _step(
                    "replace_order_item",
                    order_id=order_id,
                    old_product_id=old_product_id,
                    new_product_id=new_product_id,
                    quantity=quantity,
                ),
                _step(
                    "issue_refund",
                    order_id=order_id,
                    amount_cents=refund,
                    payment_method=payment_method,
                ),
            ],
        )
    if family == "return_eligible":
        refund = old_price * quantity
        return (
            {
                "kind": "return_refund",
                "order_id": order_id,
                "product_id": old_product_id,
                "quantity": quantity,
                "refund_cents": refund,
                "payment_method": payment_method,
            },
            inspect
            + [
                _step(
                    "request_return",
                    order_id=order_id,
                    product_id=old_product_id,
                    quantity=quantity,
                ),
                _step(
                    "issue_refund",
                    order_id=order_id,
                    amount_cents=refund,
                    payment_method=payment_method,
                ),
            ],
        )
    if family in {"cancel_shipped", "return_expired"}:
        return (
            {"kind": "escalate_unchanged", "order_id": order_id},
            inspect + [_step("escalate_to_human", reason=_escalation_reason(family))],
        )
    raise ValueError(f"unsupported family: {family}")


def _step(tool: str, **arguments: Any) -> dict[str, Any]:
    return {"tool": tool, "arguments": arguments}


def _escalation_reason(family: str) -> str:
    if family == "cancel_shipped":
        return "The order has shipped and cannot be cancelled with the available tools."
    return "The return window has expired, so the request requires human review."


def entity_tokens(tasks: Iterable[AgentTask]) -> set[str]:
    tokens = set()
    for task in tasks:
        tokens.update(user["email"] for user in task.initial_state["users"])
        tokens.update(product["name"] for product in task.initial_state["products"])
        tokens.update(f"user-id:{user['id']}" for user in task.initial_state["users"])
        tokens.update(f"product-id:{product['id']}" for product in task.initial_state["products"])
        tokens.update(f"order-id:{order['id']}" for order in task.initial_state["orders"])
    return tokens
