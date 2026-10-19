from __future__ import annotations

import inspect
import json
import sqlite3
from copy import deepcopy
from functools import wraps
from typing import Any

SCORING_PROTOCOL = "strict-v2-target-inspection"
REASON_CODE_SCORING_PROTOCOL = "strict-v3-reason-code"
SUPPORTED_SCORING_PROTOCOLS = {SCORING_PROTOCOL, REASON_CODE_SCORING_PROTOCOL}


def _drop_null_tool_slots(method):
    """Accept schema-generated null slots while keeping non-null calls strict.

    Transformers' tool-call loop invokes the bound environment methods directly.
    Some chat templates emit every field from a shared schema and fill unused
    fields with ``null``.  The original method signatures remain exposed to
    schema generation via ``__signature__``; only the runtime wrapper removes
    null values before dispatching to the strict implementation.
    """

    @wraps(method)
    def wrapped(self, *args: Any, **kwargs: Any) -> str:
        return method(
            self, *args, **{key: value for key, value in kwargs.items() if value is not None}
        )

    wrapped.__signature__ = inspect.signature(method)  # type: ignore[attr-defined]
    return wrapped


class RetailToolEnvironment:
    """Stateful retail environment with executable policy checks and rewards."""

    def __init__(self) -> None:
        self._connection: sqlite3.Connection | None = None
        self._goal: dict[str, Any] = {}
        self._trace: list[dict[str, Any]] = []
        self._initial_business_state: dict[str, list[dict[str, Any]]] = {}
        self._max_steps = 10
        self._optimal_steps = 1
        self._reward_mode = "shaped"
        self._observed_user_ids: set[int] = set()
        self._observed_order_ids: set[int] = set()
        self._inspected_order_ids: set[int] = set()
        self._inventory_checks: set[tuple[int, int]] = set()

    def reset(
        self,
        initial_state_json: str,
        goal_json: str,
        max_steps: int,
        optimal_steps: int,
        reward_mode: str = "shaped",
        scoring_protocol: str = SCORING_PROTOCOL,
        **_: Any,
    ) -> None:
        if reward_mode not in {"sparse", "shaped"}:
            raise ValueError(f"unsupported reward mode: {reward_mode}")
        if scoring_protocol not in SUPPORTED_SCORING_PROTOCOLS:
            raise ValueError(f"unsupported scoring protocol: {scoring_protocol}")
        if self._connection is not None:
            self._connection.close()
        self._connection = sqlite3.connect(":memory:")
        self._connection.row_factory = sqlite3.Row
        self._create_schema()
        initial_state = json.loads(initial_state_json)
        self._load_state(initial_state)
        self._goal = json.loads(goal_json)
        self._trace = []
        self._max_steps = int(max_steps)
        self._optimal_steps = int(optimal_steps)
        self._reward_mode = reward_mode
        self._scoring_protocol = scoring_protocol
        self._observed_user_ids = set()
        self._observed_order_ids = set()
        self._inspected_order_ids = set()
        self._inventory_checks = set()
        self._initial_business_state = self._business_state()

    @_drop_null_tool_slots
    def lookup_user(self, email: str) -> str:
        """Look up a customer before accessing their orders.

        Args:
            email: Exact customer email from the request.

        Returns:
            A JSON object containing the matching customer, or an error.
        """
        args = {"email": email}
        if error := self._preflight("lookup_user", args):
            return error
        row = self._db.execute(
            "SELECT id, name, email FROM users WHERE lower(email) = lower(?)", (email,)
        ).fetchone()
        if row is None:
            return self._finish("lookup_user", args, False, {"error": "user_not_found"})
        result = dict(row)
        self._observed_user_ids.add(int(row["id"]))
        return self._finish("lookup_user", args, True, {"user": result})

    @_drop_null_tool_slots
    def list_orders(self, user_id: int) -> str:
        """List a previously looked-up customer's orders.

        Args:
            user_id: Customer identifier returned by lookup_user.

        Returns:
            A JSON list of the customer's orders.
        """
        args = {"user_id": user_id}
        if error := self._preflight("list_orders", args):
            return error
        if user_id not in self._observed_user_ids:
            return self._finish(
                "list_orders",
                args,
                False,
                {"error": "user_must_be_looked_up_first"},
                violation="unobserved_identifier",
            )
        rows = self._db.execute(
            """
            SELECT id, status, payment_method, total_cents, days_since_delivery
            FROM orders WHERE user_id = ? ORDER BY id
            """,
            (user_id,),
        ).fetchall()
        orders = [dict(row) for row in rows]
        self._observed_order_ids.update(int(row["id"]) for row in rows)
        return self._finish("list_orders", args, True, {"orders": orders})

    @_drop_null_tool_slots
    def get_order(self, order_id: int) -> str:
        """Inspect an order and its line items.

        Args:
            order_id: Order identifier returned by list_orders.

        Returns:
            A JSON object containing order and item details.
        """
        args = {"order_id": order_id}
        if error := self._preflight("get_order", args):
            return error
        if order_id not in self._observed_order_ids:
            return self._finish(
                "get_order",
                args,
                False,
                {"error": "order_must_be_listed_first"},
                violation="unobserved_identifier",
            )
        order = self._db.execute(
            """
            SELECT id, user_id, status, payment_method, total_cents,
                   days_since_delivery, pending_refund_cents
            FROM orders WHERE id = ?
            """,
            (order_id,),
        ).fetchone()
        if order is None:
            return self._finish("get_order", args, False, {"error": "order_not_found"})
        items = self._db.execute(
            """
            SELECT oi.product_id, p.name, oi.quantity, oi.unit_price_cents
            FROM order_items oi JOIN products p ON p.id = oi.product_id
            WHERE oi.order_id = ? ORDER BY oi.product_id
            """,
            (order_id,),
        ).fetchall()
        self._inspected_order_ids.add(order_id)
        return self._finish(
            "get_order",
            args,
            True,
            {"order": dict(order), "items": [dict(item) for item in items]},
        )

    @_drop_null_tool_slots
    def search_products(self, query: str) -> str:
        """Search products by name.

        Args:
            query: Product name or a distinctive substring.

        Returns:
            A JSON list of matching products and prices.
        """
        args = {"query": query}
        if error := self._preflight("search_products", args):
            return error
        rows = self._db.execute(
            """
            SELECT id, name, price_cents FROM products
            WHERE lower(name) LIKE ? ORDER BY id
            """,
            (f"%{query.lower()}%",),
        ).fetchall()
        return self._finish(
            "search_products",
            args,
            bool(rows),
            {"products": [dict(row) for row in rows]} if rows else {"error": "no_products"},
        )

    @_drop_null_tool_slots
    def check_inventory(self, product_id: int, quantity: int) -> str:
        """Check whether a product has enough stock.

        Args:
            product_id: Product identifier returned by search_products.
            quantity: Required quantity.

        Returns:
            A JSON object containing stock and availability.
        """
        args = {"product_id": product_id, "quantity": quantity}
        if error := self._preflight("check_inventory", args):
            return error
        row = self._db.execute("SELECT stock FROM products WHERE id = ?", (product_id,)).fetchone()
        if row is None or quantity <= 0:
            return self._finish(
                "check_inventory", args, False, {"error": "invalid_product_or_quantity"}
            )
        self._inventory_checks.add((product_id, quantity))
        stock = int(row["stock"])
        return self._finish(
            "check_inventory",
            args,
            True,
            {"product_id": product_id, "stock": stock, "available": stock >= quantity},
        )

    @_drop_null_tool_slots
    def cancel_order(self, order_id: int) -> str:
        """Cancel a processing order and create a pending refund.

        Args:
            order_id: Inspected order identifier.

        Returns:
            A JSON object describing the cancelled order and pending refund.
        """
        args = {"order_id": order_id}
        if error := self._preflight("cancel_order", args):
            return error
        if order_id not in self._inspected_order_ids:
            return self._policy_error("cancel_order", args, "order_not_inspected")
        row = self._db.execute(
            "SELECT status, total_cents, pending_refund_cents FROM orders WHERE id = ?",
            (order_id,),
        ).fetchone()
        if row is None or row["status"] != "processing" or row["pending_refund_cents"] != 0:
            return self._policy_error("cancel_order", args, "order_not_cancellable")
        self._db.execute(
            "UPDATE orders SET status = 'cancelled', pending_refund_cents = ? WHERE id = ?",
            (int(row["total_cents"]), order_id),
        )
        self._db.commit()
        return self._finish(
            "cancel_order",
            args,
            True,
            {"status": "cancelled", "pending_refund_cents": int(row["total_cents"])},
            mutated=True,
        )

    @_drop_null_tool_slots
    def replace_order_item(
        self,
        order_id: int,
        old_product_id: int,
        new_product_id: int,
        quantity: int,
    ) -> str:
        """Replace a whole line item in a processing order.

        Args:
            order_id: Inspected order identifier.
            old_product_id: Existing product identifier from get_order.
            new_product_id: Replacement product identifier from search_products.
            quantity: Whole line-item quantity to replace.

        Returns:
            A JSON object with the new total and any pending refund.
        """
        args = {
            "order_id": order_id,
            "old_product_id": old_product_id,
            "new_product_id": new_product_id,
            "quantity": quantity,
        }
        if error := self._preflight("replace_order_item", args):
            return error
        if order_id not in self._inspected_order_ids:
            return self._policy_error("replace_order_item", args, "order_not_inspected")
        if (new_product_id, quantity) not in self._inventory_checks:
            return self._policy_error("replace_order_item", args, "inventory_not_checked")
        order = self._db.execute(
            "SELECT status, total_cents FROM orders WHERE id = ?", (order_id,)
        ).fetchone()
        old_item = self._db.execute(
            """
            SELECT quantity, unit_price_cents FROM order_items
            WHERE order_id = ? AND product_id = ?
            """,
            (order_id, old_product_id),
        ).fetchone()
        new_product = self._db.execute(
            "SELECT price_cents, stock FROM products WHERE id = ?", (new_product_id,)
        ).fetchone()
        if order is None or order["status"] != "processing":
            return self._policy_error("replace_order_item", args, "order_not_modifiable")
        if old_item is None or int(old_item["quantity"]) != quantity:
            return self._policy_error("replace_order_item", args, "whole_line_replacement_required")
        if new_product is None or int(new_product["stock"]) < quantity:
            return self._policy_error("replace_order_item", args, "insufficient_inventory")
        old_subtotal = int(old_item["unit_price_cents"]) * quantity
        new_subtotal = int(new_product["price_cents"]) * quantity
        if new_subtotal > old_subtotal:
            return self._policy_error("replace_order_item", args, "additional_payment_unsupported")
        refund = old_subtotal - new_subtotal
        new_total = int(order["total_cents"]) - refund
        self._db.execute(
            """
            UPDATE order_items SET product_id = ?, unit_price_cents = ?
            WHERE order_id = ? AND product_id = ?
            """,
            (new_product_id, int(new_product["price_cents"]), order_id, old_product_id),
        )
        self._db.execute(
            "UPDATE products SET stock = stock - ? WHERE id = ?", (quantity, new_product_id)
        )
        self._db.execute(
            "UPDATE products SET stock = stock + ? WHERE id = ?", (quantity, old_product_id)
        )
        self._db.execute(
            """
            UPDATE orders SET total_cents = ?, pending_refund_cents = pending_refund_cents + ?
            WHERE id = ?
            """,
            (new_total, refund, order_id),
        )
        self._db.commit()
        return self._finish(
            "replace_order_item",
            args,
            True,
            {"total_cents": new_total, "pending_refund_cents": refund},
            mutated=True,
        )

    @_drop_null_tool_slots
    def request_return(self, order_id: int, product_id: int, quantity: int) -> str:
        """Request a return for an eligible delivered line item.

        Args:
            order_id: Inspected delivered order identifier.
            product_id: Product identifier from get_order.
            quantity: Whole line-item quantity to return.

        Returns:
            A JSON object with return status and pending refund.
        """
        args = {"order_id": order_id, "product_id": product_id, "quantity": quantity}
        if error := self._preflight("request_return", args):
            return error
        if order_id not in self._inspected_order_ids:
            return self._policy_error("request_return", args, "order_not_inspected")
        order = self._db.execute(
            "SELECT status, days_since_delivery FROM orders WHERE id = ?", (order_id,)
        ).fetchone()
        item = self._db.execute(
            """
            SELECT quantity, unit_price_cents FROM order_items
            WHERE order_id = ? AND product_id = ?
            """,
            (order_id, product_id),
        ).fetchone()
        existing = self._db.execute(
            "SELECT 1 FROM returns WHERE order_id = ? AND product_id = ?",
            (order_id, product_id),
        ).fetchone()
        if (
            order is None
            or order["status"] != "delivered"
            or order["days_since_delivery"] is None
            or int(order["days_since_delivery"]) > 30
        ):
            return self._policy_error("request_return", args, "return_window_closed")
        if item is None or int(item["quantity"]) != quantity or existing is not None:
            return self._policy_error("request_return", args, "invalid_return_item")
        refund = int(item["unit_price_cents"]) * quantity
        self._db.execute(
            "INSERT INTO returns(order_id, product_id, quantity, status) VALUES (?, ?, ?, 'approved')",
            (order_id, product_id, quantity),
        )
        self._db.execute(
            """
            UPDATE orders SET pending_refund_cents = pending_refund_cents + ? WHERE id = ?
            """,
            (refund, order_id),
        )
        self._db.commit()
        return self._finish(
            "request_return",
            args,
            True,
            {"return_status": "approved", "pending_refund_cents": refund},
            mutated=True,
        )

    @_drop_null_tool_slots
    def issue_refund(self, order_id: int, amount_cents: int, payment_method: str) -> str:
        """Issue the exact pending amount to the order's original payment method.

        Args:
            order_id: Inspected order identifier.
            amount_cents: Exact pending refund amount in integer cents.
            payment_method: Original payment method reported by get_order.

        Returns:
            A JSON object describing the completed refund.
        """
        args = {
            "order_id": order_id,
            "amount_cents": amount_cents,
            "payment_method": payment_method,
        }
        if error := self._preflight("issue_refund", args):
            return error
        if order_id not in self._inspected_order_ids:
            return self._policy_error("issue_refund", args, "order_not_inspected")
        row = self._db.execute(
            "SELECT payment_method, pending_refund_cents FROM orders WHERE id = ?", (order_id,)
        ).fetchone()
        if (
            row is None
            or int(row["pending_refund_cents"]) <= 0
            or amount_cents != int(row["pending_refund_cents"])
            or payment_method != row["payment_method"]
        ):
            return self._policy_error("issue_refund", args, "refund_must_match_pending_original")
        self._db.execute(
            "INSERT INTO refunds(order_id, amount_cents, payment_method) VALUES (?, ?, ?)",
            (order_id, amount_cents, payment_method),
        )
        self._db.execute("UPDATE orders SET pending_refund_cents = 0 WHERE id = ?", (order_id,))
        self._db.commit()
        return self._finish(
            "issue_refund",
            args,
            True,
            {"refund_status": "completed", "amount_cents": amount_cents},
            mutated=True,
        )

    @_drop_null_tool_slots
    def escalate_to_human(self, reason: str, reason_code: str | None = None) -> str:
        """Escalate a request that cannot be completed under policy.

        Args:
            reason: Concise explanation of the policy or capability limitation.
            reason_code: Optional machine-readable policy reason used by the strict
                reason-code scoring protocol.

        Returns:
            A JSON object confirming escalation.
        """
        args = {"reason": reason}
        if reason_code is not None:
            args["reason_code"] = reason_code
        if error := self._preflight("escalate_to_human", args):
            return error
        if not self._inspected_order_ids:
            return self._policy_error("escalate_to_human", args, "inspect_order_before_escalation")
        if self._scoring_protocol == REASON_CODE_SCORING_PROTOCOL:
            expected = self._expected_escalation_reason_code()
            if reason_code != expected:
                return self._policy_error(
                    "escalate_to_human", args, "escalation_reason_code_mismatch"
                )
        self._db.execute(
            "INSERT INTO escalations(reason, reason_code) VALUES (?, ?)",
            (reason, reason_code),
        )
        self._db.commit()
        return self._finish("escalate_to_human", args, True, {"status": "escalated"}, mutated=True)

    def get_reward(self) -> float:
        """Return the configured terminal reward for the completed rollout."""
        report = self.reward_report
        return float(report[self._reward_mode])

    @property
    def trace(self) -> list[dict[str, Any]]:
        return deepcopy(self._trace)

    @property
    def reward_report(self) -> dict[str, Any]:
        if self._connection is None:
            return {
                "success": False,
                "terminal_goal_satisfied": False,
                "target_order_inspected": False,
                "format_recovery_success": False,
                "policy_compliant": True,
                "violations": [],
                "steps": 0,
                "valid_tool_rate": 0.0,
                "efficient": False,
                "sparse": 0.0,
                "shaped": 0.0,
            }
        success = self._goal_satisfied()
        violations = [step["violation"] for step in self._trace if step["violation"]]
        terminal_goal = self._terminal_goal_satisfied()
        target_order_inspected = self._target_order_inspected()
        format_recovery = bool(
            terminal_goal
            and violations
            and all(violation == "invalid_tool_call" for violation in violations)
        )
        successful_calls = sum(bool(step["success"]) for step in self._trace)
        valid_rate = successful_calls / len(self._trace) if self._trace else 0.0
        compliant = not violations
        efficient = success and len(self._trace) <= self._optimal_steps + 1
        progress = self._correct_prefix_length() / max(self._optimal_steps, 1)
        sparse = 1.0 if success else 0.0
        shaped = (
            -0.2
            if not self._trace
            else (1.0 if success else -0.2)
            + (0.2 if compliant else -0.5)
            + 0.1 * valid_rate
            + (0.1 if efficient else 0.0)
            + (0.4 * progress if not success else 0.0)
        )
        return {
            "success": success,
            "terminal_goal_satisfied": terminal_goal,
            "target_order_inspected": target_order_inspected,
            "format_recovery_success": format_recovery,
            "policy_compliant": compliant,
            "violations": violations,
            "steps": len(self._trace),
            "valid_tool_rate": valid_rate,
            "efficient": efficient,
            "progress": progress,
            "sparse": sparse,
            "shaped": shaped,
        }

    @property
    def state_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        if self._connection is None:
            return {}
        return self._full_state()

    @property
    def _db(self) -> sqlite3.Connection | None:
        return self._connection

    def _create_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE
            );
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                price_cents INTEGER NOT NULL,
                stock INTEGER NOT NULL
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                payment_method TEXT NOT NULL,
                total_cents INTEGER NOT NULL,
                days_since_delivery INTEGER,
                pending_refund_cents INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE order_items (
                order_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                unit_price_cents INTEGER NOT NULL,
                PRIMARY KEY(order_id, product_id)
            );
            CREATE TABLE refunds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                amount_cents INTEGER NOT NULL,
                payment_method TEXT NOT NULL
            );
            CREATE TABLE returns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE escalations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reason TEXT NOT NULL,
                reason_code TEXT
            );
            """
        )

    def _load_state(self, state: dict[str, list[dict[str, Any]]]) -> None:
        table_columns = {
            "users": ("id", "name", "email"),
            "products": ("id", "name", "price_cents", "stock"),
            "orders": (
                "id",
                "user_id",
                "status",
                "payment_method",
                "total_cents",
                "days_since_delivery",
                "pending_refund_cents",
            ),
            "order_items": ("order_id", "product_id", "quantity", "unit_price_cents"),
            "refunds": ("id", "order_id", "amount_cents", "payment_method"),
            "returns": ("id", "order_id", "product_id", "quantity", "status"),
            "escalations": ("id", "reason", "reason_code"),
        }
        for table, columns in table_columns.items():
            rows = state.get(table, [])
            placeholders = ", ".join("?" for _ in columns)
            names = ", ".join(columns)
            for row in rows:
                self._db.execute(
                    f"INSERT INTO {table} ({names}) VALUES ({placeholders})",
                    tuple(row.get(column) for column in columns),
                )
        self._db.commit()

    def _preflight(self, name: str, args: dict[str, Any]) -> str | None:
        _ = self._db
        if len(self._trace) < self._max_steps:
            return None
        return self._finish(
            name,
            args,
            False,
            {"error": "tool_call_limit_exceeded"},
            violation="step_limit",
        )

    def _policy_error(self, name: str, args: dict[str, Any], code: str) -> str:
        return self._finish(name, args, False, {"error": code}, violation=code)

    def _record_external_failure(self, name: str, args: dict[str, Any], error: str) -> str:
        return self._finish(
            name,
            args,
            False,
            {"error": error},
            violation="invalid_tool_call",
        )

    def _finish(
        self,
        name: str,
        args: dict[str, Any],
        success: bool,
        result: dict[str, Any],
        *,
        violation: str | None = None,
        mutated: bool = False,
    ) -> str:
        self._trace.append(
            {
                "tool": name,
                "arguments": deepcopy(args),
                "success": success,
                "violation": violation,
                "mutated": mutated,
                "result": deepcopy(result),
            }
        )
        return json.dumps({"ok": success, **result}, sort_keys=True)

    def _goal_satisfied(self) -> bool:
        if any(step["violation"] for step in self._trace):
            return False
        return self._target_order_inspected() and self._terminal_goal_satisfied()

    def _target_order_inspected(self) -> bool:
        order_id = self._goal.get("order_id")
        return isinstance(order_id, int) and order_id in self._inspected_order_ids

    def _terminal_goal_satisfied(self) -> bool:
        """Check database outcome only; strict success still rejects all violations."""
        kind = self._goal.get("kind")
        order_id = int(self._goal.get("order_id", -1))
        if kind == "cancel_refund":
            order = self._db.execute(
                "SELECT status, pending_refund_cents FROM orders WHERE id = ?", (order_id,)
            ).fetchone()
            return bool(
                order
                and order["status"] == "cancelled"
                and order["pending_refund_cents"] == 0
                and self._refund_matches(order_id)
            )
        if kind == "exchange_refund":
            item = self._db.execute(
                "SELECT product_id, quantity FROM order_items WHERE order_id = ?", (order_id,)
            ).fetchone()
            order = self._db.execute(
                "SELECT total_cents, pending_refund_cents FROM orders WHERE id = ?", (order_id,)
            ).fetchone()
            return bool(
                item
                and order
                and item["product_id"] == self._goal["new_product_id"]
                and item["quantity"] == self._goal["quantity"]
                and order["total_cents"] == self._goal["expected_total_cents"]
                and order["pending_refund_cents"] == 0
                and self._refund_matches(order_id)
            )
        if kind == "return_refund":
            returned = self._db.execute(
                """
                SELECT quantity, status FROM returns
                WHERE order_id = ? AND product_id = ?
                """,
                (order_id, self._goal["product_id"]),
            ).fetchone()
            order = self._db.execute(
                "SELECT pending_refund_cents FROM orders WHERE id = ?", (order_id,)
            ).fetchone()
            return bool(
                returned
                and order
                and returned["quantity"] == self._goal["quantity"]
                and returned["status"] == "approved"
                and order["pending_refund_cents"] == 0
                and self._refund_matches(order_id)
            )
        if kind == "escalate_unchanged":
            escalations = self._db.execute("SELECT COUNT(*) AS count FROM escalations").fetchone()
            return bool(
                escalations
                and escalations["count"] == 1
                and self._business_state_without_escalations()
                == self._state_without_escalations(self._initial_business_state)
            )
        raise ValueError(f"unsupported goal kind: {kind}")

    def _expected_escalation_reason_code(self) -> str | None:
        order_id = self._goal.get("order_id")
        if not isinstance(order_id, int):
            return None
        order = self._db.execute(
            "SELECT status, days_since_delivery FROM orders WHERE id = ?", (order_id,)
        ).fetchone()
        if order is None:
            return None
        if order["status"] == "shipped":
            return "shipped_order_cannot_cancel"
        if (
            order["status"] == "delivered"
            and order["days_since_delivery"] is not None
            and int(order["days_since_delivery"]) > 30
        ):
            return "return_window_expired"
        return None

    def _correct_prefix_length(self) -> int:
        expected = {
            "cancel_refund": [
                "lookup_user",
                "list_orders",
                "get_order",
                "cancel_order",
                "issue_refund",
            ],
            "exchange_refund": [
                "lookup_user",
                "list_orders",
                "get_order",
                "search_products",
                "check_inventory",
                "replace_order_item",
                "issue_refund",
            ],
            "return_refund": [
                "lookup_user",
                "list_orders",
                "get_order",
                "request_return",
                "issue_refund",
            ],
            "escalate_unchanged": ["lookup_user", "list_orders", "get_order", "escalate_to_human"],
        }.get(self._goal.get("kind"), [])
        prefix = 0
        for step, expected_name in zip(self._trace, expected, strict=False):
            if not step["success"] or step["tool"] != expected_name:
                break
            prefix += 1
        return prefix

    def _refund_matches(self, order_id: int) -> bool:
        rows = self._db.execute(
            "SELECT amount_cents, payment_method FROM refunds WHERE order_id = ?", (order_id,)
        ).fetchall()
        return len(rows) == 1 and dict(rows[0]) == {
            "amount_cents": self._goal["refund_cents"],
            "payment_method": self._goal["payment_method"],
        }

    def _full_state(self) -> dict[str, list[dict[str, Any]]]:
        return {
            table: [dict(row) for row in self._db.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in (
                "users",
                "products",
                "orders",
                "order_items",
                "refunds",
                "returns",
                "escalations",
            )
        }

    def _business_state(self) -> dict[str, list[dict[str, Any]]]:
        state = self._full_state()
        state.pop("users")
        return state

    def _business_state_without_escalations(self) -> dict[str, list[dict[str, Any]]]:
        return self._state_without_escalations(self._business_state())

    @staticmethod
    def _state_without_escalations(
        state: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        result = deepcopy(state)
        result.pop("escalations", None)
        return result
