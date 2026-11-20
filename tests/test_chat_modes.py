from __future__ import annotations

import unittest

from app.chat_modes import FAULT_TRACE, fault_trace_followups, get_mode


class FaultTraceModeTests(unittest.TestCase):
    def test_fault_trace_policy_controls_retrieval_profile(self) -> None:
        mode = get_mode("fault_trace")
        self.assertEqual(mode, FAULT_TRACE)
        self.assertGreaterEqual(mode.min_top_k, 15)
        self.assertEqual(mode.min_hops, 3)
        self.assertTrue(mode.force_expand)
        self.assertTrue(mode.follow_references)

    def test_followups_include_starting_block_and_cross_sheet_target(self) -> None:
        context = """
        4900500 Perform figure 2-1, Preparation A, then go to figure 2-4,
        block 1.
        4900040 Perform figure 2-1, then go to figure 2-9, block 1.
        [diagram edge E6]
        From: 4900500 initial relay test
        Relationship: flows_to
        To: TO SHT 3
        """

        queries = fault_trace_followups(
            "Trace maintenance actions for fault code 4900500", context
        )

        self.assertIn("4900500 Figure 2-4 Block 1", queries)
        self.assertIn("Figure 2-4 Sheet 3", queries)
        self.assertNotIn("Figure 2-1 Sheet 3", queries)
        self.assertFalse(any("2-9" in query for query in queries))


if __name__ == "__main__":
    unittest.main()
