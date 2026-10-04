"""Write-ahead log: encoding, crash recovery, torn-tail truncation."""

from __future__ import annotations

import os
import tempfile
import unittest

from kvstore.errors import ParseError, ValidationError
from kvstore.wal import WriteAheadLog, decode_record, encode_record, replay


class EncodingTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        payload = {"op": "put", "key": "a", "seq": 1, "value": "1"}
        self.assertEqual(decode_record(encode_record(payload)), payload)

    def test_checksum_mismatch_is_reported(self) -> None:
        line = encode_record({"op": "put", "key": "a", "seq": 1, "value": "1"})
        tampered = "00000000" + line[8:]
        with self.assertRaises(ParseError):
            decode_record(tampered)

    def test_unknown_op_is_rejected(self) -> None:
        with self.assertRaises(ParseError):
            decode_record(encode_record({"op": "merge", "key": "a", "seq": 1}))

    def test_missing_value_for_put_is_rejected(self) -> None:
        with self.assertRaises(ParseError):
            decode_record(encode_record({"op": "put", "key": "a", "seq": 1}))

    def test_replay_folds_records(self) -> None:
        state = replay(
            [
                {"op": "put", "key": "a", "seq": 1, "value": "1"},
                {"op": "put", "key": "b", "seq": 2, "value": "2"},
                {"op": "del", "key": "a", "seq": 3},
            ]
        )
        self.assertEqual(state, {"a": None, "b": "2"})


class RecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "wal.log")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_records_survive_reopen(self) -> None:
        with WriteAheadLog(self.path) as log:
            log.append("put", "a", "1")
            log.append("put", "b", "2")
        report = WriteAheadLog(self.path).recover()
        self.assertEqual(len(report.records), 2)
        self.assertEqual(report.truncated_bytes, 0)

    def test_partial_tail_is_truncated_and_reported(self) -> None:
        with WriteAheadLog(self.path) as log:
            log.append("put", "a", "1")
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write("deadbeef {\"op\":\"put\",\"key\":\"b\",\"seq\":2,\"va")
        report = WriteAheadLog(self.path).recover()
        self.assertEqual([record["key"] for record in report.records], ["a"])
        self.assertGreater(report.truncated_bytes, 0)
        # the repair is on disk: a second recovery sees a clean log
        again = WriteAheadLog(self.path).recover()
        self.assertEqual(again.truncated_bytes, 0)

    def test_sequence_continues_after_recovery(self) -> None:
        with WriteAheadLog(self.path) as log:
            log.append("put", "a", "1")
        with WriteAheadLog(self.path) as log:
            self.assertEqual(log.recover().last_sequence, 1)
            self.assertEqual(log.append("put", "b", "2"), 2)

    def test_append_requires_a_value_for_put(self) -> None:
        with WriteAheadLog(self.path) as log:
            with self.assertRaises(ValidationError):
                log.append("put", "a")

    def test_reset_clears_the_log(self) -> None:
        with WriteAheadLog(self.path) as log:
            log.append("put", "a", "1")
            log.reset()
            self.assertEqual(log.recover().records, [])
            self.assertEqual(log.append("put", "b", "2"), 1)


if __name__ == "__main__":
    unittest.main()
