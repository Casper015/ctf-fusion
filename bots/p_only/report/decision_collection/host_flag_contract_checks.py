"""Regress the protocol status mistake without playing any games."""
import unittest
from host_flag_coverage import ground_flags


class FlagProtocol(unittest.TestCase):
    def test_home_and_dropped_are_ground_opportunities(self):
        self.assertEqual(ground_flags('F 0 home 9 10 -1 -1\nF 1 dropped 10 11 -1 43\n'), [(9, 10), (10, 11)])

    def test_carried_and_cooldown_have_no_ground_position(self):
        self.assertEqual(ground_flags('F 0 cooldown -1 -1 -1 57\nF 1 carried -1 -1 0 -1\n'), [])

    def test_invented_status_fails_instead_of_silently_losing_coverage(self):
        with self.assertRaises(ValueError):
            ground_flags('F 0 ground 9 10 -1 -1\n')


if __name__ == '__main__':
    unittest.main()
