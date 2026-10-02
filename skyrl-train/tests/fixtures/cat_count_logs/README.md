# CatCount native gate logs

These captures preserve native metric payloads, the telemetry run identity and
`Training done!` in their original order. ANSI colors and unrelated setup output
are excluded. Values and repeated payloads are unchanged.

The healthy Phase C run took 635.58 seconds. The completed-step divergence
control took 680 seconds; both ranks completed optimizer steps while rank 1
used unsynchronized gradients. It fails the DP weight checksum row.

- `healthy.txt`: source `phase-c-wrap-healthy-a1-native.log`, source SHA256 `15d3fe17d9719a36b221d1d0eb99af9ecd1626901eef4539c34faf29f42f4189`, trimmed SHA256 `b3471f92a0aedad04e3ad28c505bd654936a54dad9a06b14be852e0474a845ea`.
- `divergence.txt`: source `wrap-divergence-a2-complete-native.log`, source SHA256 `719c2d419fdf9aa20f1438faf9971e253f1d24a3a54726c81ccdac0b87bd3922`, trimmed SHA256 `168abb1c6b505c7e693faa3778efa82a496683a11d9ba1280638534d3ecd2033`.
