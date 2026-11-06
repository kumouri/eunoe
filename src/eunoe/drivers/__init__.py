"""Drivers: model, provider and billing. The only part of the kit that can spend.

Every paid call goes through `eunoe.drivers.base.SpendGate`, which refuses unless the run passed
`--allow-spend`. With `EUNOE_OFFLINE=1`, as in CI and the tests, nothing leaves the machine at all.
"""
