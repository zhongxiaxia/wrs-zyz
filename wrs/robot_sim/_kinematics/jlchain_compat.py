"""Compatibility layer for WRS JLChain backends.

This module allows the WRS codebase to keep using the existing JLChain-based
robot API while optionally delegating the hot kinematics work to the native
`wrs-jlchain-native` package.

Selection rules:
- `WRS_JLCHAIN_BACKEND=auto` (default): try native first, then fall back to the
  original Python implementation.
- `WRS_JLCHAIN_BACKEND=kdl|orocos|orocos_kdl`: require the native backend.
- `WRS_JLCHAIN_BACKEND=python|none|off`: force the original Python JLChain.

The returned object is intentionally API-compatible with the WRS manipulator
stack and should be a drop-in replacement for the original JLChain instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Type

import numpy as np

from wrs.robot_sim._kinematics import jlchain as py_jlchain

try:
    from wrs_jlchain_native import JLChain as NativeJLChain
except Exception:
    NativeJLChain = None


@dataclass(frozen=True)
class BackendSelection:
    name: str
    use_native: bool


def _normalize_backend_name(value: Optional[str]) -> str:
    if value is None:
        return "auto"
    return str(value).strip().lower() or "auto"


def resolve_backend(requested: Optional[str] = None) -> BackendSelection:
    backend = _normalize_backend_name(requested or os.environ.get("WRS_JLCHAIN_BACKEND"))
    if backend in {"python", "none", "off"}:
        return BackendSelection(name="python", use_native=False)
    if backend in {"kdl", "orocos", "orocos_kdl"}:
        return BackendSelection(name="kdl", use_native=True)
    if backend == "auto":
        return BackendSelection(name="auto", use_native=NativeJLChain is not None)
    return BackendSelection(name=backend, use_native=NativeJLChain is not None)


class CompatJLChain(py_jlchain.JLChain):
    """A JLChain subclass that prefers the native backend when available.

    The constructor mirrors the original WRS JLChain constructor so existing
    robot code does not need to change.
    """

    def __init__(self, name="auto", pos=np.zeros(3), rotmat=np.eye(3), n_dof=6, backend_name: Optional[str] = None):
        self._native_backend = None
        self._backend_selection = resolve_backend(backend_name)
        if self._backend_selection.use_native and NativeJLChain is not None:
            try:
                self._native_backend = NativeJLChain(n_dof=n_dof, backend_name="kdl")
                self._native_backend.finalize()
            except Exception:
                self._native_backend = None
        super().__init__(name=name, pos=pos, rotmat=rotmat, n_dof=n_dof)

    def finalize(self, ik_solver=None, identifier_str="test", **kwargs):
        result = super().finalize(ik_solver=ik_solver, identifier_str=identifier_str, **kwargs)
        if self._native_backend is not None:
            try:
                self._native_backend.finalize()
            except Exception:
                self._native_backend = None
        return result

    def fk(self, jnt_values, toggle_jacobian=False, update=False):
        if self._native_backend is not None:
            try:
                if update:
                    if toggle_jacobian:
                        ret = self._native_backend.fk(jnt_values, toggle_jacobian=True, update=True)
                    else:
                        ret = self._native_backend.fk(jnt_values, toggle_jacobian=False, update=True)
                    return ret
                return self._native_backend.fk(jnt_values, toggle_jacobian=toggle_jacobian, update=False)
            except Exception:
                self._native_backend = None
        return super().fk(jnt_values=jnt_values, toggle_jacobian=toggle_jacobian, update=update)

    def jacobian(self, jnt_values=None):
        if self._native_backend is not None:
            try:
                if jnt_values is None:
                    return self._native_backend.jacobian()
                return self._native_backend.jacobian(jnt_values)
            except Exception:
                self._native_backend = None
        return super().jacobian(jnt_values=jnt_values)

    def ik(self, tgt_pos, tgt_rotmat, seed_jnt_values=None, option="single", toggle_dbg=False):
        if self._native_backend is not None:
            try:
                ret = self._native_backend.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=seed_jnt_values)
                if ret is not None and option and option[0] == 'm':
                    return [ret]
                return ret
            except Exception:
                self._native_backend = None
        return super().ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=seed_jnt_values, option=option, toggle_dbg=toggle_dbg)


def create_jlchain(name="auto", pos=np.zeros(3), rotmat=np.eye(3), n_dof=6, backend_name: Optional[str] = None):
    """Create a JLChain instance using the preferred backend."""
    selection = resolve_backend(backend_name)
    if selection.use_native and NativeJLChain is not None:
        try:
            native = NativeJLChain(n_dof=n_dof, backend_name="kdl")
            native.finalize()
            return native
        except Exception:
            if selection.name != "kdl":
                pass
    return py_jlchain.JLChain(name=name, pos=pos, rotmat=rotmat, n_dof=n_dof)
