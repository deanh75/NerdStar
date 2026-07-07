from __future__ import annotations
import collections.abc
import typing
__all__: list[str] = ['forEachNested', 'getSchema', 'getSize', 'getTypeName', 'pack', 'packArray', 'packInto', 'unpack', 'unpackArray']
def forEachNested(t: type, fn: collections.abc.Callable[[str, str], None]) -> None:
    """
    Call a function to retrieve the (type string, schema) for each nested struct
    """
def getSchema(t: type) -> str:
    """
    Retrieve schema for the specified type
    """
def getSize(t: type) -> int:
    """
    Returns the serialized size in bytes
    """
def getTypeName(t: type) -> str:
    """
    Retrieve the type name for the specified type
    """
def pack(v: typing.Any) -> bytes:
    """
    Serialize object into byte buffer
    """
def packArray(seq: collections.abc.Sequence) -> bytes:
    """
    Serialize objects into byte buffer
    """
def packInto(v: typing.Any, b: collections.abc.Buffer) -> None:
    """
    Serialize object into byte buffer. Buffer must be exact size.
    """
def unpack(t: type, b: collections.abc.Buffer) -> typing.Any:
    """
    Convert byte buffer into object of specified type. Buffer must be exact
    size.
    """
def unpackArray(t: type, b: collections.abc.Buffer) -> list[typing.Any]:
    """
    Convert byte buffer into list of objects of specified type. Buffer must be
    exact size.
    """
