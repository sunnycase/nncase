// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Nncase.Buffers;

internal sealed class ReinterpretedMemoryManager<TFrom, TTo> : MemoryManager<TTo>
    where TFrom : unmanaged, IEquatable<TFrom>
    where TTo : unmanaged, IEquatable<TTo>
{
    private readonly Memory<TFrom> _memory;
    private readonly int _length;

    public ReinterpretedMemoryManager(Memory<TFrom> memory)
    {
        var byteLength = checked((long)memory.Length * Unsafe.SizeOf<TFrom>());
        if (byteLength % Unsafe.SizeOf<TTo>() != 0)
        {
            throw new ArgumentException(
                $"Source byte length {byteLength} is not divisible by destination element size {Unsafe.SizeOf<TTo>()}.",
                nameof(memory));
        }

        _memory = memory;
        _length = checked((int)(byteLength / Unsafe.SizeOf<TTo>()));
    }

    public override Span<TTo> GetSpan() => MemoryMarshal.Cast<TFrom, TTo>(_memory.Span);

    public override unsafe MemoryHandle Pin(int elementIndex = 0)
    {
        if ((uint)elementIndex > (uint)_length)
        {
            throw new ArgumentOutOfRangeException(nameof(elementIndex));
        }

        var sourceHandle = _memory.Pin();
        var lease = new PinLease(sourceHandle);
        var byteOffset = checked((nint)elementIndex * Unsafe.SizeOf<TTo>());
        return new MemoryHandle((byte*)sourceHandle.Pointer + byteOffset, default, lease);
    }

    public override void Unpin()
    {
    }

    protected override void Dispose(bool disposing)
    {
    }

    private sealed class PinLease : IPinnable
    {
        private MemoryHandle _sourceHandle;

        public PinLease(MemoryHandle sourceHandle)
        {
            _sourceHandle = sourceHandle;
        }

        public MemoryHandle Pin(int elementIndex = 0) => throw new NotSupportedException();

        public void Unpin()
        {
            _sourceHandle.Dispose();
            _sourceHandle = default;
        }
    }
}
