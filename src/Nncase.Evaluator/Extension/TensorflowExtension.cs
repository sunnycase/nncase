// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Google.Protobuf.WellKnownTypes;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.c_api;
using Shape = Tensorflow.Shape;

namespace Nncase.Evaluator;

/// <summary>
/// TensorFlow extension.
/// </summary>
public static class TensorflowExtension
{
    private static unsafe readonly DeallocatorArgs* _deallocatorArgs;

    static unsafe TensorflowExtension()
    {
        _deallocatorArgs = (DeallocatorArgs*)Marshal.AllocHGlobal(Marshal.SizeOf<DeallocatorArgs>());
        *_deallocatorArgs = new DeallocatorArgs
        {
            gc_handle = IntPtr.Zero,
            deallocator_called = false,
        };
    }

    /// <summary>
    /// Convert <see cref="Tensorflow.Tensor"/> to <see cref="Tensor"/>.
    /// </summary>
    /// <param name="tensor">Tensorflow tensor.</param>
    /// <returns>Converted tensor.</returns>
    public static Tensor ToTensor(this Tensorflow.Tensor tensor)
    {
        var mmgr = new TFTensorMemoryManager(tensor.Handle);
        return Tensor.FromBytes(ToDataType(tensor.dtype), mmgr.Memory, tensor.shape.as_int_list());
    }

    /// <summary>
    /// Convert <see cref="Tensorflow.Tensor"/> to <see cref="TensorValue"/>.
    /// </summary>
    /// <param name="tensor">Tensorflow tensor.</param>
    /// <returns>Converted value.</returns>
    public static TensorValue ToValue(this Tensorflow.Tensor tensor)
    {
        return tensor.ToTensor();
    }

    /// <summary>
    /// Convert <see cref="Tensor"/> to <see cref="Tensorflow.Tensor"/>.
    /// </summary>
    /// <param name="tensor">Tensor.</param>
    /// <returns>Converted torch tensor.</returns>
    public static unsafe Tensorflow.Tensor ToTFTensor(this Tensor tensor)
    {
        // TODO: Fix null reference exception
#if false
        var dtype = tensor.ElementType.ToTFType();
        var bufferHandle = tensor.PinBuffer();
        c_api.Deallocator deallocator = (IntPtr data, IntPtr size, ref c_api.DeallocatorArgs args) =>
        {
            bufferHandle.Dispose();
        };
        var handle = c_api.TF_NewTensor(dtype, tensor.Dimensions.ToLongs(), tensor.Rank, (IntPtr)bufferHandle.Pointer, (ulong)tensor.Length * (ulong)tensor.ElementType.SizeInBytes, deallocator, (IntPtr)_deallocatorArgs);
        return new Tensorflow.Tensor(handle);
#else
        return new NDArray(tensor.BytesBuffer.ToArray(), tensor.Dimensions.ToArray(), tensor.ElementType.ToTFType());
#endif
    }

    public static TF_DataType ToTFType(this DataType dt) => _dataTypesToTorchType[dt];

    public static DataType ToDataType(this TF_DataType dt) => _TorchTypeTodataTypes[dt];

    private static readonly Dictionary<DataType, TF_DataType> _dataTypesToTorchType = new()
    {
        { DataTypes.Boolean, TF_DataType.TF_BOOL },
        { DataTypes.Int8, TF_DataType.TF_INT8 },
        { DataTypes.Int16, TF_DataType.TF_INT16 },
        { DataTypes.Int32, TF_DataType.TF_INT32 },
        { DataTypes.Int64, TF_DataType.TF_INT64 },
        { DataTypes.UInt8, TF_DataType.TF_UINT8 },
        { DataTypes.Float16, TF_DataType.TF_HALF },
        { DataTypes.BFloat16, TF_DataType.TF_BFLOAT16 },
        { DataTypes.Float32, TF_DataType.TF_FLOAT },
        { DataTypes.Float64, TF_DataType.TF_DOUBLE },
    };

    private static readonly Dictionary<TF_DataType, DataType> _TorchTypeTodataTypes = new()
    {
        { TF_DataType.TF_BOOL, DataTypes.Boolean },
        { TF_DataType.TF_INT8, DataTypes.Int8 },
        { TF_DataType.TF_INT16, DataTypes.Int16 },
        { TF_DataType.TF_INT32, DataTypes.Int32 },
        { TF_DataType.TF_INT64, DataTypes.Int64 },
        { TF_DataType.TF_UINT8, DataTypes.UInt8 },
        { TF_DataType.TF_HALF, DataTypes.Float16 },
        { TF_DataType.TF_BFLOAT16, DataTypes.BFloat16 },
        { TF_DataType.TF_FLOAT, DataTypes.Float32 },
        { TF_DataType.TF_DOUBLE, DataTypes.Float64 },
    };

    private sealed class TFTensorMemoryManager : MemoryManager<byte>
    {
        private readonly SafeTensorHandle _tensor;

        public TFTensorMemoryManager(SafeTensorHandle tensor)
        {
            bool success = false;
            tensor.DangerousAddRef(ref success);
            if (!success)
            {
                throw new InvalidOperationException("Add ref failed.");
            }

            _tensor = tensor;
        }

        public unsafe override Span<byte> GetSpan() =>
            new Span<byte>(c_api.TF_TensorData(_tensor).ToPointer(), (int)c_api.TF_TensorByteSize(_tensor));

        public unsafe override MemoryHandle Pin(int elementIndex = 0)
        {
            var basePtr = c_api.TF_TensorData(_tensor).ToPointer();
            var pointer = Unsafe.Add<byte>(basePtr, elementIndex);
            return new MemoryHandle(pointer, pinnable: this);
        }

        public override void Unpin()
        {
        }

        protected override void Dispose(bool disposing)
        {
            _tensor.DangerousRelease();
        }
    }
}
