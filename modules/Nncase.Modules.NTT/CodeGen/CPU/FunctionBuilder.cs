// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CodeGen.NTT;
using Nncase.IR;
using Nncase.Targets;
using Nncase.Utilities;

namespace Nncase.CodeGen.NTT;

/// <summary>
/// StackVM function builder.
/// </summary>
internal class FunctionBuilder
{
    private readonly uint _id;
    private readonly SectionManager _sectionManager;
    private readonly BinaryWriter _textWriter;
    private readonly BinaryWriter _rdataWriter;
    private readonly IReadOnlyList<BinaryWriter> _threadLocalRdataWriters;
    private readonly IReadOnlyList<BinaryWriter> _warpLocalRdataWriters;
    private readonly IReadOnlyList<BinaryWriter> _blockLocalRdataWriters;

    public FunctionBuilder(uint id, BinaryWriter rdataWriter, IReadOnlyList<BinaryWriter> threadLocalRdataWriters, IReadOnlyList<BinaryWriter> warpLocalRdataWriters, IReadOnlyList<BinaryWriter> blockLocalRdataWriters, Targets.NTTTargetOptions targetOptions)
    {
        _id = id;
        _sectionManager = new();
        _textWriter = _sectionManager.GetWriter(WellknownSectionNames.Text);
        _rdataWriter = rdataWriter;
        _threadLocalRdataWriters = threadLocalRdataWriters;
        _warpLocalRdataWriters = warpLocalRdataWriters;
        _blockLocalRdataWriters = blockLocalRdataWriters;
        TargetOptions = targetOptions;
    }

    public NTTTargetOptions TargetOptions { get; }

    public unsafe ILinkableFunction Build(BaseFunction baseFunc)
    {
        if (baseFunc is TIR.PrimFunction primFunc)
        {
            if (!primFunc.Name.Contains("device_func", StringComparison.Ordinal))
            {
                // 1. write the rdata
                ulong rdataPoolSize = ulong.MinValue;
                foreach (var (@const, range) in primFunc.SchedResult.Rdatas)
                {
                    var tensor = ((TensorConst)@const).Value;
                    var size = range.Max - range.Min;
                    rdataPoolSize = System.Math.Max(range.Max, rdataPoolSize);
                    if ((ulong)tensor.Length * (ulong)tensor.ElementType.SizeInBytes != size)
                    {
                        throw new InvalidDataException("The Buffer Size Not Equal!");
                    }

                    _rdataWriter.Position(checked((long)range.Min));
                    tensor.Serialize(_rdataWriter.BaseStream);
                }

                // 2. write the local rdatas
                var threadLocalRdataPoolSize = SerializeLocalRdata(primFunc.SchedResult.ThreadLocalRdatas, _threadLocalRdataWriters, "t");
                var warpLocalRdataPoolSize = SerializeLocalRdata(primFunc.SchedResult.WarpLocalRdatas, _warpLocalRdataWriters, "w");
                var blockLocalRdataPoolSize = SerializeLocalRdata(primFunc.SchedResult.BlockLocalRdatas, _blockLocalRdataWriters, "b");

                // 3. build function.
                var visitor = new KernelCSourceConvertVisitor(TargetOptions);
                visitor.Visit(primFunc);
                var functionCSource = visitor.GetCSource();

                // 4. write the kernel desc
                using (var writer = _sectionManager.GetWriter(LinkableKernelFunction.KernelHeaderSectionName))
                {
                    var header = default(KernelDescHeader);
                    header.OutputAlign = (uint)primFunc.SchedResult.OutputAlign;
                    header.LocalDataAlign = (uint)primFunc.SchedResult.DataAlign;
                    header.OutputPoolSize = primFunc.SchedResult.OutputUsage;
                    header.LocalDataPoolSize = primFunc.SchedResult.DataUsage;
                    header.WarpLocalDataPoolSize = primFunc.SchedResult.WarpLocalDataPoolSize;
                    header.BlockLocalDataPoolSize = primFunc.SchedResult.BlockLocalDataPoolSize;
                    writer.Write(ref header);
                }

                var memoryPoolDesc = new KernelMemoryPoolDesc(
                    rdataPoolSize,
                    threadLocalRdataPoolSize,
                    warpLocalRdataPoolSize,
                    blockLocalRdataPoolSize);
                var kernelDescSection = new LinkedSection(_sectionManager.GetContent(LinkableKernelFunction.KernelHeaderSectionName)!, ".desc", 0, 8, (uint)sizeof(KernelDescHeader));
                return new LinkableKernelFunction(_id, primFunc, functionCSource, memoryPoolDesc, _sectionManager.GetContent(WellknownSectionNames.Text)!, kernelDescSection);
            }
            else
            {
                var visitor = new DeviceCSourceConvertVisitor();
                visitor.Visit(primFunc);
                var header = visitor.GetHeader();
                return new LinkableDeviceFunction(_id, primFunc, header, _sectionManager.GetContent(WellknownSectionNames.Text)!);
            }
        }
        else if (baseFunc is Fusion fusion)
        {
            var visitor = new LambdaCSourceConvertVisitor();
            visitor.Visit(fusion);
            var header = visitor.GetHeader();
            return new LinkableLambdaFunction(_id, fusion, header, _sectionManager.GetContent(WellknownSectionNames.Text)!);
        }

        throw new NotSupportedException($"the {baseFunc.GetType()} {baseFunc.Name} is notsupport for codegen!");
    }

    private ulong SerializeLocalRdata(IReadOnlyDictionary<Const, ValueRange<ulong>> localRdatas, IReadOnlyList<BinaryWriter> localRdataWriters, string scopeName)
    {
        ulong localRdataPoolSize = ulong.MinValue;
        foreach (var (@const, range) in localRdatas)
        {
            var tensor = ((TensorConst)@const).Value;
            var distributedType = (DistributedType)@const.CheckedType;
            var size = range.Max - range.Min;
            localRdataPoolSize = System.Math.Max(range.Max, localRdataPoolSize);
            var dividedDims = DistributedUtility.GetDividedTensorType(distributedType).Shape.ToValueArray();
            var localStrides = TensorUtilities.GetDefaultStrides(dividedDims);
            for (int i = 0; i < localRdataWriters.Count; i++)
            {
                var localRdataWriter = localRdataWriters[i];
                var shardIndex = GetScopedShardIndex(i, scopeName);
                (var localOffset, var localShape) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
                var linearOffset = TensorUtilities.GetLinearOffset(tensor.Strides, localOffset);

                if ((ulong)TensorUtilities.GetProduct(localShape) * (ulong)tensor.ElementType.SizeInBytes > size)
                {
                    throw new InvalidDataException("The Buffer Size Not Equal!");
                }

                localRdataWriter.Position(checked((long)range.Min));
                tensor.Serialize(localRdataWriter.BaseStream, linearOffset, localShape, localStrides);
            }
        }

        return localRdataPoolSize;
    }

    private int[] GetScopedShardIndex(int writerIndex, string scopeName)
    {
        var hierarchies = TargetOptions.Hierarchies[0];
        var scopeIndex = TargetOptions.HierarchyNames.IndexOf(scopeName, StringComparison.Ordinal);
        if (scopeIndex < 0)
        {
            return DistributedUtility.GetUnraveledIndex(writerIndex, hierarchies);
        }

        var scopedHierarchies = hierarchies[..(scopeIndex + 1)];
        return DistributedUtility.GetUnraveledIndex(writerIndex, scopedHierarchies)
            .Concat(Enumerable.Repeat(0, hierarchies.Length - scopedHierarchies.Length))
            .ToArray();
    }
}
