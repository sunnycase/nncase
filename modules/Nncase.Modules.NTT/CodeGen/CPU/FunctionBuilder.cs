// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CodeGen.NTT;
using Nncase.IR;
using Nncase.IR.Shapes;
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
    private readonly IReadOnlyList<BinaryWriter> _blockLocalRdataWriters;
    private readonly ulong _chipLocalRdataBase;
    private readonly ulong _mergedRdataPoolSize;

    public FunctionBuilder(
        uint id,
        BinaryWriter rdataWriter,
        IReadOnlyList<BinaryWriter> blockLocalRdataWriters,
        Targets.NTTTargetOptions targetOptions,
        ulong chipLocalRdataBase,
        ulong mergedRdataPoolSize)
    {
        _id = id;
        _sectionManager = new();
        _textWriter = _sectionManager.GetWriter(WellknownSectionNames.Text);
        _rdataWriter = rdataWriter;
        _blockLocalRdataWriters = blockLocalRdataWriters;
        _chipLocalRdataBase = chipLocalRdataBase;
        _mergedRdataPoolSize = mergedRdataPoolSize;
        TargetOptions = targetOptions;
    }

    public NTTTargetOptions TargetOptions { get; }

    public unsafe ILinkableFunction Build(BaseFunction baseFunc)
    {
        if (baseFunc is TIR.PrimFunction primFunc)
        {
            if (primFunc.Role != FunctionRole.ScheduledRegion)
            {
                // 1. write the rdata
                SerializeGlobalRdata(primFunc.SchedResult.Rdatas, 0, "rdata");
                SerializeGlobalRdata(primFunc.SchedResult.ChipLocalRdatas, _chipLocalRdataBase, "chip-local rdata");

                // 2. write the local rdatas
                var blockLocalRdataPoolSize = SerializeLocalRdata(primFunc.SchedResult.BlockLocalRdatas, _blockLocalRdataWriters, "b");

                // 3. build function.
                var visitor = new KernelCSourceConvertVisitor(TargetOptions, _chipLocalRdataBase);
                visitor.Visit(primFunc);
                var functionCSource = visitor.GetCSource();

                // 4. write the kernel desc
                using (var writer = _sectionManager.GetWriter(LinkableKernelFunction.KernelHeaderSectionName))
                {
                    var entryAbi = KernelEntryAbiLayout.Create(primFunc);
                    var header = default(KernelDescHeader);
                    header.OutputAlign = checked((uint)entryAbi.OutputAlignment);
                    header.LocalDataAlign = (uint)primFunc.SchedResult.DataAlign;
                    header.OutputPoolSize = entryAbi.OutputPoolSize;
                    header.LocalDataPoolSize = primFunc.SchedResult.DataUsage;
                    header.BlockLocalDataPoolSize = primFunc.SchedResult.BlockLocalDataPoolSize;
                    writer.Write(ref header);
                }

                var memoryPoolDesc = new KernelMemoryPoolDesc(
                    _mergedRdataPoolSize,
                    blockLocalRdataPoolSize);
                var kernelDescSection = new LinkedSection(_sectionManager.GetContent(LinkableKernelFunction.KernelHeaderSectionName)!, ".desc", 0, 8, (uint)sizeof(KernelDescHeader));
                return new LinkableKernelFunction(_id, primFunc, functionCSource, memoryPoolDesc, _sectionManager.GetContent(WellknownSectionNames.Text)!, kernelDescSection);
            }
            else
            {
                var visitor = new DeviceCSourceConvertVisitor(TargetOptions);
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

    private void SerializeGlobalRdata(
        IReadOnlyDictionary<Const, ValueRange<ulong>> rdatas,
        ulong baseOffset,
        string poolName)
    {
        foreach (var (@const, range) in rdatas)
        {
            var tensor = ((TensorConst)@const).Value;
            var size = range.Max - range.Min;
            var tensorSize = checked((ulong)tensor.Length * (ulong)tensor.ElementType.SizeInBytes);
            if (tensorSize != size)
            {
                throw new InvalidDataException(
                    $"The {poolName} allocation for {@const} is {size} bytes, but its tensor payload is {tensorSize} bytes.");
            }

            _rdataWriter.Position(checked((long)(baseOffset + range.Min)));
            tensor.Serialize(_rdataWriter.BaseStream);
        }
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
            var dividedDims = DistributedUtility.GetDividedTensorType(distributedType, DistributedUtility.DivideFlags.MaxShape).Shape.ToValueArray();
            var localStrides = TensorUtilities.GetDefaultStrides(dividedDims);
            for (int i = 0; i < localRdataWriters.Count; i++)
            {
                var localRdataWriter = localRdataWriters[i];
                var shardIndex = GetScopedShardIndex(i, scopeName);
                (var localOffsetExpr, var localShapeExpr) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex, DistributedUtility.DivideFlags.MaxShape);
                var localOffset = new RankedShape(localOffsetExpr).ToValueArray();
                var localShape = new RankedShape(localShapeExpr).ToValueArray();
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
        var scopeIndex = GetScopeIndex(scopeName, hierarchies.Length);
        if (scopeIndex < 0)
        {
            return DistributedUtility.GetUnraveledIndex(writerIndex, hierarchies);
        }

        var scopedHierarchies = hierarchies[..(scopeIndex + 1)];
        return DistributedUtility.GetUnraveledIndex(writerIndex, scopedHierarchies)
            .Concat(Enumerable.Repeat(0, hierarchies.Length - scopedHierarchies.Length))
            .ToArray();
    }

    private int GetScopeIndex(string scopeName, int rank)
    {
        if (scopeName.Length == 1 && scopeName[0] is 'c' or 'd' or 'b')
        {
            var levels = Placement.NormalizeHierarchyLevels(TargetOptions.HierarchyLevels, TargetOptions.HierarchyNames, rank);
            return levels.LastIndexOf(scopeName[0]);
        }

        return -1;
    }
}
