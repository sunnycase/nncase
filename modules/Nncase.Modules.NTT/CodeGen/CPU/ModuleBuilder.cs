// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Targets;

namespace Nncase.CodeGen.NTT;

/// <summary>
/// K230CoreModule builder.
/// </summary>
public sealed class NTTModuleBuilder : IModuleBuilder
{
    private readonly SectionManager _sectionManager;
    private readonly BinaryWriter _rdataWriter;
    private readonly BinaryWriter[] _threadLocalRdataWriters;
    private readonly BinaryWriter[] _threadLocalCacheWriters;
    private readonly BinaryWriter[] _warpLocalRdataWriters;
    private readonly BinaryWriter[] _blockLocalRdataWriters;

    public NTTModuleBuilder(string moduleKind, CompileOptions options)
    {
        var targetOptions = (NTTTargetOptions)options.TargetOptions;
        var hierarchies = targetOptions.Hierarchies[0];
        ModuleKind = moduleKind;
        _sectionManager = new();
        _rdataWriter = _sectionManager.GetWriter(WellknownSectionNames.Rdata);

        var shardCount = TensorUtilities.GetProduct(hierarchies);
        _threadLocalRdataWriters = new BinaryWriter[shardCount];
        _threadLocalCacheWriters = new BinaryWriter[shardCount];
        for (int i = 0; i < shardCount; i++)
        {
            _threadLocalRdataWriters[i] = _sectionManager.GetWriter(WellknownSectionNames.ThreadLocalRdata, i);
            _threadLocalCacheWriters[i] = _sectionManager.GetWriter(WellknownSectionNames.ThreadLocalCache, i);
        }

        var isCUDA = ModuleKind == CUDATarget.Kind;
        var warpsCount = isCUDA ? shardCount / hierarchies[^1] : 0;
        _warpLocalRdataWriters = new BinaryWriter[warpsCount];
        for (int i = 0; i < _warpLocalRdataWriters.Length; i++)
        {
            _warpLocalRdataWriters[i] = _sectionManager.GetWriter(WellknownSectionNames.WarpLocalRdata, i);
        }

        var blocksCount = isCUDA ? (hierarchies.Length > 1 ? warpsCount / hierarchies[^2] : 1) : shardCount / hierarchies[^1];
        _blockLocalRdataWriters = new BinaryWriter[blocksCount];
        for (int i = 0; i < _blockLocalRdataWriters.Length; i++)
        {
            _blockLocalRdataWriters[i] = _sectionManager.GetWriter(WellknownSectionNames.BlockLocalRdata, i);
        }

        CompileOptions = options;
    }

    public CompileOptions CompileOptions { get; }

    /// <inheritdoc/>
    public string ModuleKind { get; }

    /// <inheritdoc/>
    public ILinkableModule Build(IReadOnlyList<BaseFunction> functions)
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;

        // 1. write the module header
        using (var writer = _sectionManager.GetWriter(LinkedModule.ModuleHeaderSectionName))
        {
            var header = default(ModuleDescHeader);
            var hasWarp = targetOptions.HierarchyNames.Contains('w', StringComparison.Ordinal);
            header.ThreadDim = (uint)targetOptions.Hierarchies[0][^1];
            if (hasWarp)
            {
                header.WarpDim = targetOptions.Hierarchies[0].Length < 2 ? 1 : (uint)targetOptions.Hierarchies[0][^2];
                header.BlockDim = targetOptions.Hierarchies[0].Length < 3 ? 1 : (uint)targetOptions.Hierarchies[0][^3];
                header.ChipDim = targetOptions.Hierarchies[0].Length < 4 ? 1 : (uint)targetOptions.Hierarchies[0][^4];
            }
            else
            {
                header.WarpDim = 1;
                header.BlockDim = targetOptions.Hierarchies[0].Length < 2 ? 1 : (uint)targetOptions.Hierarchies[0][^2];
                header.ChipDim = targetOptions.Hierarchies[0].Length < 3 ? 1 : (uint)targetOptions.Hierarchies[0][^3];
            }

            writer.Write(ref header);

            // cache offsets.
            if (targetOptions.MemoryCapacities.Length - 1 > 3)
            {
                throw new NotSupportedException("Only support up to 3 level cache.");
            }

            writer.Write(targetOptions.MemoryCapacities.Length - 1);
            int cacheStart = 0;
            for (int i = 0; i < targetOptions.MemoryCapacities.Length - 1; i++)
            {
                writer.Write(cacheStart);
                writer.Write(targetOptions.MemoryCapacities[i]);
                cacheStart += targetOptions.MemoryCapacities[i];
            }
        }

        var linkableFunctions = functions.OfType<BaseFunction>().Select((f, i) => new FunctionBuilder((uint)i, _rdataWriter, _threadLocalRdataWriters, _warpLocalRdataWriters, _blockLocalRdataWriters, (Targets.NTTTargetOptions)CompileOptions.TargetOptions).Build(f)).ToArray();
        _rdataWriter.Flush();
        var threadLocalRdataContents = Enumerable.Range(0, _threadLocalRdataWriters.Length).Select(i =>
        {
            _threadLocalRdataWriters[i].Flush();
            return _sectionManager.GetContent(WellknownSectionNames.ThreadLocalRdata, i)!;
        }).ToArray();

        var threadLocalCacheContents = Enumerable.Range(0, _threadLocalCacheWriters.Length).Select(i =>
        {
            for (int j = 0; j < targetOptions.MemoryCapacities.Length - 1; j++)
            {
                var zeros = new byte[targetOptions.MemoryCapacities[j]];
                _threadLocalCacheWriters[i].Write(zeros);
            }

            _threadLocalCacheWriters[i].Flush();
            return _sectionManager.GetContent(WellknownSectionNames.ThreadLocalCache, i)!;
        }).ToArray();

        var warpLocalRdataContents = Enumerable.Range(0, _warpLocalRdataWriters.Length).Select(i =>
        {
            _warpLocalRdataWriters[i].Flush();
            return _sectionManager.GetContent(WellknownSectionNames.WarpLocalRdata, i)!;
        }).ToArray();

        var blockLocalRdataContents = Enumerable.Range(0, _blockLocalRdataWriters.Length).Select(i =>
        {
            _blockLocalRdataWriters[i].Flush();
            return _sectionManager.GetContent(WellknownSectionNames.BlockLocalRdata, i)!;
        }).ToArray();

        return new LinkableModule(ModuleKind, _sectionManager.GetContent(LinkedModule.ModuleHeaderSectionName)!, _sectionManager.GetContent(WellknownSectionNames.Rdata)!, threadLocalRdataContents, threadLocalCacheContents, warpLocalRdataContents, blockLocalRdataContents, linkableFunctions, CompileOptions);
    }
}
